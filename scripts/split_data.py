"""
Split MIMIC-CDM pkl files into train + test + remaining for EvoTest experiments.

**Patient-level split**: Groups admissions by subject_id so that all admissions
for the same patient land in the same split. This prevents data leakage where
the model trains on one visit and is tested on another visit from the same patient.

Train set (~10 patients): Used during EvoTest evolution episodes.
Test set (~100 patients): Held-out evaluation. Never seen during evolution.
Remaining: Reserved for Option C synthesis or future evolution batches.

Supports two conditions:
  - abdominal (default): 7 pathologies from CDM-IV
  - chest: 5 cardiac pathologies from CDM-III (4) + CDM-IV (PE)

Note: CDM-III hadm_ids are NOT in MIMIC-IV admissions.csv — we use MIMIC-III
ADMISSIONS.csv for those pathologies.

Note: Because we split by patient (not admission), the exact number of admissions
per split may differ slightly from n_train/n_test when patients have multiple
admissions. The script targets n_train patients and n_test patients.

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --condition chest
    python scripts/split_data.py --seed 42 --n_train 10 --n_test 100
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


ABDOMINAL_PATHOLOGIES = [
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis",
    "cholangitis", "bowel_obstruction", "pyelonephritis",
]

CHEST_PATHOLOGIES = [
    "myocardial_infarction", "pulmonary_embolism", "congestive_heart_failure",
    "aortic_stenosis", "mitral_regurgitation",
]

# Pathologies from CDM-III (use MIMIC-III admissions for hadm→subject mapping)
CDM_III_PATHOLOGIES = {
    "myocardial_infarction", "congestive_heart_failure",
    "aortic_stenosis", "mitral_regurgitation",
}

# Small pathologies: reduced test set so we have enough for train
SMALL_PATHOLOGY_SIZES = {
    "congestive_heart_failure": {"n_test": 60},
    "aortic_stenosis": {"n_test": 35},
    "mitral_regurgitation": {"n_test": 15},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CDM_III = PROJECT_ROOT / "MIMIC-CDM-III"
BASE_CDM_IV = PROJECT_ROOT / "MIMIC-CDM-IV"
OUTPUT_DIR = PROJECT_ROOT / "data_splits"

ADMISSIONS_CSV_IV = Path(os.environ.get(
    "MIMIC_IV_ADMISSIONS_CSV",
    "/Users/tianyuhan/Documents/data/mimiciv/3.1/hosp/admissions.csv",
))
ADMISSIONS_CSV_III = Path(os.environ.get(
    "MIMIC_III_ADMISSIONS_CSV",
    "/Users/tianyuhan/Documents/data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv",
))


def load_hadm_to_subject_iv() -> dict:
    """Load hadm_id -> subject_id mapping from MIMIC-IV admissions table."""
    df = pd.read_csv(ADMISSIONS_CSV_IV, usecols=["hadm_id", "subject_id"])
    return dict(zip(df["hadm_id"], df["subject_id"]))


def load_hadm_to_subject_iii() -> dict:
    """Load hadm_id -> subject_id mapping from MIMIC-III admissions table."""
    df = pd.read_csv(ADMISSIONS_CSV_III, usecols=["HADM_ID", "SUBJECT_ID"])
    return dict(zip(df["HADM_ID"], df["SUBJECT_ID"]))


def get_base_dir(pathology: str) -> Path:
    """Return the correct data directory for a pathology."""
    if pathology in CDM_III_PATHOLOGIES:
        return BASE_CDM_III
    return BASE_CDM_IV


def load_pkl(pathology: str) -> dict:
    base = get_base_dir(pathology)
    path = base / f"{pathology}_hadm_info_first_diag.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def split_and_save(
    pathology: str,
    hadm_to_subject: dict,
    n_train: int = 10,
    n_test: int = 100,
    seed: int = 42,
) -> None:
    # Adjust n_test for small pathologies
    if pathology in SMALL_PATHOLOGY_SIZES:
        n_test = SMALL_PATHOLOGY_SIZES[pathology]["n_test"]

    data = load_pkl(pathology)
    hadm_ids = list(data.keys())
    total = len(hadm_ids)

    # Group hadm_ids by subject_id (patient)
    subject_to_hadms = defaultdict(list)
    for hadm_id in hadm_ids:
        subject_id = hadm_to_subject.get(hadm_id)
        if subject_id is None:
            raise ValueError(f"{pathology}: hadm_id {hadm_id} not found in admissions.csv")
        subject_to_hadms[subject_id].append(hadm_id)

    unique_subjects = list(subject_to_hadms.keys())
    n_subjects = len(unique_subjects)
    n_multi = sum(1 for hadms in subject_to_hadms.values() if len(hadms) > 1)

    assert n_subjects >= n_train + n_test, (
        f"{pathology}: only {n_subjects} unique patients, need {n_train + n_test}"
    )

    # Shuffle patients, not admissions
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)

    train_subjects = unique_subjects[:n_train]
    test_subjects = unique_subjects[n_train : n_train + n_test]
    remaining_subjects = unique_subjects[n_train + n_test :]

    # Collect all hadm_ids for each split
    train_ids = [h for s in train_subjects for h in subject_to_hadms[s]]
    test_ids = [h for s in test_subjects for h in subject_to_hadms[s]]
    remaining_ids = [h for s in remaining_subjects for h in subject_to_hadms[s]]

    # Build sub-dicts
    train_data = {k: data[k] for k in train_ids}
    test_data = {k: data[k] for k in test_ids}
    remaining_data = {k: data[k] for k in remaining_ids}

    # Sanity: no patient-level leakage
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Patient train/test overlap!"
    assert len(set(train_subjects) & set(remaining_subjects)) == 0, "Patient train/remaining overlap!"
    # Also verify at hadm_id level
    assert len(set(train_ids) & set(test_ids)) == 0, "hadm_id train/test overlap!"
    assert len(set(train_ids) & set(remaining_ids)) == 0, "hadm_id train/remaining overlap!"
    assert len(train_ids) + len(test_ids) + len(remaining_ids) == total, "Lost admissions!"

    # Save
    out = OUTPUT_DIR / pathology
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(out / "test.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open(out / "remaining.pkl", "wb") as f:
        pickle.dump(remaining_data, f)

    # Save hadm_ids as text for inspection
    for name, ids in [("train", train_ids), ("test", test_ids), ("remaining", remaining_ids)]:
        with open(out / f"{name}_hadm_ids.txt", "w") as f:
            for hid in ids:
                f.write(f"{hid}\n")

    # Save subject_ids as text for verification
    for name, sids in [("train", train_subjects), ("test", test_subjects), ("remaining", remaining_subjects)]:
        with open(out / f"{name}_subject_ids.txt", "w") as f:
            for sid in sids:
                f.write(f"{sid}\n")

    print(f"{pathology}:")
    print(f"  Total: {total} admissions, {n_subjects} unique patients ({n_multi} with multiple admissions)")
    print(f"  Train: {len(train_subjects)} patients, {len(train_data)} admissions → {out / 'train.pkl'}")
    print(f"  Test:  {len(test_subjects)} patients, {len(test_data)} admissions → {out / 'test.pkl'}")
    print(f"  Remaining: {len(remaining_subjects)} patients, {len(remaining_data)} admissions → {out / 'remaining.pkl'}")

    # Print a sample
    sample_id = train_ids[0]
    sample = train_data[sample_id]
    hist = sample.get("Patient History", "")[:200]
    diag = sample.get("Discharge Diagnosis", "N/A")
    print(f"  Sample train admission {sample_id} (subject {hadm_to_subject[sample_id]}):")
    print(f"    Diagnosis: {diag}")
    print(f"    History: {hist}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="Split MIMIC-CDM data (patient-level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument(
        "--condition",
        choices=["abdominal", "chest"],
        default="abdominal",
        help="Condition domain: abdominal (7 GI pathologies) or chest (5 cardiac pathologies)",
    )
    args = parser.parse_args()

    if args.condition == "abdominal":
        pathologies = ABDOMINAL_PATHOLOGIES
    else:
        pathologies = CHEST_PATHOLOGIES

    print(f"Condition: {args.condition}")
    print(f"Splitting with seed={args.seed}, n_train={args.n_train} patients, n_test={args.n_test} patients")
    print(f"Pathologies: {pathologies}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load hadm→subject mappings
    # For chest condition, we may need both MIMIC-III and MIMIC-IV mappings
    hadm_to_subject = {}
    if args.condition == "chest":
        # CDM-III pathologies need MIMIC-III admissions
        needs_iii = any(p in CDM_III_PATHOLOGIES for p in pathologies)
        needs_iv = any(p not in CDM_III_PATHOLOGIES for p in pathologies)
        if needs_iii:
            print(f"Loading MIMIC-III admissions from {ADMISSIONS_CSV_III}...")
            hadm_to_subject.update(load_hadm_to_subject_iii())
            print(f"  {len(hadm_to_subject)} MIMIC-III admission mappings loaded")
        if needs_iv:
            print(f"Loading MIMIC-IV admissions from {ADMISSIONS_CSV_IV}...")
            iv_map = load_hadm_to_subject_iv()
            hadm_to_subject.update(iv_map)
            print(f"  {len(iv_map)} MIMIC-IV admission mappings loaded")
    else:
        print(f"Loading hadm_id -> subject_id mapping from {ADMISSIONS_CSV_IV}...")
        hadm_to_subject = load_hadm_to_subject_iv()
        print(f"  {len(hadm_to_subject)} admission mappings loaded")
    print()

    for pathology in pathologies:
        split_and_save(pathology, hadm_to_subject, args.n_train, args.n_test, args.seed)

    print("Done. Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    for p in pathologies:
        print(f"    {p}/")
        print(f"      train.pkl               (EvoTest evolution batch)")
        print(f"      test.pkl                (held-out evaluation)")
        print(f"      remaining.pkl           (Option C synthesis pool)")
        print(f"      train_hadm_ids.txt")
        print(f"      test_hadm_ids.txt")
        print(f"      remaining_hadm_ids.txt")
        print(f"      train_subject_ids.txt   (patient IDs for leakage verification)")
        print(f"      test_subject_ids.txt")
        print(f"      remaining_subject_ids.txt")


if __name__ == "__main__":
    main()
