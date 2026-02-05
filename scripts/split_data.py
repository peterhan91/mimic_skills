"""
Split MIMIC-CDM-IV pkl files into train (10) + test (100) for EvoTest experiments.

Train set (10 admissions): Used during EvoTest evolution episodes.
  - The Evolver Agent iterates on these, evolving the config across episodes.
  - Small by design: forces the evolved skill to generalize, not memorize.

Test set (100 admissions): Held-out evaluation. Never seen during evolution.
  - Used for final comparison: Static vs Reflexion vs EvoTest.
  - Mirrors the paper's evaluation protocol.

Remaining admissions: Reserved for Option C synthesis (sampling discharge summaries)
  or for future use as additional evolution batches.

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --seed 42 --n_train 10 --n_test 100
"""

import argparse
import os
import pickle
import random
from pathlib import Path


PATHOLOGIES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

BASE_MIMIC = Path(__file__).resolve().parent.parent / "MIMIC-CDM-IV"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_splits"


def load_pkl(pathology: str) -> dict:
    path = BASE_MIMIC / f"{pathology}_hadm_info_first_diag.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def split_and_save(
    pathology: str,
    n_train: int = 10,
    n_test: int = 100,
    seed: int = 42,
) -> None:
    data = load_pkl(pathology)
    hadm_ids = list(data.keys())
    total = len(hadm_ids)

    assert total >= n_train + n_test, (
        f"{pathology}: only {total} admissions, need {n_train + n_test}"
    )

    rng = random.Random(seed)
    rng.shuffle(hadm_ids)

    train_ids = hadm_ids[:n_train]
    test_ids = hadm_ids[n_train : n_train + n_test]
    remaining_ids = hadm_ids[n_train + n_test :]

    # Build sub-dicts
    train_data = {k: data[k] for k in train_ids}
    test_data = {k: data[k] for k in test_ids}
    remaining_data = {k: data[k] for k in remaining_ids}

    # Save
    out = OUTPUT_DIR / pathology
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(out / "test.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open(out / "remaining.pkl", "wb") as f:
        pickle.dump(remaining_data, f)

    # Also save just the IDs as text for easy inspection
    for name, ids in [("train", train_ids), ("test", test_ids), ("remaining", remaining_ids)]:
        with open(out / f"{name}_hadm_ids.txt", "w") as f:
            for hid in ids:
                f.write(f"{hid}\n")

    print(f"{pathology}:")
    print(f"  Total admissions: {total}")
    print(f"  Train: {len(train_data)} → {out / 'train.pkl'}")
    print(f"  Test:  {len(test_data)} → {out / 'test.pkl'}")
    print(f"  Remaining: {len(remaining_data)} → {out / 'remaining.pkl'}")

    # Sanity check: verify no overlap
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test overlap!"
    assert len(set(train_ids) & set(remaining_ids)) == 0, "Train/remaining overlap!"

    # Print a sample admission from train for verification
    sample_id = train_ids[0]
    sample = train_data[sample_id]
    hist = sample.get("Patient History", "")[:200]
    diag = sample.get("Discharge Diagnosis", "N/A")
    print(f"  Sample train admission {sample_id}:")
    print(f"    Diagnosis: {diag}")
    print(f"    History: {hist}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="Split MIMIC-CDM-IV data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()

    print(f"Splitting with seed={args.seed}, n_train={args.n_train}, n_test={args.n_test}")
    print(f"Source: {BASE_MIMIC}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for pathology in PATHOLOGIES:
        split_and_save(pathology, args.n_train, args.n_test, args.seed)

    print("Done. Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    for p in PATHOLOGIES:
        print(f"    {p}/")
        print(f"      train.pkl          (10 admissions — EvoTest evolution batch)")
        print(f"      test.pkl           (100 admissions — held-out evaluation)")
        print(f"      remaining.pkl      (rest — Option C synthesis pool)")
        print(f"      train_hadm_ids.txt")
        print(f"      test_hadm_ids.txt")
        print(f"      remaining_hadm_ids.txt")


if __name__ == "__main__":
    main()
