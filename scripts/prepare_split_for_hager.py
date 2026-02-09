"""
Copies a split pkl as {pathology}_hadm_info_first_diag.pkl so Hager's
framework can load it via `load_hadm_from_file(f"{pathology}_hadm_info_first_diag")`.

This avoids modifying run.py. Just point base_mimic at the output directory.

Usage:
  python scripts/prepare_split_for_hager.py --pathology appendicitis --split train
  python scripts/prepare_split_for_hager.py --pathology appendicitis --split test
  python scripts/prepare_split_for_hager.py --all --split train
"""

import argparse
import shutil
from pathlib import Path

PATHOLOGIES = [
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis",
    "cholangitis", "bowel_obstruction", "pyelonephritis",
]
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data_splits"


def prepare(pathology: str, split: str) -> None:
    src = SPLITS_DIR / pathology / f"{split}.pkl"
    dst = SPLITS_DIR / pathology / f"{pathology}_hadm_info_first_diag.pkl"

    if not src.exists():
        print(f"ERROR: {src} not found. Run split_data.py first.")
        return

    shutil.copy2(src, dst)
    print(f"{pathology}: {split}.pkl → {dst.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathology", type=str, help="Single pathology")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test", "remaining"])
    parser.add_argument("--all", action="store_true", help="Prepare all 4 pathologies")
    args = parser.parse_args()

    if args.all:
        for p in PATHOLOGIES:
            prepare(p, args.split)
    elif args.pathology:
        prepare(args.pathology, args.split)
    else:
        parser.error("Specify --pathology or --all")

    print(f"\nHager's run.py can now load with: base_mimic=data_splits/{{pathology}}")


if __name__ == "__main__":
    main()
