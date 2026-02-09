"""
Sanitize skill files to remove diagnosis leakage before injection into the agent.

Hager's framework sanitizes patient data (Patient History, PE, Radiology) by
replacing disease names with "____". Our skill files must follow the same
convention — otherwise we re-introduce the exact leakage the framework removes.

What we mask (consistent with Hager's CreateDataset.py sanitize_list, plus
procedure names that map 1:1 to a specific disease):

  appendicitis:      "acute appendicitis", "appendicitis", "appendectomy"
  cholecystitis:     "acute cholecystitis", "cholecystitis", "cholecystostomy",
                     "cholecystectomy"
  pancreatitis:      "acute pancreatitis", "pancreatitis", "pancreatectomy"
  diverticulitis:    "acute diverticulitis", "diverticulitis"
  cholangitis:       "acute cholangitis", "ascending cholangitis", "cholangitis"
  bowel obstruction: "small bowel obstruction", "large bowel obstruction",
                     "bowel obstruction", "intestinal obstruction"
  pyelonephritis:    "acute pyelonephritis", "pyelonephritis"

What we do NOT mask (organ names / anatomical terms are needed for reasoning):
  "appendix", "gallbladder", "pancreas", "colon", "cecum", "diverticulum",
  "bile duct", "kidney", "ureter", "bowel", "intestine"

Usage:
  # Sanitize a skill file (prints to stdout)
  python scripts/sanitize_skill.py skills/v1/acute_abdominal_pain.md

  # Sanitize and write to output file
  python scripts/sanitize_skill.py skills/v1/acute_abdominal_pain.md -o out.md

  # Sanitize in-place
  python scripts/sanitize_skill.py skills/v1/acute_abdominal_pain.md --inplace

  # As a library
  from scripts.sanitize_skill import sanitize_skill_text
  clean_text = sanitize_skill_text(raw_skill_text)
"""

import argparse
import re
import sys
from pathlib import Path

# Mask string — same as Hager's framework (dataset.py line 734).
MASK = "____"

# Disease names and procedure names that directly reveal the diagnosis.
# Sorted longest-first so "acute appendicitis" matches before "appendicitis".
DISEASE_TERMS = sorted(
    [
        # Appendicitis
        "acute appendicitis",
        "appendicitis",
        "appendectomy",
        # Cholecystitis
        "acute cholecystitis",
        "cholecystitis",
        "cholecystectomy",
        "cholecystostomy",
        # Pancreatitis
        "acute pancreatitis",
        "pancreatitis",
        "pancreatectomy",
        # Diverticulitis
        "acute diverticulitis",
        "diverticulitis",
        # Cholangitis
        "acute cholangitis",
        "ascending cholangitis",
        "cholangitis",
        # Bowel obstruction
        "small bowel obstruction",
        "large bowel obstruction",
        "bowel obstruction",
        "intestinal obstruction",
        # Pyelonephritis
        "acute pyelonephritis",
        "pyelonephritis",
    ],
    key=len,
    reverse=True,
)


def sanitize_skill_text(text: str) -> str:
    """Replace disease/procedure names with '____' (Hager's mask).

    Processes longest terms first to avoid partial matches
    (e.g., "acute appendicitis" before "appendicitis").
    Case-insensitive.
    """
    result = text
    for term in DISEASE_TERMS:
        result = re.sub(re.escape(term), MASK, result, flags=re.IGNORECASE)
    return result


def report_changes(original: str, sanitized: str) -> list[str]:
    """Report what was changed for review."""
    changes = []
    for term in DISEASE_TERMS:
        count = len(re.findall(re.escape(term), original, re.IGNORECASE))
        if count:
            changes.append(f"  {term!r}: {count} occurrence(s) → {MASK}")
    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize skill files to remove diagnosis leakage"
    )
    parser.add_argument("skill_path", type=str, help="Path to skill .md file")
    parser.add_argument("-o", "--output", type=str, help="Output path (default: stdout)")
    parser.add_argument(
        "--inplace", action="store_true", help="Modify the file in place"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print what was changed to stderr"
    )
    args = parser.parse_args()

    skill_path = Path(args.skill_path)
    if not skill_path.exists():
        print(f"ERROR: {skill_path} not found", file=sys.stderr)
        sys.exit(1)

    original = skill_path.read_text()
    sanitized = sanitize_skill_text(original)

    if args.report or args.inplace:
        changes = report_changes(original, sanitized)
        if changes:
            print("Sanitization changes:", file=sys.stderr)
            for c in changes:
                print(c, file=sys.stderr)
        else:
            print("No disease terms found — skill is clean.", file=sys.stderr)

    if args.inplace:
        skill_path.write_text(sanitized)
        print(f"Sanitized in place: {skill_path}", file=sys.stderr)
    elif args.output:
        Path(args.output).write_text(sanitized)
        print(f"Wrote sanitized skill to: {args.output}", file=sys.stderr)
    else:
        print(sanitized)


if __name__ == "__main__":
    main()
