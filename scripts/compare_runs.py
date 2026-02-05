"""
Compare two trajectory JSONs (e.g., baseline vs evolved) and produce a
markdown report with per-metric deltas and per-patient changes.

Usage:
    python scripts/compare_runs.py \
        --baseline trajectories/baseline_cholecystitis_train10.json \
        --evolved trajectories/v1_cholecystitis_train10.json \
        --output comparisons/v1_vs_baseline_cholecystitis.md
"""

import argparse
import json
import os
from pathlib import Path


def load_trajectories(path):
    """Load trajectory JSON."""
    with open(path, "r") as f:
        return json.load(f)


def compare_runs(baseline_path, evolved_path, output_path):
    """Compare two runs and produce a markdown report."""
    baseline = load_trajectories(baseline_path)
    evolved = load_trajectories(evolved_path)

    pathology = baseline.get("pathology", "unknown")
    n_base = baseline["n_patients"]
    n_evol = evolved["n_patients"]

    # Build per-patient lookup by hadm_id
    base_by_id = {a["hadm_id"]: a for a in baseline["admissions"]}
    evol_by_id = {a["hadm_id"]: a for a in evolved["admissions"]}

    # Find common patients
    common_ids = sorted(set(base_by_id.keys()) & set(evol_by_id.keys()))

    lines = []
    lines.append(f"# Comparison: Evolved vs Baseline ({pathology}, {len(common_ids)} common patients)")
    lines.append("")
    lines.append(f"- **Baseline**: `{baseline_path}` ({n_base} patients)")
    lines.append(f"- **Evolved**: `{evolved_path}` ({n_evol} patients)")
    lines.append(f"- **Common patients**: {len(common_ids)}")
    lines.append("")

    # Aggregate metrics comparison
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Evolved | Delta |")
    lines.append("|---|---|---|---|")

    # Binary metrics
    binary_metrics = [
        "Diagnosis", "Gracious Diagnosis", "Physical Examination",
        "Late Physical Examination", "Action Parsing",
        "Treatment Parsing", "Diagnosis Parsing",
    ]
    # Value metrics
    value_metrics = ["Laboratory Tests", "Imaging", "Rounds", "Invalid Tools"]

    for key in binary_metrics + value_metrics:
        base_vals = [base_by_id[_id]["scores"].get(key, 0) for _id in common_ids]
        evol_vals = [evol_by_id[_id]["scores"].get(key, 0) for _id in common_ids]
        n = len(common_ids)

        if key in binary_metrics:
            base_count = sum(1 for v in base_vals if v > 0)
            evol_count = sum(1 for v in evol_vals if v > 0)
            base_pct = 100 * base_count / n if n else 0
            evol_pct = 100 * evol_count / n if n else 0
            delta = evol_pct - base_pct
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {key} | {base_count}/{n} ({base_pct:.0f}%) "
                f"| {evol_count}/{n} ({evol_pct:.0f}%) "
                f"| {sign}{delta:.0f}% |"
            )
        elif key == "Invalid Tools":
            base_total = sum(base_vals)
            evol_total = sum(evol_vals)
            delta = evol_total - base_total
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {key} | {base_total} | {evol_total} | {sign}{delta} |"
            )
        else:
            base_avg = sum(base_vals) / n if n else 0
            evol_avg = sum(evol_vals) / n if n else 0
            delta = evol_avg - base_avg
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {key} | {base_avg:.2f} | {evol_avg:.2f} | {sign}{delta:.2f} |"
            )

    lines.append("")

    # Per-patient diagnosis changes
    lines.append("## Per-Patient Diagnosis Changes")
    lines.append("")
    lines.append("| hadm_id | Baseline Dx | Evolved Dx | Changed? |")
    lines.append("|---|---|---|---|")

    n_fixed = 0
    n_broken = 0
    for _id in common_ids:
        base_dx = base_by_id[_id]["scores"].get("Diagnosis", 0)
        evol_dx = evol_by_id[_id]["scores"].get("Diagnosis", 0)
        base_answer = base_by_id[_id].get("answers", {}).get("Diagnosis", "?")
        evol_answer = evol_by_id[_id].get("answers", {}).get("Diagnosis", "?")

        if base_dx == 0 and evol_dx == 1:
            status = "FIXED"
            n_fixed += 1
        elif base_dx == 1 and evol_dx == 0:
            status = "BROKEN"
            n_broken += 1
        elif base_dx == 1 and evol_dx == 1:
            status = "ok"
        else:
            status = "still wrong"

        base_label = f"{'correct' if base_dx else 'WRONG'} ({base_answer[:30]})"
        evol_label = f"{'correct' if evol_dx else 'WRONG'} ({evol_answer[:30]})"
        lines.append(f"| {_id} | {base_label} | {evol_label} | {status} |")

    lines.append("")
    lines.append(f"**Fixed**: {n_fixed} patients | **Broken**: {n_broken} patients | **Net**: {n_fixed - n_broken:+d}")
    lines.append("")

    # Per-patient PE changes
    lines.append("## Per-Patient PE Ordering Changes")
    lines.append("")
    lines.append("| hadm_id | Baseline PE First | Evolved PE First | Changed? |")
    lines.append("|---|---|---|---|")

    n_pe_fixed = 0
    n_pe_broken = 0
    for _id in common_ids:
        base_pe = base_by_id[_id]["scores"].get("Physical Examination", 0)
        evol_pe = evol_by_id[_id]["scores"].get("Physical Examination", 0)

        if base_pe == 0 and evol_pe == 1:
            status = "FIXED"
            n_pe_fixed += 1
        elif base_pe == 1 and evol_pe == 0:
            status = "BROKEN"
            n_pe_broken += 1
        elif base_pe == 1 and evol_pe == 1:
            status = "ok"
        else:
            status = "still wrong"

        lines.append(f"| {_id} | {'yes' if base_pe else 'NO'} | {'yes' if evol_pe else 'NO'} | {status} |")

    lines.append("")
    lines.append(f"**Fixed**: {n_pe_fixed} patients | **Broken**: {n_pe_broken} patients | **Net**: {n_pe_fixed - n_pe_broken:+d}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    base_dx_count = sum(1 for _id in common_ids if base_by_id[_id]["scores"].get("Diagnosis", 0) > 0)
    evol_dx_count = sum(1 for _id in common_ids if evol_by_id[_id]["scores"].get("Diagnosis", 0) > 0)
    n = len(common_ids)
    lines.append(f"- Diagnosis accuracy: {base_dx_count}/{n} -> {evol_dx_count}/{n} ({evol_dx_count - base_dx_count:+d})")

    base_pe_count = sum(1 for _id in common_ids if base_by_id[_id]["scores"].get("Physical Examination", 0) > 0)
    evol_pe_count = sum(1 for _id in common_ids if evol_by_id[_id]["scores"].get("Physical Examination", 0) > 0)
    lines.append(f"- PE first: {base_pe_count}/{n} -> {evol_pe_count}/{n} ({evol_pe_count - base_pe_count:+d})")

    base_labs = sum(base_by_id[_id]["scores"].get("Laboratory Tests", 0) for _id in common_ids)
    evol_labs = sum(evol_by_id[_id]["scores"].get("Laboratory Tests", 0) for _id in common_ids)
    lines.append(f"- Lab score total: {base_labs} -> {evol_labs} ({evol_labs - base_labs:+d})")

    base_img = sum(base_by_id[_id]["scores"].get("Imaging", 0) for _id in common_ids)
    evol_img = sum(evol_by_id[_id]["scores"].get("Imaging", 0) for _id in common_ids)
    lines.append(f"- Imaging score total: {base_img} -> {evol_img} ({evol_img - base_img:+d})")

    report = "\n".join(lines)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Comparison report written to {output_path}")

    # Also print to stdout
    print(f"\n{report}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare two trajectory JSONs and produce a markdown report"
    )
    parser.add_argument(
        "--baseline", type=str, required=True,
        help="Path to baseline trajectory JSON"
    )
    parser.add_argument(
        "--evolved", type=str, required=True,
        help="Path to evolved trajectory JSON"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for markdown report"
    )
    args = parser.parse_args()

    compare_runs(args.baseline, args.evolved, args.output)


if __name__ == "__main__":
    main()
