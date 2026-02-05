"""
Extract structured trajectories from agent results + eval pkl + patient data.

Produces a JSON file with per-patient trajectories, scores, and discharge
summaries — the input format needed by the Evolver (evolve_skill.py).

Usage:
    python scripts/extract_trajectories.py \
        --results_dir results/cholecystitis_ZeroShot_.../ \
        --pathology cholecystitis \
        --patient_data data_splits/cholecystitis/train.pkl \
        --output trajectories/baseline_cholecystitis_train10.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Add framework to path
FRAMEWORK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "codes_Hager",
    "MIMIC-Clinical-Decision-Making-Framework",
)
sys.path.insert(0, FRAMEWORK_DIR)

from utils.logging import read_from_pickle_file


def find_pkl(results_dir, suffix):
    """Find a pkl file matching a suffix in results_dir."""
    results_dir = Path(results_dir)
    candidates = list(results_dir.glob(f"*{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError(f"No *{suffix} found in {results_dir}")
    else:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]


def load_patient_data(patient_data_path):
    """Load patient data pkl."""
    with open(patient_data_path, "rb") as f:
        return pickle.load(f)


def load_eval_results(results_dir):
    """Load eval pkl. Returns dict: {hadm_id: {"scores": {...}, "answers": {...}}}."""
    eval_pkl = find_pkl(results_dir, "_eval.pkl")
    with open(eval_pkl, "rb") as f:
        return pickle.load(f)


def parse_trajectory_steps(intermediate_steps):
    """Convert intermediate_steps list of (AgentAction, observation) tuples to JSON-safe dicts."""
    steps = []
    for action, observation in intermediate_steps:
        step = {
            "tool": action.tool,
            "tool_input": _safe_tool_input(action.tool_input),
            "log": action.log,
            "custom_parsings": action.custom_parsings,
            "observation": str(observation)[:2000],  # Truncate very long observations
        }
        steps.append(step)
    return steps


def _safe_tool_input(tool_input):
    """Make tool_input JSON-serializable."""
    if isinstance(tool_input, dict):
        result = {}
        for k, v in tool_input.items():
            if isinstance(v, dict):
                result[k] = {str(kk): str(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                result[k] = [str(item) for item in v]
            else:
                result[k] = str(v)
        return result
    return str(tool_input)


def _safe_answers(answers):
    """Make answers dict JSON-serializable."""
    result = {}
    for k, v in answers.items():
        if isinstance(v, dict):
            result[k] = {str(kk): _json_safe(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            result[k] = [_json_safe(item) for item in v]
        else:
            result[k] = _json_safe(v)
    return result


def _json_safe(val):
    """Convert a value to something JSON-serializable."""
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(item) for item in val]
    # Tensors, numpy arrays, etc.
    return str(val)


def extract_trajectories(results_dir, pathology, patient_data_path, output_path):
    """Extract structured trajectories and write to JSON."""
    results_pkl = find_pkl(results_dir, "_results.pkl")
    hadm_info = load_patient_data(patient_data_path)
    eval_results = load_eval_results(results_dir)

    admissions = []

    for result_dict in read_from_pickle_file(str(results_pkl)):
        for _id, result in result_dict.items():
            patient_data = hadm_info.get(_id, {})
            eval_data = eval_results.get(_id, {"scores": {}, "answers": {}})

            admission = {
                "hadm_id": _id,
                "input": result.get("input", ""),
                "output": result.get("output", ""),
                "trajectory": parse_trajectory_steps(
                    result.get("intermediate_steps", [])
                ),
                "scores": eval_data.get("scores", {}),
                "answers": _safe_answers(eval_data.get("answers", {})),
                "discharge_summary": patient_data.get("Discharge", ""),
                "discharge_diagnosis": patient_data.get("Discharge Diagnosis", ""),
            }
            admissions.append(admission)

    # Compute aggregate scores
    aggregate = {}
    if admissions:
        score_keys = list(admissions[0]["scores"].keys())
        for key in score_keys:
            values = [a["scores"].get(key, 0) for a in admissions]
            aggregate[key] = sum(values) / len(values) if values else 0

    output_data = {
        "pathology": pathology,
        "n_patients": len(admissions),
        "results_dir": str(results_dir),
        "aggregate": aggregate,
        "admissions": admissions,
    }

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Wrote {len(admissions)} trajectories to {output_path}")

    # Print aggregate scores
    print(f"\n{'='*60}")
    print(f"TRAJECTORY SUMMARY: {pathology} ({len(admissions)} patients)")
    print(f"{'='*60}")
    for key, val in aggregate.items():
        if key in ("Diagnosis", "Gracious Diagnosis", "Physical Examination",
                    "Late Physical Examination", "Action Parsing",
                    "Treatment Parsing", "Diagnosis Parsing"):
            n_success = sum(1 for a in admissions if a["scores"].get(key, 0) > 0)
            print(f"  {key:30s}: {n_success}/{len(admissions)} ({100*val:.0f}%)")
        elif key in ("Laboratory Tests", "Imaging"):
            print(f"  {key:30s}: avg={val:.2f}")
        elif key == "Rounds":
            print(f"  {key:30s}: avg={val:.1f}")
        elif key == "Invalid Tools":
            total = sum(a["scores"].get(key, 0) for a in admissions)
            print(f"  {key:30s}: {total} total")

    # List failed patients
    failed = [
        a for a in admissions
        if a["scores"].get("Diagnosis", 0) == 0
    ]
    if failed:
        print(f"\nFAILED DIAGNOSES ({len(failed)}):")
        for a in failed:
            agent_dx = a["answers"].get("Diagnosis", "?")
            true_dx = a["discharge_diagnosis"][:80]
            print(f"  hadm_id={a['hadm_id']}: agent='{agent_dx}' | true='{true_dx}'")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured trajectories from agent results"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing *_results.pkl and *_eval.pkl"
    )
    parser.add_argument(
        "--pathology", type=str, required=True,
        help="Pathology name"
    )
    parser.add_argument(
        "--patient_data", type=str, required=True,
        help="Path to patient data pkl"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON path"
    )
    args = parser.parse_args()

    extract_trajectories(
        args.results_dir, args.pathology, args.patient_data, args.output
    )


if __name__ == "__main__":
    main()
