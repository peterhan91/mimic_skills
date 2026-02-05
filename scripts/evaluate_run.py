"""
Run PathologyEvaluator on a results pkl to produce an eval pkl.

Hager's run.py does NOT call the evaluator — evaluation is done separately.
This script follows the same pattern as evaluate_cdm.py in the Analysis repo.

Usage:
    python scripts/evaluate_run.py \
        --results_dir results/cholecystitis_ZeroShot_.../ \
        --pathology cholecystitis \
        --patient_data data_splits/cholecystitis/train.pkl
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Add framework to path so we can import evaluators
FRAMEWORK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "codes_Hager",
    "MIMIC-Clinical-Decision-Making-Framework",
)
sys.path.insert(0, FRAMEWORK_DIR)

from utils.logging import read_from_pickle_file, append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator


EVALUATORS = {
    "appendicitis": AppendicitisEvaluator,
    "cholecystitis": CholecystitisEvaluator,
    "diverticulitis": DiverticulitisEvaluator,
    "pancreatitis": PancreatitisEvaluator,
}


def load_patient_data(patient_data_path):
    """Load patient data pkl (hadm_info dict)."""
    with open(patient_data_path, "rb") as f:
        hadm_info = pickle.load(f)
    return hadm_info


def find_results_pkl(results_dir):
    """Find the *_results.pkl file in a results directory."""
    results_dir = Path(results_dir)
    candidates = list(results_dir.glob("*_results.pkl"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError(f"No *_results.pkl found in {results_dir}")
    else:
        # Pick the most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"WARNING: Multiple results pkls found, using {candidates[0]}")
        return candidates[0]


def evaluate_run(results_dir, pathology, patient_data_path):
    """Run evaluation on all patients in a results pkl.

    Returns:
        dict: {hadm_id: {"scores": {...}, "answers": {...}}} for all patients
    """
    results_pkl = find_results_pkl(results_dir)
    hadm_info = load_patient_data(patient_data_path)
    evaluator_cls = EVALUATORS[pathology]

    eval_results = {}
    n_evaluated = 0

    for result_dict in read_from_pickle_file(str(results_pkl)):
        for _id, result in result_dict.items():
            if _id not in hadm_info:
                print(f"WARNING: Patient {_id} not found in patient data, skipping")
                continue

            # Build reference tuple (same as evaluate_cdm.py)
            reference = (
                hadm_info[_id].get("Discharge Diagnosis", ""),
                hadm_info[_id].get("ICD Diagnosis", []),
                hadm_info[_id].get("Procedures ICD9", []),
                hadm_info[_id].get("Procedures ICD10", []),
                hadm_info[_id].get("Procedures Discharge", []),
            )

            # Convert intermediate_steps to use our AgentAction format
            trajectory = result.get("intermediate_steps", [])

            # Fresh evaluator per patient
            evaluator = evaluator_cls()

            eval_result = evaluator._evaluate_agent_trajectory(
                prediction=result.get("output", ""),
                input=result.get("input", ""),
                agent_trajectory=trajectory,
                reference=reference,
            )

            eval_results[_id] = eval_result
            n_evaluated += 1

            # Print per-patient summary
            dx = eval_result["scores"].get("Diagnosis", 0)
            pe = eval_result["scores"].get("Physical Examination", 0)
            labs = eval_result["scores"].get("Laboratory Tests", 0)
            imaging = eval_result["scores"].get("Imaging", 0)
            print(
                f"  Patient {_id}: Dx={dx} PE={pe} Labs={labs} Imaging={imaging}"
            )

    # Save eval pkl
    eval_pkl_path = str(results_pkl).replace("_results.pkl", "_eval.pkl")
    with open(eval_pkl_path, "wb") as f:
        pickle.dump(eval_results, f)
    print(f"\nSaved eval results to {eval_pkl_path}")

    # Print aggregate summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {pathology} ({n_evaluated} patients)")
    print(f"{'='*60}")

    if n_evaluated == 0:
        print("No patients evaluated!")
        return eval_results

    # Aggregate scores
    score_keys = list(next(iter(eval_results.values()))["scores"].keys())
    for key in score_keys:
        values = [eval_results[_id]["scores"][key] for _id in eval_results]
        if key in ("Diagnosis", "Gracious Diagnosis", "Physical Examination",
                    "Late Physical Examination", "Action Parsing",
                    "Treatment Parsing", "Diagnosis Parsing"):
            # Binary metrics: count successes
            total = sum(1 for v in values if v > 0)
            print(f"  {key:30s}: {total}/{n_evaluated} ({100*total/n_evaluated:.0f}%)")
        elif key in ("Laboratory Tests", "Imaging"):
            # Multi-point metrics: show average
            avg = sum(values) / n_evaluated
            print(f"  {key:30s}: avg={avg:.2f}")
        elif key == "Rounds":
            avg = sum(values) / n_evaluated
            print(f"  {key:30s}: avg={avg:.1f}")
        elif key == "Invalid Tools":
            total = sum(values)
            print(f"  {key:30s}: {total} total")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent run results")
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing *_results.pkl"
    )
    parser.add_argument(
        "--pathology", type=str, required=True,
        choices=list(EVALUATORS.keys()),
        help="Pathology to evaluate"
    )
    parser.add_argument(
        "--patient_data", type=str, required=True,
        help="Path to patient data pkl (e.g., data_splits/cholecystitis/train.pkl)"
    )
    args = parser.parse_args()

    evaluate_run(args.results_dir, args.pathology, args.patient_data)


if __name__ == "__main__":
    main()
