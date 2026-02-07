"""Entry point for the OpenAI Agents SDK clinical diagnostic agent.

Usage:
    # Run on 1 patient with GPT-4o (smoke test)
    python run.py --pathology appendicitis --data-dir ../data_splits/appendicitis --max-patients 1

    # Run full test split
    python run.py --pathology appendicitis --data-dir ../data_splits/appendicitis --split test

    # Use LiteLLM for non-OpenAI models
    python run.py --pathology appendicitis --litellm-model anthropic/claude-sonnet-4-5-20250929

    # Inject a skill
    python run.py --pathology appendicitis --skill-path ../skills/v1/acute_abdominal_pain.md
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# Add our own package to path
_OUR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _OUR_ROOT not in sys.path:
    sys.path.insert(0, _OUR_ROOT)

from hager_imports import load_hadm_from_file, load_evaluator
from manager import ClinicalDiagnosisManager, ManagerConfig
from evaluator_adapter import convert_sdk_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK Clinical Diagnostic Agent"
    )
    parser.add_argument(
        "--pathology",
        choices=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
        required=True,
        help="Pathology to diagnose",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--litellm-model",
        default=None,
        help="LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-5-20250929'",
    )
    parser.add_argument(
        "--litellm-base-url",
        default=None,
        help="Base URL for LiteLLM (e.g. http://gpu:8000/v1)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing {pathology}_hadm_info_first_diag.pkl",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "full"],
        default="train",
        help="Which data split to use (default: train = 10 patients)",
    )
    parser.add_argument(
        "--lab-mapping",
        default=None,
        help="Path to lab_test_mapping.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Output directory for results (default: ../results)",
    )
    parser.add_argument(
        "--annotate-clinical",
        action="store_true",
        default=True,
        help="Enable clinical annotations on lab results (default: True)",
    )
    parser.add_argument(
        "--no-annotate-clinical",
        action="store_false",
        dest="annotate_clinical",
    )
    parser.add_argument(
        "--skill-path",
        default=None,
        help="Path to SKILL.md for skill injection",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit number of patients to process",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum agent turns per patient (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed (default: 2023)",
    )
    return parser.parse_args()


def resolve_data_dir(args):
    """Resolve data directory from args or defaults."""
    if args.data_dir:
        return args.data_dir
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_splits",
        args.pathology,
    )


def resolve_lab_mapping(args):
    """Resolve lab_test_mapping.pkl path from args or defaults."""
    if args.lab_mapping:
        return args.lab_mapping
    # Check common locations
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "MIMIC-CDM-IV",
            "lab_test_mapping.pkl",
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_splits",
            args.pathology,
            "lab_test_mapping.pkl",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"lab_test_mapping.pkl not found. Tried: {candidates}. "
        "Pass --lab-mapping explicitly."
    )


def load_patients(data_dir, pathology, split):
    """Load patient data from pickle file."""
    if split == "full":
        filename = f"{pathology}_hadm_info_first_diag"
    else:
        filename = split
    return load_hadm_from_file(filename, base_mimic=data_dir)


def build_reference_tuple(patient_data):
    """Build the reference tuple expected by PathologyEvaluator."""
    return (
        patient_data.get("Discharge Diagnosis", ""),
        patient_data.get("ICD Diagnosis", []),
        patient_data.get("Procedures ICD9", []),
        patient_data.get("Procedures ICD10", []),
        patient_data.get("Procedures Discharge", []),
        [],  # Extra slot expected by evaluator
    )


async def main(args):
    data_dir = resolve_data_dir(args)
    lab_mapping_path = resolve_lab_mapping(args)

    # Resolve model
    model = args.model
    if args.litellm_model:
        from agents.extensions.models.litellm_model import LitellmModel

        model = LitellmModel(
            model=args.litellm_model,
            **({"base_url": args.litellm_base_url} if args.litellm_base_url else {}),
        )

    # Build manager config
    config = ManagerConfig(
        model=model,
        lab_test_mapping_path=lab_mapping_path,
        annotate_clinical=args.annotate_clinical,
        skill_path=args.skill_path,
        max_turns=args.max_turns,
    )

    manager = ClinicalDiagnosisManager(config)

    # Load patient data
    patients = load_patients(data_dir, args.pathology, args.split)
    patient_ids = list(patients.keys())
    if args.max_patients:
        patient_ids = patient_ids[: args.max_patients]

    print(f"Running {args.pathology} diagnosis on {len(patient_ids)} patients")
    print(f"Model: {args.litellm_model or args.model}")
    print(f"Data: {data_dir} (split={args.split})")
    if args.skill_path:
        print(f"Skill: {args.skill_path}")

    # Setup output directory
    dt = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y_%H:%M:%S")
    model_label = (args.litellm_model or args.model).replace("/", "_")
    run_name = f"{args.pathology}_sdk_{model_label}_{dt}"
    if args.annotate_clinical:
        run_name += "_CLANNOT"
    if args.skill_path:
        skill_label = Path(args.skill_path).stem
        run_name += f"_SKILL_{skill_label}"

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"{run_name}_results.pkl")
    eval_path = os.path.join(output_dir, f"{run_name}_eval.pkl")

    # Run diagnosis for each patient
    all_results = {}
    all_evals = {}
    total = len(patient_ids)

    for i, patient_id in enumerate(patient_ids):
        print(f"\n[{i + 1}/{total}] Patient {patient_id}")

        evaluator = load_evaluator(args.pathology)

        try:
            final_output, run_result, ctx = await manager.run(
                patient_id, patients[patient_id]
            )

            # Convert to evaluator format
            trajectory, prediction = convert_sdk_result(
                run_result,
                final_output,
                itemid_log=ctx.lab_itemid_log,
            )

            # Evaluate
            reference = build_reference_tuple(patients[patient_id])
            eval_result = evaluator._evaluate_agent_trajectory(
                prediction=prediction,
                input=patients[patient_id]["Patient History"],
                agent_trajectory=trajectory,
                reference=reference,
            )

            all_results[patient_id] = {
                "diagnosis": final_output.diagnosis,
                "confidence": final_output.confidence,
                "treatment": final_output.treatment,
                "severity": final_output.severity,
                "key_evidence": final_output.key_evidence,
                "differential": final_output.differential,
                "reasoning": final_output.reasoning,
                "prediction": prediction,
                "trajectory_length": len(trajectory),
            }
            all_evals[patient_id] = eval_result

            # Print summary
            scores = eval_result["scores"]
            print(f"  Diagnosis: {final_output.diagnosis} "
                  f"(correct={scores['Diagnosis']}, "
                  f"gracious={scores['Gracious Diagnosis']})")
            print(f"  PE first: {scores['Physical Examination']}, "
                  f"Labs: {scores['Laboratory Tests']}, "
                  f"Imaging: {scores['Imaging']}")
            print(f"  Action Parsing: {scores['Action Parsing']}, "
                  f"Invalid Tools: {scores['Invalid Tools']}, "
                  f"Rounds: {scores['Rounds']}")
            print(f"  Treatment: {final_output.treatment[:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[patient_id] = {"error": str(e)}
            all_evals[patient_id] = {"error": str(e)}

    # Save results
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    with open(eval_path, "wb") as f:
        pickle.dump(all_evals, f)

    # Save human-readable summary
    summary_path = os.path.join(output_dir, f"{run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": {
                    "pathology": args.pathology,
                    "model": args.litellm_model or args.model,
                    "split": args.split,
                    "annotate_clinical": args.annotate_clinical,
                    "skill_path": args.skill_path,
                    "max_turns": args.max_turns,
                    "n_patients": len(patient_ids),
                },
                "results": {
                    str(k): v for k, v in all_results.items()
                },
                "evaluations": {
                    str(k): v for k, v in all_evals.items()
                },
            },
            f,
            indent=2,
            default=str,
        )

    # Print aggregate metrics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    valid_evals = {
        k: v for k, v in all_evals.items()
        if isinstance(v, dict) and "scores" in v
    }

    if valid_evals:
        score_keys = list(next(iter(valid_evals.values()))["scores"].keys())
        for key in score_keys:
            values = [v["scores"][key] for v in valid_evals.values()]
            avg = sum(values) / len(values)
            print(f"  {key}: {avg:.3f} (n={len(values)})")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
