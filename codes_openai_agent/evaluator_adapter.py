"""Bridge between OpenAI Agents SDK RunResult and Hager's PathologyEvaluator.

Converts SDK tool call items into the (AgentAction, observation) trajectory
format that PathologyEvaluator._evaluate_agent_trajectory() expects.

Key advantage: custom_parsings=0 for all actions since SDK function calling
produces valid JSON — the evaluator will report Action Parsing: 0 and
Invalid Tools: 0, directly measuring improvement over LangChain.
"""

import json
from typing import Sequence, Tuple

from models import DiagnosticResult
from hager_imports import AgentAction, load_evaluator  # noqa: F401


# Map SDK function_tool names back to Hager's tool names used in evaluation
TOOL_NAME_MAP = {
    "physical_examination": "Physical Examination",
    "laboratory_tests": "Laboratory Tests",
    "imaging": "Imaging",
    "diagnostic_criteria": "Diagnostic Criteria",
    # Sub-agent tools are not scored by PathologyEvaluator
    "interpret_lab_results": None,
    "challenge_diagnosis": None,
}


def _extract_action_input(tool_name: str, args: dict):
    """Convert SDK function arguments into the action_input format
    that PathologyEvaluator expects.

    The evaluator accesses:
    - action.tool_input["action_input"] for lab itemids (list[int])
    - action.tool_input["action_input"]["region"] and ["modality"] for imaging
    """
    if tool_name == "physical_examination":
        return ""
    elif tool_name == "laboratory_tests":
        return args.get("test_names", [])
    elif tool_name == "imaging":
        return {"region": args.get("region", ""), "modality": args.get("modality", "")}
    elif tool_name == "diagnostic_criteria":
        return args.get("pathologies", [])
    return args


def convert_sdk_result(
    run_result,
    final_output: DiagnosticResult,
    itemid_log: list[list] | None = None,
) -> Tuple[Sequence[Tuple], str]:
    """Convert an OpenAI Agents SDK RunResult into PathologyEvaluator format.

    Args:
        run_result: The RunResult from Runner.run().
        final_output: The structured DiagnosticResult.
        itemid_log: Optional list of itemid lists, one per laboratory_tests call,
            capturing the actual expanded itemids used. If provided, these replace
            the raw test_names in the trajectory for accurate evaluation.

    Returns:
        trajectory: List of (AgentAction, observation) tuples.
        prediction: Formatted "Final Diagnosis: ...\\nTreatment: ..." string.
    """
    # Collect tool calls and their outputs
    tool_calls = {}   # call_id -> (tool_name, args_dict)
    tool_outputs = {}  # call_id -> output_string

    for item in run_result.new_items:
        if item.type == "tool_call_item":
            # ResponseFunctionToolCall: direct attrs name, arguments, call_id
            call_id = item.raw_item.call_id
            name = item.raw_item.name
            args = json.loads(item.raw_item.arguments)
            tool_calls[call_id] = (name, args)
        elif item.type == "tool_call_output_item":
            # FunctionCallOutput is a TypedDict with call_id
            raw = item.raw_item
            call_id = raw["call_id"] if isinstance(raw, dict) else raw.call_id
            tool_outputs[call_id] = str(item.output)

    # Build trajectory in order
    trajectory = []
    lab_call_index = 0

    # Iterate new_items to maintain chronological order
    for item in run_result.new_items:
        if item.type != "tool_call_item":
            continue

        call_id = item.raw_item.call_id
        sdk_name, args = tool_calls[call_id]
        hager_name = TOOL_NAME_MAP.get(sdk_name)

        # Skip sub-agent tools (lab interpreter, challenger) — not scored
        if hager_name is None:
            continue

        action_input = _extract_action_input(sdk_name, args)

        # For lab tests, substitute expanded itemids if available
        if sdk_name == "laboratory_tests" and itemid_log is not None:
            if lab_call_index < len(itemid_log):
                action_input = itemid_log[lab_call_index]
            lab_call_index += 1

        action = AgentAction(
            tool=hager_name,
            tool_input={"action_input": action_input},
            log=json.dumps(args),
            custom_parsings=0,  # SDK = always valid JSON, no parsing heuristics
        )

        observation = tool_outputs.get(call_id, "")
        trajectory.append((action, observation))

    # Format prediction string for evaluator's parse_diagnosis/parse_treatment
    prediction = (
        f"Final Diagnosis: {final_output.diagnosis}\n"
        f"Treatment: {final_output.treatment}"
    )

    return trajectory, prediction
