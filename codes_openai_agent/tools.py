"""@function_tool wrappers around Hager's retrieve_* functions.

These tools are called by the orchestrator agent via the OpenAI Agents SDK.
The SDK enforces JSON schema validation on inputs, eliminating tool
hallucination (Hager failure mode #5) and parsing errors (failure mode #8).
"""

from agents import function_tool, RunContextWrapper

from context import PatientContext
from hager_imports import get_actions, get_nlp


@function_tool
async def physical_examination(
    context: RunContextWrapper[PatientContext],
) -> str:
    """Perform a physical examination on the patient. Returns all physical exam findings. No input needed."""
    ctx = context.context
    ctx.pe_done = True
    actions = get_actions()
    result = actions.retrieve_physical_examination(action_results=ctx.patient_data)
    return f"Physical Examination:\n{result}"


@function_tool
async def laboratory_tests(
    context: RunContextWrapper[PatientContext],
    test_names: list[str],
) -> str:
    """Run laboratory tests. Specify test names as a list, e.g. ["WBC", "CBC", "Lipase", "ALT", "AST", "Bilirubin"]. Panel names like "CBC", "BMP", "CMP", "LFP" are expanded automatically."""
    ctx = context.context
    nlp_mod = get_nlp()
    actions = get_actions()

    # Convert human-readable test names to MIMIC itemids
    itemids = nlp_mod.convert_labs_to_itemid(test_names, ctx.lab_test_mapping_df)

    result = actions.retrieve_lab_tests(
        action_input=itemids,
        action_results=ctx.patient_data,
        lab_test_mapping_df=ctx.lab_test_mapping_df,
        include_ref_range=True,
        bin_lab_results=False,
        annotate_clinical=ctx.annotate_clinical,
    )

    ctx.labs_done = True
    ctx.lab_results_raw.append(result)
    ctx.lab_itemid_log.append(itemids)  # For evaluator_adapter
    return f"Laboratory Tests:\n{result}"


@function_tool
async def imaging(
    context: RunContextWrapper[PatientContext],
    modality: str,
    region: str,
) -> str:
    """Order an imaging scan. Specify the modality (e.g. "CT", "Ultrasound", "MRI", "Radiograph") and region (e.g. "Abdomen", "Chest", "Head"). Returns the radiologist's report."""
    ctx = context.context
    actions = get_actions()

    action_input = {"region": region, "modality": modality}
    result = actions.retrieve_imaging(
        action_input=action_input,
        action_results=ctx.patient_data,
        already_requested_scans=ctx.already_requested_scans,
    )

    ctx.imaging_done = True
    return f"Imaging:\n{result}"


@function_tool
async def diagnostic_criteria(
    context: RunContextWrapper[PatientContext],
    pathologies: list[str],
) -> str:
    """Look up diagnostic criteria for specific pathologies. Specify pathology names, e.g. ["appendicitis"] or ["cholecystitis", "pancreatitis"]. Returns clinical diagnostic criteria and guidelines."""
    actions = get_actions()
    result = actions.retrieve_diagnostic_criteria(action_input=pathologies)
    return f"Diagnostic Criteria:\n{result}"
