"""Guardrails for the clinical diagnostic agent.

- PE-first tool input guardrail: blocks lab/imaging calls if PE hasn't been done
  (addresses Hager failure mode #3: agents skip physical examination).
  Uses tool_input_guardrail (tool-level) — attached via function_tool's
  tool_input_guardrails parameter.

- Diagnosis output guardrail: validates that structured DiagnosticResult has
  meaningful content before accepting it.
  Uses output_guardrail (agent-level) — attached to the orchestrator Agent.
"""

from agents import (
    GuardrailFunctionOutput,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    output_guardrail,
    tool_input_guardrail,
    RunContextWrapper,
)

from context import PatientContext
from models import DiagnosticResult


@tool_input_guardrail
async def pe_first_guardrail(
    data: ToolInputGuardrailData,
) -> ToolGuardrailFunctionOutput:
    """Block lab/imaging calls if PE hasn't been performed yet.

    Attached to laboratory_tests and imaging tools via tool_input_guardrails.
    Returns a reject_content message so the model self-corrects and calls PE first.
    """
    ctx: PatientContext = data.context.context
    if ctx.pe_done:
        return ToolGuardrailFunctionOutput.allow()
    return ToolGuardrailFunctionOutput.reject_content(
        message="You must perform Physical Examination first before ordering tests or imaging."
    )


@output_guardrail
async def diagnosis_guardrail(
    ctx: RunContextWrapper[PatientContext],
    agent,
    output: DiagnosticResult,
) -> GuardrailFunctionOutput:
    """Validate that the structured diagnostic output has meaningful content."""
    issues = []
    if not output.diagnosis or len(output.diagnosis) < 3:
        issues.append("Diagnosis is empty or too short")
    if not output.treatment or len(output.treatment) < 5:
        issues.append("Treatment is empty or too short")
    if not output.key_evidence:
        issues.append("No key evidence provided")
    if issues:
        return GuardrailFunctionOutput(
            output_info={"issues": issues},
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(
        output_info={"status": "valid"},
        tripwire_triggered=False,
    )
