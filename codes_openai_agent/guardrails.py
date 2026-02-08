"""Guardrails for the clinical diagnostic agent.

- Diagnosis output guardrail: validates that structured DiagnosticResult has
  meaningful content before accepting it.
  Uses output_guardrail (agent-level) — attached to the orchestrator Agent.
"""

from agents import (
    GuardrailFunctionOutput,
    output_guardrail,
    RunContextWrapper,
)

from context import PatientContext
from models import DiagnosticResult


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
