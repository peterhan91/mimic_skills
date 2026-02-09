"""Lab Interpreter sub-agent — dedicated specialist for combinatorial lab reasoning.

Addresses Hager failure mode #2 (lab interpretation: 26-77% correct).
Exposed to the orchestrator via .as_tool() — the orchestrator calls it
after receiving raw lab results and gets structured interpretation back.
"""

from agents import Agent, RunContextWrapper

from context import PatientContext
from models import LabInterpretation


LAB_INTERPRETER_INSTRUCTIONS = (
    "You are a clinical laboratory specialist. Analyze the provided lab results "
    "and produce a structured interpretation.\n\n"
    "Focus on:\n"
    "1. Identify ALL abnormal values (high or low relative to reference ranges)\n"
    "2. Explain the clinical significance of each abnormality\n"
    "3. Look for PATTERNS across multiple values — what do combined abnormalities suggest?\n"
    "   - e.g., elevated WBC + elevated CRP = acute inflammatory process\n"
    "   - e.g., elevated lipase >3x ULN + abdominal pain = acute pancreatitis\n"
    "   - e.g., elevated ALT/AST + elevated bilirubin = hepatobiliary pathology\n"
    "4. Suggest follow-up tests that would help narrow the differential\n\n"
    "Do NOT name specific diseases in your diagnosis — describe the pathological "
    "process (e.g., 'acute inflammatory process' not 'appendicitis').\n"
    "Be concise and clinically precise."
)


def lab_interpreter_instructions(
    ctx: RunContextWrapper[PatientContext],
    agent: Agent,
) -> str:
    """Dynamic instructions: base + evolved skill if present."""
    skill = ctx.context.lab_interpreter_skill
    if skill:
        return LAB_INTERPRETER_INSTRUCTIONS + f"\n\n## Evolved Skill\n\n{skill}"
    return LAB_INTERPRETER_INSTRUCTIONS


async def _extract_lab_output(r):
    """Extract structured lab interpretation as a string for the orchestrator."""
    out = r.final_output
    return (
        f"Lab Interpretation:\n{out.summary}\n\n"
        f"Abnormal Findings:\n"
        + "\n".join(f"- {f}" for f in out.abnormal_findings)
        + f"\n\nClinical Significance: {out.clinical_significance}\n"
        f"Suggested Follow-up: {', '.join(out.suggested_followup)}"
    )


def create_lab_interpreter_tool(model_name="gpt-4o-mini"):
    """Create the Lab Interpreter agent and return it as a tool.

    Args:
        model_name: Model identifier (str for OpenAI, LitellmModel for others).
    """
    lab_interpreter = Agent(
        name="Lab Interpreter",
        instructions=lab_interpreter_instructions,
        output_type=LabInterpretation,
        model=model_name,
    )
    return lab_interpreter.as_tool(
        tool_name="interpret_lab_results",
        tool_description=(
            "Analyze lab results for clinical significance and patterns. "
            "Pass the raw lab result text. Returns a structured interpretation "
            "with abnormal findings, clinical significance, and suggested follow-up."
        ),
        custom_output_extractor=_extract_lab_output,
        max_turns=3,
    )
