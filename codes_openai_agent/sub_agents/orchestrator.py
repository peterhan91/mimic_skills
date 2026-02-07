"""Orchestrator agent — main diagnostic agent with dynamic instructions.

Uses direct tools (PE, Labs, Imaging, DiagCrit) and sub-agent tools
(Lab Interpreter, Challenger) to diagnose patients. Produces a
DiagnosticResult via structured output.
"""

from agents import Agent, RunContextWrapper

from context import PatientContext
from models import DiagnosticResult
from tools import (
    physical_examination,
    laboratory_tests,
    imaging,
    diagnostic_criteria,
)
from sub_agents.lab_interpreter import create_lab_interpreter_tool
from sub_agents.challenger import create_challenger_tool


def clinical_instructions(
    ctx: RunContextWrapper[PatientContext],
    agent: Agent,
) -> str:
    """Generate per-patient dynamic instructions.

    This function is called by the SDK at each turn, allowing us to
    inject skill content and patient-specific context dynamically.
    """
    base = """\
You are a medical AI assistant diagnosing a patient presenting with acute abdominal pain.

WORKFLOW — follow this order strictly:
1. Physical Examination — ALWAYS do this FIRST before any other action
2. Laboratory Tests — order targeted tests based on PE findings
   - After receiving results, call interpret_lab_results to understand patterns
3. Imaging — order based on clinical picture (PE + labs)
4. Diagnostic Criteria — optionally look up criteria for suspected pathologies
5. Challenge Diagnosis — BEFORE finalizing, call challenge_diagnosis with your
   evidence summary and proposed diagnosis
6. Final Diagnosis — provide your structured diagnostic output

RULES:
- You MUST perform Physical Examination as your very first action
- Order only clinically indicated laboratory tests (avoid shotgun approach)
- Always use interpret_lab_results after receiving lab values
- Consider differential diagnoses throughout your reasoning
- Call challenge_diagnosis before committing to a final diagnosis
- Your final diagnosis must be a single specific pathology
- Treatment must include specific interventions (medications, procedures, supportive care)
- Assess severity to guide treatment intensity"""

    # Inject skill content if available
    skill = ctx.context.skill_content
    if skill:
        base += f"\n\n## Clinical Reasoning Skill\n\n{skill}"

    return base


def create_orchestrator(model_name="gpt-4o", sub_agent_model_name=None) -> Agent:
    """Create the orchestrator agent with all tools attached.

    Args:
        model_name: Model identifier for the orchestrator. For OpenAI models,
            pass the string directly (e.g. "gpt-4o"). For LiteLLM models,
            pass a LitellmModel instance instead.
        sub_agent_model_name: Model identifier for sub-agents (Lab Interpreter,
            Challenger). Defaults to the same as model_name.
    """
    if sub_agent_model_name is None:
        sub_agent_model_name = model_name

    lab_interpreter_tool = create_lab_interpreter_tool(model_name=sub_agent_model_name)
    challenger_tool = create_challenger_tool(model_name=sub_agent_model_name)

    return Agent(
        name="Clinical Diagnostician",
        instructions=clinical_instructions,
        tools=[
            physical_examination,
            laboratory_tests,
            imaging,
            diagnostic_criteria,
            lab_interpreter_tool,
            challenger_tool,
        ],
        output_type=DiagnosticResult,
        model=model_name,
    )
