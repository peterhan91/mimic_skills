"""Orchestrator agent — main diagnostic agent with dynamic instructions.

Uses direct tools (PE, Labs, Imaging, DiagCrit) and sub-agent tools
(Lab Interpreter, Challenger) to diagnose patients. Produces text output
that is parsed into a DiagnosticResult by the manager.
"""

from agents import Agent, ModelSettings, RunContextWrapper

from context import PatientContext
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

AVAILABLE TOOLS:
- Physical Examination — returns all physical exam findings (no input needed)
- Laboratory Tests — order specific tests by name
  - After receiving results, call interpret_lab_results to understand patterns
- Imaging — order a scan by modality and region
- Diagnostic Criteria — look up criteria for suspected pathologies
- Challenge Diagnosis — get a second opinion before finalizing

Use the tools you judge clinically appropriate for this patient. Not every
case requires every tool — let the clinical picture guide your workup.

RULES:
- Order only clinically indicated tests (avoid shotgun approach)
- Consider differential diagnoses throughout your reasoning
- Your final diagnosis must be a single specific pathology
- Treatment must include specific interventions (medications, procedures, supportive care)
- Assess severity to guide treatment intensity

OUTPUT FORMAT:
When you have completed your workup and are ready to diagnose, respond with:

Final Diagnosis: [single specific pathology]
Treatment: [specific interventions including medications, procedures, supportive care]"""

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
        model=model_name,
        model_settings=ModelSettings(temperature=0.0, parallel_tool_calls=False),
    )
