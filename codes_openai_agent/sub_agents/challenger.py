"""Challenger sub-agent — devil's advocate for diagnostic rigor.

Based on research showing that challenger/devil's advocate patterns
improve diagnostic accuracy from 0% to 76% on biased scenarios.
Called by the orchestrator before finalizing a diagnosis.
"""

from agents import Agent

from models import ChallengerFeedback


challenger = Agent(
    name="Diagnostic Challenger",
    instructions=(
        "You are a senior clinician reviewing a proposed diagnosis. Your role is to "
        "act as a devil's advocate and challenge the diagnosis rigorously.\n\n"
        "Given the patient's history, examination findings, lab results, imaging, "
        "and the proposed diagnosis, you must:\n\n"
        "1. Identify evidence that CONTRADICTS the proposed diagnosis\n"
        "2. Suggest ALTERNATIVE diagnoses that also fit the available evidence\n"
        "3. Point out MISSING tests or findings that should have been obtained\n"
        "4. Check for ANCHORING BIAS — is the diagnostician fixating on one finding?\n\n"
        "Be rigorous but fair:\n"
        "- If the diagnosis is well-supported by strong evidence, recommend 'accept'\n"
        "- If there are significant gaps or contradictions, recommend 'reconsider'\n"
        "- If the evidence clearly points elsewhere, recommend 'reject'\n\n"
        "Your feedback helps prevent diagnostic errors."
    ),
    output_type=ChallengerFeedback,
    model="gpt-4o-mini",
)

async def _extract_challenger_output(r):
    """Extract structured challenger feedback as a string for the orchestrator."""
    out = r.final_output
    return (
        f"Challenge: {out.challenge}\n"
        f"Overlooked Evidence: {', '.join(out.overlooked_evidence)}\n"
        f"Alternative Diagnoses: {', '.join(out.alternative_diagnoses)}\n"
        f"Recommendation: {out.recommendation}"
    )


challenger_tool = challenger.as_tool(
    tool_name="challenge_diagnosis",
    tool_description=(
        "Review a proposed diagnosis against the evidence and challenge assumptions. "
        "Pass a summary of findings and the proposed diagnosis. "
        "Call this BEFORE committing to a final diagnosis."
    ),
    custom_output_extractor=_extract_challenger_output,
)
