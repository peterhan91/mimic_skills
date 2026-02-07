"""Pydantic output models for the OpenAI Agents SDK clinical diagnostic agent.

DiagnosticResult replaces Hager's 339-line DiagnosisWorkflowParser with
structured output that the SDK enforces via output_type.
"""

from pydantic import BaseModel, Field


class DiagnosticResult(BaseModel):
    """Structured diagnostic output enforced by the SDK's output_type."""

    reasoning: str = Field(
        description="Chain of thought: explain your reasoning process, "
        "what evidence supports and contradicts each hypothesis."
    )
    diagnosis: str = Field(
        description="Single final diagnosis (e.g., 'Appendicitis'). "
        "Do not include multiple diagnoses."
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'."
    )
    key_evidence: list[str] = Field(
        description="List of key supporting findings from PE, labs, and imaging."
    )
    differential: list[str] = Field(
        description="Alternative diagnoses considered and why they were ruled out."
    )
    treatment: str = Field(
        description="Recommended treatment plan including medications, "
        "procedures, and supportive care."
    )
    severity: str = Field(
        description="Severity assessment: 'mild', 'moderate', or 'severe'."
    )


class LabInterpretation(BaseModel):
    """Output from the Lab Interpreter sub-agent."""

    summary: str = Field(
        description="Brief interpretation of the lab panel for the orchestrator."
    )
    abnormal_findings: list[str] = Field(
        description="List of abnormal lab values with clinical significance."
    )
    clinical_significance: str = Field(
        description="What the abnormal pattern suggests diagnostically."
    )
    suggested_followup: list[str] = Field(
        description="Additional tests that should be ordered based on findings."
    )


class ChallengerFeedback(BaseModel):
    """Output from the Challenger sub-agent (devil's advocate)."""

    challenge: str = Field(
        description="What's wrong or incomplete about the proposed diagnosis."
    )
    overlooked_evidence: list[str] = Field(
        description="Evidence that was not adequately considered."
    )
    alternative_diagnoses: list[str] = Field(
        description="Alternative diagnoses that fit the evidence."
    )
    recommendation: str = Field(
        description="'accept', 'reconsider', or 'reject' the proposed diagnosis."
    )
