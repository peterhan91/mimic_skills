"""Patient simulator for tau-clinic evaluation mode.

When patient_simulator=True, the agent receives only a brief chief complaint
instead of the full Patient History. It must gather additional history by
calling the "Ask Patient" tool, which queries an LLM-backed patient simulator.

The simulator's hidden context is the sanitized Patient History (already
processed by Hager's sanitize_hadm_texts — disease names removed/masked).
The Discharge field is NOT used to prevent diagnosis leakage.

As an additional safety net, every simulator response is passed through
sanitize_skill_text() to catch any disease names the LLM might infer
and state in its response.

Usage:
    # In run.py
    if args.get("patient_simulator"):
        sim = PatientSimulator(
            patient_history=patient["Patient History"],
            llm=llm,
            tags=tags,
        )
        patient_input = extract_chief_complaint(patient["Patient History"])
    else:
        patient_input = patient["Patient History"].strip()
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field

try:
    from langchain.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool

from loguru import logger


# ── Sanitization import ──────────────────────────────────────────────

def _get_sanitizer():
    """Import sanitize_skill_text, falling back to identity if unavailable."""
    try:
        proj_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        scripts_dir = os.path.join(proj_root, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from sanitize_skill import sanitize_skill_text
        return sanitize_skill_text
    except ImportError:
        logger.warning(
            "Could not import sanitize_skill; patient simulator responses "
            "will not be sanitized for diagnosis leakage"
        )
        return lambda text: text


_sanitize = _get_sanitizer()


# ── Chief complaint extraction ───────────────────────────────────────

def extract_chief_complaint(patient_history: str, n_sentences: int = 3) -> str:
    """Extract a brief chief complaint from the full Patient History.

    Strategy: take the text before "Past Medical History:" (the HPI),
    then return the first N sentences. Skips trivially short fragments
    like "Mr." or "___." that result from MIMIC anonymization.

    Examples:
        Input:  "Mr. ___ presents with 4 days of RLQ pain. Says symptoms
                 started after heavy dinner. Reports decreased appetite
                 and chills.  Past Medical History: PMH: none ..."
        Output: "Mr. ___ presents with 4 days of RLQ pain. Says symptoms
                 started after heavy dinner. Reports decreased appetite
                 and chills."
    """
    text = patient_history.strip()
    if not text:
        return text

    # Split off everything after "Past Medical History:" (case-insensitive)
    hpi_match = re.split(
        r"Past Medical History:", text, maxsplit=1, flags=re.IGNORECASE
    )
    hpi = hpi_match[0].strip()

    if not hpi:
        return text  # fallback: return everything

    # Split into sentences: period followed by whitespace
    # Keep trivially short fragments (e.g., "Mr.", "___." ) attached to the next sentence
    raw_parts = re.split(r"(?<=\.)\s+", hpi)
    sentences = []
    buf = ""
    for part in raw_parts:
        buf = (buf + " " + part).strip() if buf else part
        # A real sentence should be >10 chars (skip "Mr.", "___.", "Ms.")
        if len(buf) > 10:
            sentences.append(buf)
            buf = ""
    if buf and sentences:
        sentences[-1] += " " + buf  # attach trailing fragment to last sentence
    elif buf:
        sentences.append(buf)

    if not sentences:
        return hpi

    result = " ".join(sentences[:n_sentences])

    # Cap at ~300 chars to keep it brief
    if len(result) > 300:
        cutoff = result[:300].rfind(" ")
        if cutoff > 50:
            return result[:cutoff] + "..."
    return result


# ── Simulator prompt ─────────────────────────────────────────────────

PATIENT_SIM_PROMPT = """{system_tag_start}You are a patient in the emergency department. A doctor is asking you questions about your condition.

RULES:
- Answer based ONLY on the medical history provided below.
- Only share information that a patient would know: your symptoms, when they started, how they feel, your past medical conditions, medications, family history, social habits.
- Do NOT use medical jargon that a typical patient would not know.
- Do NOT name any diagnoses or diseases that you have not been previously told about. If asked what you think you have, say you are not sure.
- Do NOT reveal laboratory values, imaging findings, or clinical examination results — you do not know these yet.
- Be natural and conversational. You may be vague about some details.
- Keep answers concise (2-4 sentences).
- If the doctor asks about something not covered in your history, say you are not sure or it does not apply to you.

YOUR MEDICAL HISTORY (hidden from doctor):
{patient_info}{system_tag_end}{user_tag_start}{conversation}Doctor: {question}
{user_tag_end}{ai_tag_start}Patient:"""


# ── Patient simulator ────────────────────────────────────────────────

class PatientSimulator:
    """LLM-backed patient that answers doctor questions.

    Stateless design: conversation history is passed in explicitly so
    that ToT branches can maintain independent histories.

    Args:
        patient_history: The sanitized Patient History string (HPI+PMH+Social+Family).
                         Do NOT pass the Discharge field — it contains the diagnosis.
        llm: The CustomLLM instance (uses generate_with_temperature at temp=0).
        tags: Chat template tags dict (system_tag_start, etc.).
    """

    def __init__(self, patient_history: str, llm, tags: Dict[str, str]):
        self.patient_info = patient_history
        self.llm = llm
        self.tags = tags

    def respond(
        self, question: str, history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Generate a patient response to the doctor's question.

        Args:
            question: The doctor's question.
            history: List of (question, answer) tuples from prior exchanges.
                     Pass explicitly for ToT branch isolation.

        Returns:
            The patient's response (sanitized for diagnosis leakage).
        """
        # Build conversation history string
        conv_parts = []
        for q, a in (history or []):
            conv_parts.append(f"Doctor: {q}")
            conv_parts.append(f"Patient: {a}")
        conversation = "\n".join(conv_parts)
        if conversation:
            conversation += "\n"

        prompt = PATIENT_SIM_PROMPT.format(
            patient_info=self.patient_info,
            conversation=conversation,
            question=question,
            system_tag_start=self.tags["system_tag_start"],
            system_tag_end=self.tags["system_tag_end"],
            user_tag_start=self.tags["user_tag_start"],
            user_tag_end=self.tags["user_tag_end"],
            ai_tag_start=self.tags["ai_tag_start"],
        )

        # Generate at temperature=0 for deterministic responses (ToT cache-friendly)
        response = self.llm.generate_with_temperature(
            prompt,
            stop=["Doctor:", "\nDoctor"],
            temperature=0.0,
        )

        # Safety net: sanitize any disease names the LLM might have inferred
        response = _sanitize(response)

        return response.strip()


# ── Ask Patient tool ─────────────────────────────────────────────────

class AskPatient_Input(BaseModel):
    action_input: str = Field(
        description="The question to ask the patient."
    )


class AskPatient(BaseTool):
    """Tool that lets the clinical agent ask the patient questions.

    For ZeroShot: conversation history is maintained internally.
    For ToT: the runner passes history explicitly via respond(), bypassing
    the internal state (see tot_agent.py _execute_tool).
    """

    name: str = "Ask Patient"
    description: str = (
        "Ask the patient a question about their symptoms, medical history, "
        "medications, family history, or social habits. The question must be "
        "specified in the 'Action Input' field."
    )
    args_schema: Type[BaseModel] = AskPatient_Input
    simulator: Any = None
    conversation_history: List = []

    class Config:
        arbitrary_types_allowed = True

    def _run(self, action_input: str) -> str:
        response = self.simulator.respond(
            question=action_input,
            history=self.conversation_history,
        )
        self.conversation_history.append((action_input, response))
        return response

    async def _arun(self, action_input: str) -> str:
        return self._run(action_input)
