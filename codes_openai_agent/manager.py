"""ClinicalDiagnosisManager — orchestrates patient diagnosis flow.

Follows the pattern from OpenAI's financial_research_agent example:
creates a PatientContext, builds the orchestrator, runs diagnosis,
and returns structured results with tracing.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from agents import Runner, RunErrorHandlerResult, RunErrorHandlers, trace

from context import PatientContext
from hooks import ClinicalRunHooks
from models import DiagnosticResult
from sub_agents.orchestrator import create_orchestrator

logger = logging.getLogger("sdk_agent")


@dataclass
class ManagerConfig:
    """Configuration for ClinicalDiagnosisManager."""

    model: Any = "gpt-4o"  # str for OpenAI models, LitellmModel for others
    sub_agent_model: Any = None  # Model for sub-agents; defaults to same as model
    lab_test_mapping_path: str = "./MIMIC-CDM-IV/lab_test_mapping.pkl"
    annotate_clinical: bool = True
    skill_path: Optional[str] = None
    sub_agent_skill_path: Optional[str] = None
    max_turns: int = 20


def parse_sub_agent_skills(text: str) -> Dict[str, str]:
    """Parse sub-agent skill file split by ``<!-- SECTION: name -->`` delimiters.

    Returns dict mapping section name to content, e.g.
    {"lab_interpreter": "...", "challenger": "..."}.
    """
    sections: Dict[str, str] = {}
    parts = re.split(r"<!--\s*SECTION:\s*(\w+)\s*-->", text)
    # parts = [preamble, name1, content1, name2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            sections[name] = content
    return sections


class ClinicalDiagnosisManager:
    """Manages the end-to-end clinical diagnosis flow for a patient.

    Creates context, builds the orchestrator agent with sub-agents,
    runs diagnosis, and returns the structured result.
    """

    def __init__(self, config: ManagerConfig):
        self.config = config
        self.lab_test_mapping_df = pd.read_pickle(config.lab_test_mapping_path)
        self._skill_content = self._load_skill()
        self._sub_agent_skills = self._load_sub_agent_skills()

    def _load_skill(self) -> Optional[str]:
        """Load and strip YAML frontmatter from a SKILL.md file."""
        if not self.config.skill_path or not os.path.exists(self.config.skill_path):
            return None

        with open(self.config.skill_path, "r") as f:
            raw = f.read()

        # Strip YAML frontmatter (same logic as Hager's agent.py)
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                raw = parts[2].strip()

        return raw

    def _load_sub_agent_skills(self) -> Dict[str, str]:
        """Load sub-agent skills from a section-delimited markdown file.

        Returns dict with keys 'lab_interpreter' and/or 'challenger'.
        """
        path = self.config.sub_agent_skill_path
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            raw = f.read()
        return parse_sub_agent_skills(raw)

    async def run(self, patient_id: int, patient_data: dict):
        """Run diagnosis for a single patient.

        Args:
            patient_id: The HADM ID.
            patient_data: The hadm_info_clean[id] dict from Hager's pickle.

        Returns:
            Tuple of (DiagnosticResult, RunResult, PatientContext).
            The PatientContext contains lab_itemid_log for evaluation.
        """
        context = PatientContext(
            patient_id=patient_id,
            patient_data=patient_data,
            lab_test_mapping_df=self.lab_test_mapping_df,
            model_name=self.config.model,
            annotate_clinical=self.config.annotate_clinical,
            skill_content=self._skill_content,
            lab_interpreter_skill=self._sub_agent_skills.get("lab_interpreter"),
            challenger_skill=self._sub_agent_skills.get("challenger"),
        )

        orchestrator = create_orchestrator(
            model_name=self.config.model,
            sub_agent_model_name=self.config.sub_agent_model,
        )

        patient_history = patient_data["Patient History"].strip()
        hooks = ClinicalRunHooks()

        async def _max_turns_handler(handler_input):
            """Graceful degradation: return best-effort diagnosis on turn limit."""
            logger.warning(
                f"Patient {patient_id}: max turns ({self.config.max_turns}) exceeded"
            )
            return RunErrorHandlerResult(
                final_output=DiagnosticResult(
                    diagnosis="Unable to reach diagnosis (max turns exceeded)",
                    confidence="low",
                    treatment="Recommend senior physician consultation",
                    severity="unknown",
                    key_evidence=[],
                    differential=[],
                    reasoning="Agent exceeded maximum allowed turns.",
                ),
                include_in_history=True,
            )

        with trace(
            "clinical_diagnosis",
            metadata={"patient_id": str(patient_id)},
        ):
            result = await Runner.run(
                orchestrator,
                input=f"Patient History:\n{patient_history}",
                context=context,
                max_turns=self.config.max_turns,
                hooks=hooks,
                error_handlers=RunErrorHandlers(max_turns=_max_turns_handler),
            )

        # Log token usage
        usage = result.context_wrapper.usage
        context.token_usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "requests": usage.requests,
        }
        logger.info(
            f"Patient {patient_id}: {usage.total_tokens} tokens "
            f"({usage.input_tokens} in / {usage.output_tokens} out), "
            f"{usage.requests} requests, "
            f"{len(context.tool_call_log)} tool calls: {context.tool_call_log}"
        )

        return result.final_output, result, context
