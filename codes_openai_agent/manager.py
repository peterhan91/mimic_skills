"""ClinicalDiagnosisManager — orchestrates patient diagnosis flow.

Follows the pattern from OpenAI's financial_research_agent example:
creates a PatientContext, builds the orchestrator, runs diagnosis,
and returns structured results with tracing.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from agents import Runner, trace

from context import PatientContext
from sub_agents.orchestrator import create_orchestrator


@dataclass
class ManagerConfig:
    """Configuration for ClinicalDiagnosisManager."""

    model: Any = "gpt-4o"  # str for OpenAI models, LitellmModel for others
    sub_agent_model: Any = None  # Model for sub-agents; defaults to same as model
    lab_test_mapping_path: str = "./MIMIC-CDM-IV/lab_test_mapping.pkl"
    annotate_clinical: bool = True
    skill_path: Optional[str] = None
    max_turns: int = 20


class ClinicalDiagnosisManager:
    """Manages the end-to-end clinical diagnosis flow for a patient.

    Creates context, builds the orchestrator agent with sub-agents,
    runs diagnosis, and returns the structured result.
    """

    def __init__(self, config: ManagerConfig):
        self.config = config
        self.lab_test_mapping_df = pd.read_pickle(config.lab_test_mapping_path)
        self._skill_content = self._load_skill()

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
        )

        orchestrator = create_orchestrator(
            model_name=self.config.model,
            sub_agent_model_name=self.config.sub_agent_model,
        )

        patient_history = patient_data["Patient History"].strip()

        with trace(
            "clinical_diagnosis",
            metadata={"patient_id": str(patient_id)},
        ):
            result = await Runner.run(
                orchestrator,
                input=f"Patient History:\n{patient_history}",
                context=context,
                max_turns=self.config.max_turns,
            )

        return result.final_output, result, context
