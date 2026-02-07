"""PatientContext — shared state passed through RunContextWrapper to all tools and agents."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class PatientContext:
    """Shared mutable state for a single patient diagnosis session.

    Passed as RunContextWrapper[PatientContext] to all @function_tool
    functions and agent instruction callables.
    """

    patient_id: int
    patient_data: dict  # hadm_info_clean[id] dict from Hager's pickle
    lab_test_mapping_df: pd.DataFrame
    model_name: Any = "gpt-4o"  # str or LitellmModel instance
    annotate_clinical: bool = True
    skill_content: Optional[str] = None

    # Mutable state tracking — reset per patient
    already_requested_scans: dict = field(default_factory=dict)
    pe_done: bool = False
    labs_done: bool = False
    imaging_done: bool = False
    lab_results_raw: list = field(default_factory=list)
    lab_itemid_log: list = field(default_factory=list)  # list[list[int]] for evaluator
