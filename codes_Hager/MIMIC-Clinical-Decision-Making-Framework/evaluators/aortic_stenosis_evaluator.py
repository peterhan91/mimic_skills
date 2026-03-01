from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    AORTIC_VALVE_PROCEDURES_ICD9,
    AORTIC_VALVE_PROCEDURES_ICD10,
    AORTIC_VALVE_PROCEDURES_KEYWORDS,
    ALTERNATE_AORTIC_VALVE_KEYWORDS,
)


class AorticStenosisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of aortic stenosis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "aortic stenosis"
        self.alternative_pathology_names = [
            {
                "location": "aortic",
                "modifiers": ["stenosis"],
            },
            {
                "location": "aortic valve",
                "modifiers": ["disease", "stenosis"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "valvular", "modifiers": ["disease"]},
            {"location": "aortic", "modifiers": ["valve disease"]},
            {"location": "aortic", "modifiers": ["sclerosis"]},
        ]

        self.required_lab_tests = {
            "Cardiac": [
                50963,  # NTproBNP / BNP
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Basic Metabolic Panel (BMP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(INFLAMMATION_LAB_TESTS)
        self.neutral_lab_tests.extend([
            51003,  # Troponin T
            51002,  # Troponin I
            52642,  # Troponin I (alternate)
            50911,  # CK-MB
        ])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Cardiac"]
        ]

        self.answers["Treatment Requested"] = {
            "Valve_Intervention": False,
            "Medications": False,
        }
        self.answers["Treatment Required"] = {
            "Valve_Intervention": False,
            "Medications": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        if region == "Chest":
            # Echocardiogram is the definitive diagnostic tool for valvular disease
            if modality == "Echocardiogram":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # ECG — LVH pattern, conduction abnormalities
            if modality == "ECG":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # CXR can show cardiomegaly, calcification
            if modality == "Radiograph":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # CT for calcium scoring
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### VALVE INTERVENTION ###
        if (
            procedure_checker(AORTIC_VALVE_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                AORTIC_VALVE_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                AORTIC_VALVE_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Valve_Intervention"] = True

        if (
            procedure_checker(
                AORTIC_VALVE_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
            )
            or treatment_alternative_procedure_checker(
                ALTERNATE_AORTIC_VALVE_KEYWORDS, self.answers["Treatment"]
            )
        ):
            self.answers["Treatment Requested"]["Valve_Intervention"] = True

        ### MEDICATIONS ###
        if (
            keyword_positive(self.answers["Treatment"], "diuretic")
            or keyword_positive(self.answers["Treatment"], "beta-blocker")
            or keyword_positive(self.answers["Treatment"], "beta blocker")
            or keyword_positive(self.answers["Treatment"], "ace inhibitor")
            or keyword_positive(self.answers["Treatment"], "statin")
            or keyword_positive(self.answers["Treatment"], "monitor")
            or keyword_positive(self.answers["Treatment"], "activity restrict")
        ):
            self.answers["Treatment Requested"]["Medications"] = True
