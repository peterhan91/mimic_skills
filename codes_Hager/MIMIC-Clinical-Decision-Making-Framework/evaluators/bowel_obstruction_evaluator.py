from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    COLECTOMY_PROCEDURES_ICD9,
    COLECTOMY_PROCEDURES_ICD10,
    COLECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_COLECTOMY_KEYWORDS,
    BOWEL_RESECTION_PROCEDURES_ICD9,
    BOWEL_RESECTION_PROCEDURES_ICD10,
    BOWEL_RESECTION_PROCEDURES_KEYWORDS,
    ALTERNATE_BOWEL_RESECTION_KEYWORDS,
    ADHESIOLYSIS_PROCEDURES_ICD9,
    ADHESIOLYSIS_PROCEDURES_ICD10,
    ADHESIOLYSIS_PROCEDURES_KEYWORDS,
    ALTERNATE_ADHESIOLYSIS_KEYWORDS,
    LAPAROTOMY_PROCEDURES_ICD9,
    LAPAROTOMY_PROCEDURES_ICD10,
    LAPAROTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_LAPAROTOMY_KEYWORDS,
    STOMA_PROCEDURES_ICD9,
    STOMA_PROCEDURES_ICD10,
    STOMA_PROCEDURES_KEYWORDS,
    ALTERNATE_STOMA_KEYWORDS,
)


class BowelObstructionEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of bowel obstruction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "bowel obstruction"
        self.alternative_pathology_names = [
            {
                "location": "bowel",
                "modifiers": ["obstruct"],
            },
            {
                "location": "intestin",
                "modifiers": ["obstruct"],
            },
            {
                # "ileus" is its own entity — match as location containing itself
                "location": "ileus",
                "modifiers": ["ileus"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "small bowel", "modifiers": ["obstruct"]},
            {"location": "large bowel", "modifiers": ["obstruct"]},
            {"location": "bowel", "modifiers": ["adhesion"]},
        ]

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Electrolytes": [
                50824,  # "Sodium" (Whole Blood)
                50983,  # "Sodium"
                52623,  # "Sodium"
                50822,  # "Potassium" (Whole Blood)
                50971,  # "Potassium"
                52610,  # "Potassium"
                50806,  # "Chloride" (Whole Blood)
                50902,  # "Chloride"
                52535,  # "Chloride"
                50803,  # "Bicarbonate" (Whole Blood)
                50882,  # "Bicarbonate"
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Electrolytes"]
        ]

        self.answers["Treatment Requested"] = {
            "Support": False,
            "Surgery": False,
            "Antibiotics": False,
        }
        self.answers["Treatment Required"] = {
            "Support": True,
            "Surgery": False,
            "Antibiotics": False,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # CT is gold standard for obstruction
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # Radiograph (obstruction series — upright/supine AXR)
            if modality == "Radiograph":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # Ultrasound is second-line
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def _check_surgery_required(self) -> bool:
        """Check if any surgical procedure ICD codes are present in the patient record."""
        # Bowel resection
        if (
            procedure_checker(BOWEL_RESECTION_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                BOWEL_RESECTION_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                BOWEL_RESECTION_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            return True
        # Colectomy
        if (
            procedure_checker(COLECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(COLECTOMY_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                COLECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            return True
        # Adhesiolysis
        if (
            procedure_checker(ADHESIOLYSIS_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                ADHESIOLYSIS_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                ADHESIOLYSIS_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            return True
        # Laparotomy
        if (
            procedure_checker(LAPAROTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(LAPAROTOMY_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                LAPAROTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            return True
        # Stoma creation
        if (
            procedure_checker(STOMA_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(STOMA_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                STOMA_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            return True
        return False

    def _check_surgery_requested(self) -> bool:
        """Check if the agent's treatment text mentions any surgical procedure."""
        treatment = self.answers["Treatment"]
        # Bowel resection
        if procedure_checker(
            BOWEL_RESECTION_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_BOWEL_RESECTION_KEYWORDS, treatment
        ):
            return True
        # Colectomy
        if procedure_checker(
            COLECTOMY_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_COLECTOMY_KEYWORDS, treatment
        ):
            return True
        # Adhesiolysis
        if procedure_checker(
            ADHESIOLYSIS_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_ADHESIOLYSIS_KEYWORDS, treatment
        ):
            return True
        # Laparotomy
        if procedure_checker(
            LAPAROTOMY_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_LAPAROTOMY_KEYWORDS, treatment
        ):
            return True
        # Stoma
        if procedure_checker(
            STOMA_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_STOMA_KEYWORDS, treatment
        ):
            return True
        return False

    def score_treatment(self) -> None:
        ### SUPPORT ###
        # Conservative management is the primary treatment for bowel obstruction
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### SURGERY ###
        # Surgery only required if ICD codes show it was actually performed
        if self._check_surgery_required():
            self.answers["Treatment Required"]["Surgery"] = True

        if self._check_surgery_requested():
            self.answers["Treatment Requested"]["Surgery"] = True

        ### ANTIBIOTICS ###
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True
