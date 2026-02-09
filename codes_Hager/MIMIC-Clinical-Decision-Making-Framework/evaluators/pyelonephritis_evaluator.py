from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    NEPHROSTOMY_PROCEDURES_ICD9,
    NEPHROSTOMY_PROCEDURES_ICD10,
    NEPHROSTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_NEPHROSTOMY_KEYWORDS,
    URETERAL_PROCEDURES_ICD9,
    URETERAL_PROCEDURES_ICD10,
    URETERAL_PROCEDURES_KEYWORDS,
    ALTERNATE_URETERAL_KEYWORDS,
)


class PyelonephritisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of pyelonephritis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "pyelonephritis"
        self.alternative_pathology_names = [
            {
                "location": "pyelonephrit",
                "modifiers": ["pyelonephrit"],
            },
            {
                "location": "kidney",
                "modifiers": ["infect"],
            },
            {
                "location": "renal",
                "modifiers": ["infect"],
            },
            {
                "location": "renal",
                "modifiers": ["abscess"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "urosepsis", "modifiers": ["urosepsis"]},
            {"location": "urinary tract", "modifiers": ["infect"]},
            {"location": "kidney", "modifiers": ["abscess"]},
        ]

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Renal": [
                50912,  # "Creatinine",
                52024,  # "Creatinine" (Whole Blood)
                52546,  # "Creatinine"
                51006,  # "Urea Nitrogen",
                52647,  # "Urea Nitrogen"
            ],
            "Urinalysis": [
                51516,  # "WBC" (Urine)
                51487,  # "Nitrite"
                51486,  # "Leukocytes" (Leukocyte Esterase)
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
            ADDITIONAL_LAB_TEST_MAPPING["Basic Metabolic Panel (BMP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Renal"]
            and t not in self.required_lab_tests["Urinalysis"]
        ]

        self.answers["Treatment Requested"] = {
            "Antibiotics": False,
            "Support": False,
            "Drainage": False,
        }
        self.answers["Treatment Required"] = {
            "Antibiotics": True,
            "Support": True,
            "Drainage": False,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # CT is preferred — assess abscess, obstruction, emphysematous changes
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # Ultrasound screens for hydronephrosis
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # MRI is acceptable
            if modality == "MRI":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### ANTIBIOTICS ###
        # Antibiotics is THE primary treatment for pyelonephritis
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### DRAINAGE (Nephrostomy / Ureteral procedures) ###
        # Only required if ICD codes show it was actually performed
        if (
            procedure_checker(NEPHROSTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                NEPHROSTOMY_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                NEPHROSTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
            or procedure_checker(URETERAL_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                URETERAL_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                URETERAL_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Drainage"] = True

        if (
            procedure_checker(
                NEPHROSTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
            )
            or treatment_alternative_procedure_checker(
                ALTERNATE_NEPHROSTOMY_KEYWORDS, self.answers["Treatment"]
            )
            or procedure_checker(
                URETERAL_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
            )
            or treatment_alternative_procedure_checker(
                ALTERNATE_URETERAL_KEYWORDS, self.answers["Treatment"]
            )
        ):
            self.answers["Treatment Requested"]["Drainage"] = True
