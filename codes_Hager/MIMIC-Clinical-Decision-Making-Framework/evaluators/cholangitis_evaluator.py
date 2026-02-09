from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    ERCP_PROCEDURES_ICD9,
    ERCP_PROCEDURES_ICD10,
    ERCP_PROCEDURES_KEYWORDS,
    CHOLECYSTECTOMY_PROCEDURES_ICD9,
    CHOLECYSTECTOMY_PROCEDURES_ICD10,
    CHOLECYSTECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_CHOLECYSTECTOMY_KEYWORDS,
    BILIARY_DRAINAGE_PROCEDURES_ICD9,
    BILIARY_DRAINAGE_PROCEDURES_ICD10,
    BILIARY_DRAINAGE_PROCEDURES_KEYWORDS,
    ALTERNATE_BILIARY_DRAINAGE_KEYWORDS,
)


class CholangitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of cholangitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "cholangitis"
        self.alternative_pathology_names = [
            {
                "location": "cholang",
                "modifiers": [
                    "inflam",
                    "infect",
                    "abscess",
                    "obstruct",
                    "septic",
                    "ascending",
                ],
            },
            {
                "location": "bile duct",
                "modifiers": [
                    "infect",
                    "inflam",
                    "obstruct",
                    "septic",
                ],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "biliary", "modifiers": ["sepsis"]},
            {"location": "biliary", "modifiers": ["obstruct"]},
        ]

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Liver": [
                50861,  # "Alanine Aminotransferase (ALT)",
                50878,  # "Asparate Aminotransferase (AST)",
            ],
            "Biliary": [
                50883,  # "Bilirubin, Direct",
                50884,  # "Bilirubin, Indirect",
                50885,  # "Bilirubin, Total",
                50863,  # "Alkaline Phosphatase",
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests.extend(
            [
                50927,  # "Gamma Glutamyltransferase (GGT)"
            ]
        )
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Liver"]
            and t not in self.required_lab_tests["Biliary"]
        ]

        self.answers["Treatment Requested"] = {
            "ERCP": False,
            "Biliary Drainage": False,
            "Cholecystectomy": False,
            "Antibiotics": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "ERCP": False,
            "Biliary Drainage": False,
            "Cholecystectomy": False,
            "Antibiotics": True,
            "Support": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # Preferred is US for ductal assessment
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT is acceptable
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # MRI/MRCP is acceptable
            if modality == "MRI" or modality == "MRCP":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # ERCP counts as both diagnostic and therapeutic
            if modality == "ERCP":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### ERCP ###
        if (
            procedure_checker(ERCP_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(ERCP_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(ERCP_PROCEDURES_KEYWORDS, self.procedures_discharge)
        ):
            self.answers["Treatment Required"]["ERCP"] = True

        if procedure_checker(ERCP_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]):
            self.answers["Treatment Requested"]["ERCP"] = True

        ### BILIARY DRAINAGE ###
        if (
            procedure_checker(BILIARY_DRAINAGE_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                BILIARY_DRAINAGE_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                BILIARY_DRAINAGE_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Biliary Drainage"] = True

        if procedure_checker(
            BILIARY_DRAINAGE_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_BILIARY_DRAINAGE_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Biliary Drainage"] = True

        ### CHOLECYSTECTOMY ###
        if (
            procedure_checker(CHOLECYSTECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Cholecystectomy"] = True

        if procedure_checker(
            CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Cholecystectomy"] = True

        ### ANTIBIOTICS ###
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True
