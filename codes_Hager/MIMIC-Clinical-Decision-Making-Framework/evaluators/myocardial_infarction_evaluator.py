from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    PCI_PROCEDURES_ICD9,
    PCI_PROCEDURES_ICD10,
    PCI_PROCEDURES_KEYWORDS,
    ALTERNATE_PCI_KEYWORDS,
    CABG_PROCEDURES_ICD9,
    CABG_PROCEDURES_ICD10,
    CABG_PROCEDURES_KEYWORDS,
    ALTERNATE_CABG_KEYWORDS,
)


class MyocardialInfarctionEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of myocardial infarction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "myocardial infarction"
        self.alternative_pathology_names = [
            {
                "location": "myocardi",
                "modifiers": ["infarc"],
            },
            {
                "location": "coronary",
                "modifiers": ["syndrome"],
            },
            {
                "location": "stemi",
                "modifiers": ["stemi"],  # self-match for single-word diagnosis
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "nstemi", "modifiers": ["nstemi"]},
            {"location": "heart", "modifiers": ["attack"]},
            {"location": "acute coronary", "modifiers": ["syndrome"]},
        ]

        self.required_lab_tests = {
            "Cardiac": [
                51003,  # Troponin T
                51002,  # Troponin I
                52642,  # Troponin I (alternate itemid)
            ],
            "Inflammation": INFLAMMATION_LAB_TESTS,
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
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend([
            50911,  # CK-MB (supplementary, not required per 2023 ESC)
            51196,  # D-Dimer (rule out PE)
            50963,  # NTproBNP (rule out CHF)
        ])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Cardiac"]
            and t not in self.required_lab_tests["Inflammation"]
        ]

        self.answers["Treatment Requested"] = {
            "PCI_or_CABG": False,
            "Anticoagulation": False,
            "Medications": False,
        }
        self.answers["Treatment Required"] = {
            "PCI_or_CABG": False,
            "Anticoagulation": True,
            "Medications": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        if region == "Chest":
            # ECG is the first-line test for MI (ST changes, Q waves)
            if modality == "ECG":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # Echocardiogram is supportive (wall motion) but not diagnostic for MI
            if modality == "Echocardiogram":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # CT / Radiograph can be useful for ruling out other causes
            if modality in ("CT", "Radiograph"):
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### PCI or CABG ###
        if (
            procedure_checker(PCI_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(PCI_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(PCI_PROCEDURES_KEYWORDS, self.procedures_discharge)
            or procedure_checker(CABG_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(CABG_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(CABG_PROCEDURES_KEYWORDS, self.procedures_discharge)
        ):
            self.answers["Treatment Required"]["PCI_or_CABG"] = True

        if (
            procedure_checker(PCI_PROCEDURES_KEYWORDS, [self.answers["Treatment"]])
            or treatment_alternative_procedure_checker(
                ALTERNATE_PCI_KEYWORDS, self.answers["Treatment"]
            )
            or procedure_checker(CABG_PROCEDURES_KEYWORDS, [self.answers["Treatment"]])
            or treatment_alternative_procedure_checker(
                ALTERNATE_CABG_KEYWORDS, self.answers["Treatment"]
            )
        ):
            self.answers["Treatment Requested"]["PCI_or_CABG"] = True

        ### ANTICOAGULATION ###
        if (
            keyword_positive(self.answers["Treatment"], "anticoagul")
            or keyword_positive(self.answers["Treatment"], "heparin")
            or keyword_positive(self.answers["Treatment"], "enoxaparin")
            or keyword_positive(self.answers["Treatment"], "fondaparinux")
            or keyword_positive(self.answers["Treatment"], "bivalirudin")
            or keyword_positive(self.answers["Treatment"], "aspirin")
            or keyword_positive(self.answers["Treatment"], "clopidogrel")
            or keyword_positive(self.answers["Treatment"], "ticagrelor")
            or keyword_positive(self.answers["Treatment"], "prasugrel")
            or keyword_positive(self.answers["Treatment"], "antiplatelet")
            or keyword_positive(self.answers["Treatment"], "dual antiplatelet")
            or keyword_positive(self.answers["Treatment"], "p2y12")
            or keyword_positive(self.answers["Treatment"], "blood thinner")
        ):
            self.answers["Treatment Requested"]["Anticoagulation"] = True

        ### MEDICATIONS ###
        if (
            keyword_positive(self.answers["Treatment"], "beta-blocker")
            or keyword_positive(self.answers["Treatment"], "beta blocker")
            or keyword_positive(self.answers["Treatment"], "statin")
            or keyword_positive(self.answers["Treatment"], "ace inhibitor")
            or keyword_positive(self.answers["Treatment"], "angiotensin")
            or keyword_positive(self.answers["Treatment"], "nitroglycerin")
            or keyword_positive(self.answers["Treatment"], "oxygen")
            or keyword_positive(self.answers["Treatment"], "metoprolol")
            or keyword_positive(self.answers["Treatment"], "atorvastatin")
        ):
            self.answers["Treatment Requested"]["Medications"] = True
