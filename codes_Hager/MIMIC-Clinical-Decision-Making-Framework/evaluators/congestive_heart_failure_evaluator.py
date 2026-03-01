from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import keyword_positive


class CongestiveHeartFailureEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of congestive heart failure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "congestive heart failure"
        self.alternative_pathology_names = [
            {
                "location": "heart",
                "modifiers": ["failure"],
            },
            {
                "location": "cardiac",
                "modifiers": ["failure"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "chf", "modifiers": ["chf"]},
            {"location": "heart", "modifiers": ["decompensat"]},
            {"location": "pulmonary", "modifiers": ["edema"]},
            {"location": "cardiac", "modifiers": ["decompensat"]},
        ]

        self.required_lab_tests = {
            "Cardiac": [
                50963,  # NTproBNP / BNP
            ],
            "Renal": [
                50912,  # Creatinine
                52024,
                52546,
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
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(INFLAMMATION_LAB_TESTS)
        self.neutral_lab_tests.extend([
            51003,  # Troponin T (rule out MI)
            51002,  # Troponin I
            52642,  # Troponin I (alternate)
            50911,  # CK-MB
            51196,  # D-Dimer
        ])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Cardiac"]
            and t not in self.required_lab_tests["Renal"]
        ]

        self.answers["Treatment Requested"] = {
            "Diuretics": False,
            "ACE_ARB": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Diuretics": True,
            "ACE_ARB": True,
            "Support": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        if region == "Chest":
            # CXR is first-line for detecting pulmonary edema, cardiomegaly
            if modality == "Radiograph":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # Echocardiogram — gold standard for EF assessment
            if modality == "Echocardiogram":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # ECG — arrhythmia detection, LVH, ischemic changes
            if modality == "ECG":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # CT acceptable
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### DIURETICS ###
        if (
            keyword_positive(self.answers["Treatment"], "diuretic")
            or keyword_positive(self.answers["Treatment"], "furosemide")
            or keyword_positive(self.answers["Treatment"], "lasix")
            or keyword_positive(self.answers["Treatment"], "bumetanide")
            or keyword_positive(self.answers["Treatment"], "torsemide")
            or keyword_positive(self.answers["Treatment"], "spironolactone")
        ):
            self.answers["Treatment Requested"]["Diuretics"] = True

        ### ACE INHIBITORS / ARBs / ARNI / SGLT2i ###
        if (
            keyword_positive(self.answers["Treatment"], "ace inhibitor")
            or keyword_positive(self.answers["Treatment"], "angiotensin")
            or keyword_positive(self.answers["Treatment"], "lisinopril")
            or keyword_positive(self.answers["Treatment"], "enalapril")
            or keyword_positive(self.answers["Treatment"], "losartan")
            or keyword_positive(self.answers["Treatment"], "valsartan")
            or keyword_positive(self.answers["Treatment"], "sacubitril")
            or keyword_positive(self.answers["Treatment"], "entresto")
            or keyword_positive(self.answers["Treatment"], "sglt2")
            or keyword_positive(self.answers["Treatment"], "empagliflozin")
            or keyword_positive(self.answers["Treatment"], "dapagliflozin")
            or keyword_positive(self.answers["Treatment"], "hydralazine")
            or keyword_positive(self.answers["Treatment"], "isosorbide")
        ):
            self.answers["Treatment Requested"]["ACE_ARB"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "oxygen")
            or keyword_positive(self.answers["Treatment"], "fluid restrict")
            or keyword_positive(self.answers["Treatment"], "salt restrict")
            or keyword_positive(self.answers["Treatment"], "sodium restrict")
            or keyword_positive(self.answers["Treatment"], "beta-blocker")
            or keyword_positive(self.answers["Treatment"], "beta blocker")
            or keyword_positive(self.answers["Treatment"], "carvedilol")
            or keyword_positive(self.answers["Treatment"], "metoprolol")
            or keyword_positive(self.answers["Treatment"], "bisoprolol")
            or keyword_positive(self.answers["Treatment"], "monitor")
        ):
            self.answers["Treatment Requested"]["Support"] = True
