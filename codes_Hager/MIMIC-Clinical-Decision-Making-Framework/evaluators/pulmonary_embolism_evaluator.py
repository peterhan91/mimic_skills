from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import keyword_positive


class PulmonaryEmbolismEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of pulmonary embolism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "pulmonary embolism"
        self.alternative_pathology_names = [
            {
                "location": "pulmonary",
                "modifiers": ["embol"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "pulmonary", "modifiers": ["thrombo"]},
            {"location": "venous", "modifiers": ["thromboembol"]},
        ]

        self.required_lab_tests = {
            "Coagulation": [
                51196,  # D-Dimer
                50915,  # D-Dimer (alternate itemid)
                52551,  # D-Dimer (alternate itemid)
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
            51003,  # Troponin T (prognostic, not diagnostic per ESC 2019)
            51002,  # Troponin I
            52642,  # Troponin I (alternate)
            50963,  # NTproBNP (prognostic)
            50911,  # CK-MB
        ])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Coagulation"]
        ]

        self.answers["Treatment Requested"] = {
            "Anticoagulation": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Anticoagulation": True,
            "Support": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        if region == "Chest":
            # CT Pulmonary Angiography is the gold standard
            if modality in ("CT", "CT Angiography"):
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CXR, Echocardiogram, ECG are supportive but not diagnostic
            if modality in ("Radiograph", "Echocardiogram", "Ultrasound", "ECG"):
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        # Lower extremity Doppler for DVT
        if region in ("Extremity", "Lower Extremity"):
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### ANTICOAGULATION ###
        if (
            keyword_positive(self.answers["Treatment"], "anticoagul")
            or keyword_positive(self.answers["Treatment"], "heparin")
            or keyword_positive(self.answers["Treatment"], "enoxaparin")
            or keyword_positive(self.answers["Treatment"], "warfarin")
            or keyword_positive(self.answers["Treatment"], "rivaroxaban")
            or keyword_positive(self.answers["Treatment"], "apixaban")
            or keyword_positive(self.answers["Treatment"], "fondaparinux")
            or keyword_positive(self.answers["Treatment"], "blood thinner")
            or keyword_positive(self.answers["Treatment"], "thrombolytic")
            or keyword_positive(self.answers["Treatment"], "thrombectomy")
            or keyword_positive(self.answers["Treatment"], "tpa")
        ):
            self.answers["Treatment Requested"]["Anticoagulation"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "oxygen")
            or keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "hemodynamic")
            or keyword_positive(self.answers["Treatment"], "monitor")
        ):
            self.answers["Treatment Requested"]["Support"] = True
