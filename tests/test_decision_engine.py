import unittest

from src.decision_engine import ObjectPrediction, decide_product


KNOWN_CLASSES = {"candle", "pcb2"}
CLASSIFIER_THRESHOLD = 0.50927734375
DECISION_THRESHOLD = 0.80


def decide(*predictions, processing_error=None):
    return decide_product(
        predictions,
        classifier_threshold=CLASSIFIER_THRESHOLD,
        decision_confidence_threshold=DECISION_THRESHOLD,
        known_classes=KNOWN_CLASSES,
        processing_error=processing_error,
    )


class DecisionEngineTests(unittest.TestCase):
    def test_all_normal_objects_pass(self):
        result = decide(
            ObjectPrediction("candle", 0.10),
            ObjectPrediction("pcb2", 0.20),
        )
        self.assertEqual(result.product_decision, "PASS")
        self.assertEqual(result.normal_objects, 2)
        self.assertEqual(result.defective_objects, 0)
        self.assertEqual(result.uncertain_objects, 0)

    def test_one_confident_defect_fails(self):
        result = decide(ObjectPrediction("candle", 0.95))
        self.assertEqual(result.product_decision, "FAIL")
        self.assertEqual(result.defective_objects, 1)

    def test_confident_defect_takes_priority_and_discloses_uncertainty(self):
        result = decide(
            ObjectPrediction("candle", 0.95),
            ObjectPrediction("pcb2", 0.55),
        )
        self.assertEqual(result.product_decision, "FAIL")
        self.assertEqual(result.defective_objects, 1)
        self.assertEqual(result.uncertain_objects, 1)
        self.assertIn("Uncertainty", result.decision_explanation)

    def test_uncertain_normal_prediction_requires_review(self):
        result = decide(ObjectPrediction("candle", 0.30))
        self.assertEqual(result.product_decision, "MANUAL REVIEW")
        self.assertEqual(result.uncertain_objects, 1)

    def test_uncertain_defect_prediction_requires_review(self):
        result = decide(ObjectPrediction("candle", 0.60))
        self.assertEqual(result.product_decision, "MANUAL REVIEW")
        self.assertEqual(result.uncertain_objects, 1)

    def test_zero_objects_requires_review(self):
        result = decide()
        self.assertEqual(result.product_decision, "MANUAL REVIEW")

    def test_unknown_class_requires_review(self):
        result = decide(ObjectPrediction("unknown-product", 0.05))
        self.assertEqual(result.product_decision, "MANUAL REVIEW")
        self.assertEqual(result.uncertain_objects, 1)

    def test_invalid_or_missing_confidence_requires_review(self):
        for probability in (None, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(probability=probability):
                result = decide(ObjectPrediction("candle", probability))
                self.assertEqual(result.product_decision, "MANUAL REVIEW")
                self.assertEqual(result.uncertain_objects, 1)

    def test_missing_class_requires_review(self):
        result = decide(ObjectPrediction(None, 0.1))
        self.assertEqual(result.product_decision, "MANUAL REVIEW")

    def test_processing_error_requires_review(self):
        result = decide(ObjectPrediction("candle", 0.1), processing_error="model unavailable")
        self.assertEqual(result.product_decision, "MANUAL REVIEW")
        self.assertIn("model unavailable", result.decision_explanation)


if __name__ == "__main__":
    unittest.main()
