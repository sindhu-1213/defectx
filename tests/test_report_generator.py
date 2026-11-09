import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src import database, report_generator


class InspectionReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "reports.sqlite3"
        database.initialize_database(self.db_path)
        database.create_batch(
            "batch-report",
            "2026-11-10T00:00:00+00:00",
            1,
            0.8,
            db_path=self.db_path,
        )
        self.inspection_id = database.save_inspection(
            {
                "batch_id": "batch-report",
                "inspected_at": "2026-11-10T00:00:01+00:00",
                "image_filename": "candle_sample.png",
                "product_category": "candle",
                "total_objects": 1,
                "normal_objects": 0,
                "defective_objects": 1,
                "uncertain_objects": 0,
                "product_decision": "FAIL",
                "decision_explanation": "One object is confidently defective.",
                "classification_confidences": "object 1 (candle): DEFECTIVE 0.950 confidence",
                "object_details": [
                    {
                        "class_name": "candle",
                        "predicted_class": "DEFECTIVE",
                        "confidence": 0.95,
                        "status": "defective",
                    }
                ],
                "yolo_ms": 10.1,
                "mobilenet_ms": 5.2,
                "total_inference_ms": 17.0,
                "decision_confidence_threshold": 0.8,
            },
            original_image=Image.new("RGB", (24, 16), "white"),
            annotated_image=Image.new("RGB", (24, 16), "red"),
            db_path=self.db_path,
        )
        database.complete_batch(
            "batch-report",
            successful_images=1,
            pass_count=0,
            fail_count=1,
            review_count=0,
            processing_errors=0,
            defective_objects=1,
            classified_objects=1,
            db_path=self.db_path,
        )

    def test_individual_pdf_contains_inspection_fields_and_images(self):
        record = database.get_inspection(
            self.inspection_id, db_path=self.db_path
        )
        output = Path(self.temp_dir.name) / "individual.pdf"
        returned_path = Path(
            report_generator.generate_inspection_pdf(record, output)
        )
        pdf = returned_path.read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF-"))
        for expected in (
            b"DefectX",
            b"Digital Inspection Report",
            b"candle_sample.png",
            b"FAIL",
            b"One object is confidently defective.",
            b"MobileNetV3 time",
            b"Human verification",
            b"DEFECTIVE",
        ):
            self.assertIn(expected, pdf)
        self.assertTrue(record["original_image_path"])
        self.assertTrue(record["annotated_image_path"])

    def test_batch_pdf_contains_summary_and_image_decisions(self):
        batch = database.get_batch("batch-report", db_path=self.db_path)
        records = database.list_batch_inspections(
            "batch-report", db_path=self.db_path
        )
        output = Path(self.temp_dir.name) / "batch.pdf"
        pdf = Path(report_generator.generate_batch_pdf(batch, records, output)).read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF-"))
        for expected in (
            b"Consolidated Batch Inspection Report",
            b"batch-report",
            b"Total images",
            b"Object defect rate",
            b"candle_sample.png",
            b"One object is confidently defective.",
        ):
            self.assertIn(expected, pdf)


if __name__ == "__main__":
    unittest.main()
