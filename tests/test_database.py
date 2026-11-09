import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src import database


class InspectionDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "history.sqlite3"
        database.initialize_database(self.db_path)

    def _create_batch(self, batch_id):
        database.create_batch(
            batch_id,
            "2026-11-10T00:00:00+00:00",
            1,
            0.8,
            db_path=self.db_path,
        )

    def _save_record(self, batch_id, filename, decision="PASS"):
        return database.save_inspection(
            {
                "batch_id": batch_id,
                "inspected_at": "2026-11-10T00:00:01+00:00",
                "image_filename": filename,
                "product_category": "candle",
                "total_objects": 1,
                "normal_objects": int(decision == "PASS"),
                "defective_objects": int(decision == "FAIL"),
                "uncertain_objects": int(decision == "MANUAL REVIEW"),
                "product_decision": decision,
                "decision_explanation": f"Saved {decision} decision.",
                "classification_confidences": "object 1 (candle): NORMAL 0.900 confidence",
                "object_details": [
                    {
                        "class_name": "candle",
                        "predicted_class": "NORMAL",
                        "confidence": 0.9,
                        "status": "normal",
                    }
                ],
                "yolo_ms": 12.5,
                "mobilenet_ms": 4.5,
                "total_inference_ms": 18.0,
                "decision_confidence_threshold": 0.8,
            },
            original_image=Image.new("RGB", (8, 8), "white"),
            annotated_image=Image.new("RGB", (8, 8), "green"),
            db_path=self.db_path,
        )

    def test_database_creation_and_inspection_round_trip(self):
        self.assertTrue(self.db_path.is_file())
        with sqlite3.connect(self.db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertIn("inspection_batches", tables)
        self.assertIn("inspections", tables)

        self._create_batch("batch-a")
        inspection_id = self._save_record("batch-a", "candle_normal.png")
        record = database.get_inspection(inspection_id, db_path=self.db_path)
        self.assertIsNotNone(record)
        self.assertEqual(record["product_decision"], "PASS")
        self.assertEqual(record["total_objects"], 1)
        self.assertEqual(record["object_details"][0]["confidence"], 0.9)
        self.assertTrue(Path(record["original_image_path"]).is_file())
        self.assertTrue(Path(record["annotated_image_path"]).is_file())

    def test_unique_ids_search_and_decision_filter(self):
        self._create_batch("batch-search")
        first_id = self._save_record("batch-search", "candle_normal.png", "PASS")
        second_id = self._save_record("batch-search", "candle_defect.png", "FAIL")
        self.assertNotEqual(first_id, second_id)

        by_id = database.list_inspections(
            search=first_id, db_path=self.db_path
        )
        by_filename = database.list_inspections(
            search="candle_defect", db_path=self.db_path
        )
        only_fail = database.list_inspections(
            decision="FAIL", db_path=self.db_path
        )
        self.assertEqual([row["inspection_id"] for row in by_id], [first_id])
        self.assertEqual(by_filename[0]["image_filename"], "candle_defect.png")
        self.assertEqual([row["product_decision"] for row in only_fail], ["FAIL"])

    def test_batch_and_inspection_persist_across_connections(self):
        self._create_batch("batch-persistent")
        inspection_id = self._save_record(
            "batch-persistent", "candle_normal.png"
        )
        database.complete_batch(
            "batch-persistent",
            successful_images=1,
            pass_count=1,
            fail_count=0,
            review_count=0,
            processing_errors=0,
            defective_objects=0,
            classified_objects=1,
            db_path=self.db_path,
        )

        reopened_record = database.get_inspection(
            inspection_id, db_path=self.db_path
        )
        reopened_batch = database.get_batch(
            "batch-persistent", db_path=self.db_path
        )
        self.assertEqual(reopened_record["image_filename"], "candle_normal.png")
        self.assertEqual(reopened_batch["status"], "COMPLETED")
        self.assertEqual(reopened_batch["pass_count"], 1)
        self.assertEqual(
            len(database.list_batch_inspections("batch-persistent", self.db_path)),
            1,
        )
        restarted = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from src.database import get_inspection; "
                    "record = get_inspection(sys.argv[2], sys.argv[1]); "
                    "print(record['image_filename'])"
                ),
                str(self.db_path),
                inspection_id,
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(restarted.stdout.strip(), "candle_normal.png")


if __name__ == "__main__":
    unittest.main()
