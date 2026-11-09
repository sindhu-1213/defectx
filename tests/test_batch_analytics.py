import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import app


class FakeInspectionModels:
    threshold = 0.50927734375
    class_to_idx = {"candle": 0}
    device = torch.device("cpu")

    def segment(self, image, confidence, iou, image_size):
        item = app.SegmentedObject(
            crop=image,
            class_name="candle",
            class_id=0,
            yolo_confidence=0.99,
            box=(0, 0, image.width, image.height),
            polygon=None,
        )
        return [item], 1.0

    def classify(self, objects):
        return np.array([0.1 if objects[0].crop.getpixel((0, 0))[0] < 128 else 0.95]), 1.0


class BatchAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.zip_path = Path(self.temp_dir.name) / "batch.zip"
        self.db_path = Path(self.temp_dir.name) / "history.sqlite3"

    def _write_batch(self, images):
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            for name, data in images:
                archive.writestr(name, data)

    @staticmethod
    def _png(red):
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (red, 0, 0)).save(buffer, format="PNG")
        return buffer.getvalue()

    def _run(self, threshold=0.8):
        with (
            patch.object(app, "MODELS", FakeInspectionModels()),
            patch.object(app, "DATABASE_PATH", self.db_path),
        ):
            outputs = list(
                app.inspect_zip(
                    str(self.zip_path),
                    0.25,
                    0.70,
                    640,
                    threshold,
                )
            )
        self.addCleanup(lambda: Path(outputs[-1][4]).unlink(missing_ok=True))
        return outputs[-1]

    def test_batch_processes_images_and_exports_decisions(self):
        self._write_batch(
            [
                ("normal.png", self._png(10)),
                ("defect.png", self._png(240)),
                ("invalid.png", b"not an image"),
            ]
        )
        output = self._run()
        rows = output[10]
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["product_decision"] for row in rows],
            ["FAIL", "MANUAL REVIEW", "PASS"],
        )
        self.assertIn("Processing errors</div>", output[5])
        self.assertIn("3</strong>", output[5])
        self.assertIn("50.00%", output[5])
        charts = app._build_batch_charts(rows)
        self.assertEqual([patch.get_height() for patch in charts[0].axes[0].patches], [1, 1, 1])
        self.assertEqual([patch.get_height() for patch in charts[1].axes[0].patches], [1, 1, 0])
        self.assertEqual([patch.get_height() for patch in charts[2].axes[0].patches], [1])
        self.assertEqual(len(charts[3].axes[0].patches), 2)

        with Path(output[4]).open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(
            [row["product_decision"] for row in csv_rows],
            ["FAIL", "MANUAL REVIEW", "PASS"],
        )
        self.assertTrue(all(row["decision_explanation"] for row in csv_rows))
        saved = app.database.list_inspections(db_path=self.db_path)
        self.assertEqual(len(saved), 3)
        self.assertEqual(
            {record["product_decision"] for record in saved},
            {"PASS", "FAIL", "MANUAL REVIEW"},
        )
        batch = app.database.list_batches(db_path=self.db_path)[0]
        self.assertEqual(batch["total_images"], 3)
        self.assertEqual(batch["processing_errors"], 1)

        with patch.object(app, "DATABASE_PATH", self.db_path):
            individual_pdf = Path(
                app.download_inspection_pdf(saved[0]["inspection_id"])
            )
            batch_pdf = Path(app.download_batch_pdf(batch["batch_id"]))
        self.addCleanup(lambda: individual_pdf.unlink(missing_ok=True))
        self.addCleanup(lambda: batch_pdf.unlink(missing_ok=True))
        self.assertTrue(individual_pdf.read_bytes().startswith(b"%PDF-"))
        self.assertTrue(batch_pdf.read_bytes().startswith(b"%PDF-"))

    def test_empty_zip_returns_empty_report_and_summary(self):
        self._write_batch([])
        output = self._run()
        self.assertEqual(output[10], [])
        self.assertIn("No supported images", output[0])
        self.assertIn("Images submitted</div>", output[5])
        with Path(output[4]).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            self.assertEqual(next(reader), app.RESULT_COLUMNS)
            self.assertEqual(list(reader), [])

    def test_filter_returns_only_requested_decisions(self):
        rows = [
            {**{column: "" for column in app.RESULT_COLUMNS}, "product_decision": decision}
            for decision in ("PASS", "FAIL", "MANUAL REVIEW")
        ]
        filtered = app.filter_batch_results("FAIL", rows)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][app.RESULT_COLUMNS.index("product_decision")], "FAIL")

    def test_single_sample_zip_saves_record_for_history(self):
        self._write_batch([("sample_candle.png", self._png(10))])
        output = self._run()
        self.assertEqual(len(output[10]), 1)
        record = app.database.list_inspections(db_path=self.db_path)[0]
        self.assertEqual(record["image_filename"], "sample_candle.png")
        self.assertTrue(Path(record["original_image_path"]).is_file())
        self.assertTrue(Path(record["annotated_image_path"]).is_file())

        with patch.object(app, "DATABASE_PATH", self.db_path):
            history = app.refresh_inspection_history()
            detail_html, original, annotated = app.view_inspection_details(
                record["inspection_id"]
            )
        self.assertIn(record["inspection_id"], history[1].get_config()["choices"][0][1])
        self.assertIn(record["product_decision"], detail_html)
        self.assertIsNotNone(original)
        self.assertIsNotNone(annotated)

    def test_sample_gallery_selection_loads_input_and_enables_inspection(self):
        archive_path, preview, message, sample_index, button = app.load_sample_on_select(
            SimpleNamespace(index=0)
        )
        self.addCleanup(lambda: Path(archive_path).unlink(missing_ok=True))
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(
                archive.namelist(), [app.SAMPLE_IMAGES[sample_index][0]]
            )
        self.assertEqual(sample_index, 0)
        self.assertGreater(preview.width, 0)
        self.assertIn("Sample loaded successfully", message)
        self.assertTrue(button.interactive)
        self.assertEqual(button.value, "INSPECT THIS SAMPLE")


if __name__ == "__main__":
    unittest.main()
