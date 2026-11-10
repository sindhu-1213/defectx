import unittest
from pathlib import Path
import streamlit_app

class StreamlitAppTests(unittest.TestCase):
    def test_imports_and_constants(self):
        self.assertTrue(streamlit_app.DEFAULT_YOLO.is_file())
        self.assertTrue(streamlit_app.DEFAULT_MOBILENET.is_file())
        self.assertGreater(len(streamlit_app.SAMPLE_IMAGES), 0)

    def test_sample_image_files_exist(self):
        for filename, _ in streamlit_app.SAMPLE_IMAGES:
            sample_path = streamlit_app.SAMPLE_IMAGE_DIR / filename
            self.assertTrue(sample_path.is_file(), f"Sample image missing: {sample_path}")

    def test_database_integration(self):
        records = streamlit_app.database.list_inspections(limit=5, db_path=streamlit_app.DATABASE_PATH)
        self.assertIsInstance(records, list)

if __name__ == "__main__":
    unittest.main()

