"""Data-free tests: one version string everywhere, and the error log location (no Qt window needed)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import session  # noqa: E402
import version  # noqa: E402
from app import errors, help_text  # noqa: E402


class TestVersion(unittest.TestCase):
    def test_one_version_everywhere(self):
        self.assertEqual(version.VERSION, "0.2.0")
        self.assertIn(f"Version {version.VERSION}\n", help_text.ABOUT_TEXT)
        self.assertEqual(help_text.VERSION, version.VERSION)
        self.assertEqual(session.SOFTWARE_VERSION, version.VERSION)
        result = {"sample_id": "s", "status": "ok", "confidence": "high", "reference_modality": "xenium",
                  "reference_pixel_size_um": None, "warnings": [], "diagnostics": [],
                  "matrices": {"he": np.eye(3)}}
        with tempfile.TemporaryDirectory() as tmp:
            session.save_session(Path(tmp) / "a.json", {}, {}, result)
            saved = json.loads((Path(tmp) / "a.json").read_text(encoding="utf-8"))
            session.export_result(result, Path(tmp) / "out")
            params = json.loads((Path(tmp) / "out" / "params.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["software_version"], version.VERSION)
        self.assertEqual(params["software_version"], version.VERSION)
        # the app passes version.VERSION to the summary rows (checked against the real CSV in test_app_smoke)
        row = session.not_run_row("s", "not run", None, version.VERSION)
        self.assertEqual(row["tool_version"], version.VERSION)


class TestLogPath(unittest.TestCase):
    def tearDown(self):
        errors.set_output_folder(None)
        os.environ.pop("SPATIAL_REGISTRATION_CONFIG", None)

    def test_log_in_output_folder_logs_subfolder_created_on_first_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors.set_output_folder(tmp)
            expected = Path(tmp) / "logs" / "gui_errors.log"
            self.assertEqual(errors.log_path(), expected)
            self.assertFalse(expected.parent.exists())
            errors.log_text("ctx", "hello")
            self.assertIn("hello", expected.read_text(encoding="utf-8"))

    def test_log_in_config_folder_without_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SPATIAL_REGISTRATION_CONFIG"] = str(Path(tmp) / "cfg" / "settings.json")
            errors.set_output_folder(None)
            self.assertEqual(errors.log_path(), Path(tmp) / "cfg" / "gui_errors.log")
            errors.log_text("ctx", "hi")
            self.assertTrue((Path(tmp) / "cfg" / "gui_errors.log").exists())


if __name__ == "__main__":
    unittest.main()
