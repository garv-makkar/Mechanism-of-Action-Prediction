import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from moa.data import PROJECT_ROOT, data_directory, validate_data
from moa.__main__ import EXPERIMENTS


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.write("train_features.csv", [["sig_id", "cp_type", "cp_time", "cp_dose", "g-0"], ["a", "trt_cp", "24", "D1", "0.5"], ["b", "trt_cp", "48", "D2", "1.5"]])
        self.targets([["sig_id", "label"], ["a", "0"], ["b", "1"]])

    def write(self, name, rows):
        with (self.folder / name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    def targets(self, rows):
        self.write("train_targets_scored.csv", rows)

    def test_valid_and_path_precedence(self):
        with patch.dict(os.environ, {"MOA_DATA_DIR": str(self.folder)}):
            self.assertEqual(data_directory(), self.folder)
            self.assertEqual(data_directory(PROJECT_ROOT), PROJECT_ROOT)
            self.assertEqual(validate_data()["rows"], 2)
        self.assertEqual(validate_data(self.folder)["targets"], 1)

    def test_invalid_inputs(self):
        for rows in [
            [["sig_id", "label"], ["b", "1"], ["a", "0"]],
            [["sig_id", "label"], ["a", "1"], ["a", "0"]],
            [["sig_id", "label"], ["a", "2"], ["b", "0"]],
            [["sig_id", "label"]],
            [["sig_id", "label"], ["a", "1", "extra"]],
            [["label"], ["0"]],
        ]:
            with self.subTest(rows=rows):
                self.targets(rows)
                with self.assertRaises(ValueError):
                    validate_data(self.folder)

    def test_missing_file(self):
        (self.folder / "train_features.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_data(self.folder)

    def test_cli_check(self):
        result = subprocess.run([sys.executable, "-m", "moa", "check", "--data-dir", str(self.folder)], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Data OK: 2 rows", result.stdout)
        (self.folder / "train_features.csv").unlink()
        result = subprocess.run([sys.executable, "-m", "moa", "check", "--data-dir", str(self.folder)], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing", result.stderr)


class RepositoryChecks(unittest.TestCase):
    def test_all_experiments_exist_and_compile(self):
        for relative in EXPERIMENTS.values():
            path = PROJECT_ROOT / relative
            self.assertTrue(path.is_file(), path)
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                for i, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), f"{path}:{i}", "exec")
            else:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(callable(module.main))

    def test_list_without_data(self):
        result = subprocess.run([sys.executable, "-m", "moa", "list"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in EXPERIMENTS:
            self.assertIn(name, result.stdout)


if __name__ == "__main__":
    unittest.main()
