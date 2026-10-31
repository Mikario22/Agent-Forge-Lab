import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_forge_lab.cli import main

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "synthetic_kb.json"


class CliTests(unittest.TestCase):
    def test_cli_outputs_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            args = ["agent_forge_lab", "run", str(SAMPLE), "--question",
                    "Summarize login investigations and approved maintenance",
                    "--output", str(target)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            data = json.loads(target.read_text())
            self.assertEqual(data["mode"], "offline")
            self.assertGreater(len(data["claims"]), 0)
            self.assertTrue(data["metrics"]["audit_valid"])

    def test_stdout_is_json(self):
        args = ["agent_forge_lab", "run", str(SAMPLE), "--question", "login maintenance"]
        output = io.StringIO()
        with patch("sys.argv", args), contextlib.redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue())["mode"], "offline")

    def test_offline_rejects_transfer(self):
        args = ["agent_forge_lab", "run", str(SAMPLE), "--question", "login",
                "--allow-external-data-transfer"]
        with patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    def test_openai_without_opt_in_is_rejected(self):
        args = ["agent_forge_lab", "run", str(SAMPLE), "--question", "login",
                "--provider", "openai", "--model", "example"]
        with patch("sys.argv", args):
            with self.assertRaises(PermissionError):
                main()


if __name__ == "__main__":
    unittest.main()
