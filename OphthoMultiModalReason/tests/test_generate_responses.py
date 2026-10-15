import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_responses.py"
SPEC = importlib.util.spec_from_file_location("generate_responses", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GenerateResponsesTests(unittest.TestCase):
    def test_split_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "info.txt"
            path.write_text("Part A\nHistory\nPart B\nDiagnosis", encoding="utf-8")
            part_a, part_b = MODULE.split_case(path)
        self.assertEqual(part_a, "History")
        self.assertEqual(part_b, "Diagnosis")

    def test_prompts_are_distinct(self):
        direct = MODULE.prompt_text("History", "Reasoning Disabled")
        structured = MODULE.prompt_text("History", "Reasoning Enabled")
        self.assertIn("绝对禁止", direct)
        self.assertIn("## 7.【风险与随访】", structured)
        self.assertNotEqual(direct, structured)

    def test_paid_run_requires_confirmation(self):
        args = MODULE.build_parser().parse_args(["--dataset", "examples/synthetic_dataset", "--dry-run"])
        self.assertFalse(args.confirm_paid_run)


if __name__ == "__main__":
    unittest.main()
