import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rescore_same_rubric.py"
SPEC = importlib.util.spec_from_file_location("rescore_same_rubric", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RescoreSameRubricTests(unittest.TestCase):
    def test_normalize_mode(self):
        self.assertEqual(MODULE.normalize_mode("CoT"), "Reasoning Enabled")
        self.assertEqual(MODULE.normalize_mode("Direct"), "Reasoning Disabled")
        with self.assertRaises(ValueError):
            MODULE.normalize_mode("unknown")

    def test_parse_valid_json_and_recalculate_total(self):
        raw = json.dumps(
            {
                "diagnostic_accuracy": 3,
                "data_integration": 2,
                "clinical_logic": 1,
                "total_score": 99,
                "diagnostic_accuracy_rationale": "a",
                "data_integration_rationale": "b",
                "clinical_logic_rationale": "c",
            }
        )
        score = MODULE.parse_score_response(raw)
        self.assertEqual(score["total_score"], 6)

    def test_parse_markdown_fenced_json(self):
        raw = "```json\n{\"diagnostic_accuracy\":2,\"data_integration\":3,\"clinical_logic\":2}\n```"
        score = MODULE.parse_score_response(raw)
        self.assertEqual(score["total_score"], 7)

    def test_reject_out_of_range_score(self):
        raw = '{"diagnostic_accuracy":4,"data_integration":2,"clinical_logic":2}'
        with self.assertRaises(ValueError):
            MODULE.parse_score_response(raw)

    def test_part_a_excludes_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "info.txt"
            first.write_text("Part A\nSynthetic symptom.\nPart B\n【Main Diagnosis】: Synthetic diagnosis", encoding="utf-8")
            part_a = MODULE.read_part_a(first)
        self.assertNotIn("Part B", part_a)
        self.assertNotRegex(part_a, r"【(?:主诊断|Main Diagnosis)】")

    def test_paid_run_flag_is_required_by_parser(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(["run", "--judge-model", "test/model"])
        self.assertFalse(args.confirm_paid_run)
        self.assertEqual(args.max_completion_tokens, 800)
        self.assertEqual(args.reasoning_effort, "none")
        self.assertEqual(args.max_failures, 1)
        self.assertEqual(args.run_label, "main")
        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.input_mode, "multimodal")
        self.assertEqual(args.thinking_mode, "default")
        self.assertFalse(args.skip_existing_errors)

    def test_bailian_provider_uses_dedicated_credentials(self):
        names = MODULE.PROVIDER_ENV["bailian-token-plan"]
        self.assertEqual(names, ("BAILIAN_TOKEN_PLAN_API_KEY", "BAILIAN_TOKEN_PLAN_URL"))
        names2 = MODULE.PROVIDER_ENV["bailian-token-plan-2"]
        self.assertEqual(names2, ("BAILIAN_TOKEN_PLAN_API_KEY2", "BAILIAN_TOKEN_PLAN_URL"))

    def test_text_only_messages_exclude_images(self):
        task = {"clinical_info": "a", "ground_truth": "b", "candidate_response": "c", "images": []}
        messages = MODULE.build_messages(task, "high", input_mode="text-only")
        self.assertEqual(len(messages[1]["content"]), 1)
        self.assertIn("text-only sensitivity analysis", messages[1]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
