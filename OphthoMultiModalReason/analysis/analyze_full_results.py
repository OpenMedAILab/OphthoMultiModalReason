#!/usr/bin/env python3
"""Analyze all 74 cases with the final independent-evaluator panel."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agreement_metrics import human_agreement, pairwise_agreement
from analysis_common import (
    DATA,
    complete_cases,
    cot_effects,
    model_condition_means,
    model_rankings,
    read_jsonl,
    score_rows,
    source_cot_total,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "full_analysis_summary.json"
DEFAULT_RUN_CONFIG = ROOT / "configs" / "published_run_map.json"


def load_scores(run_config: Path = DEFAULT_RUN_CONFIG):
    judge_runs = json.loads(run_config.read_text(encoding="utf-8"))["judge_runs"]
    scores, technical = {}, {}
    for judge, run_dirs in judge_runs.items():
        by_id, errors, usage, configs = {}, [], defaultdict(int), []
        for run, dirname in run_dirs:
            folder = DATA / "results" / run / dirname
            configs.append(json.loads((folder / "run_config.json").read_text())["parameters"])
            for path in folder.glob("*.json"):
                if path.name == "run_config.json" or path.name.endswith(".error.json"):
                    continue
                item = json.loads(path.read_text())
                eid = item["evaluation_id"]
                if eid in by_id:
                    raise ValueError(f"Duplicate result for {judge}: {eid}")
                by_id[eid] = item
                for key, value in (item.get("usage") or {}).items():
                    if isinstance(value, int):
                        usage[key] += value
            errors.extend(json.loads(path.read_text()) for path in folder.glob("*.error.json"))
        scores[judge] = by_id
        technical[judge] = {
            "score_files": len(by_id),
            "error_files": len(errors),
            "successful_call_usage": dict(usage),
            "run_parameters": configs,
        }
    return scores, technical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    tasks = read_jsonl(DATA / "judge_tasks.jsonl")
    humans = read_jsonl(DATA / "human_reference.jsonl")
    task_by_id = {x["evaluation_id"]: x for x in tasks}
    human_by_id = {x["evaluation_id"]: x for x in humans}
    all_cases = {x["case"] for x in tasks}
    scores, technical = load_scores(args.run_config)
    complete = {
        judge: complete_cases(all_cases, task_by_id, values)
        for judge, values in scores.items()
    }
    if not all(len(values) == 74 for values in complete.values()):
        raise RuntimeError(
            "Full analysis gate failed; every judge must have all 74 complete cases: "
            + json.dumps({k: len(v) for k, v in complete.items()})
        )
    common = sorted(set.intersection(*(set(values) for values in complete.values())))
    rows = score_rows(set(common), task_by_id, human_by_id, scores)
    if len(rows) != 444:
        raise RuntimeError(f"Expected 444 common rows, found {len(rows)}")
    judges = list(scores)
    rng = np.random.default_rng(20261014)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_74_case_replacement_judge_analysis",
        "analysis_roles": {
            "primary_independent_multimodal_judges": ["qwen", "claude"],
            "secondary_text_only_sensitivity_judge": "deepseek_text",
            "legacy_exploratory_judges_excluded": True,
        },
        "selection": {
            "cases": len(common),
            "responses": len(rows),
            "complete_cases_by_judge": {k: len(v) for k, v in complete.items()},
        },
        "technical": technical,
        "human_agreement": human_agreement(rows, judges),
        "pairwise_judge_agreement": pairwise_agreement(rows, judges),
        "cot_minus_direct": {
            scorer: cot_effects(rows, scorer, rng)
            for scorer in ["human", *judges]
        },
        "source_stratified_cot_total": {
            scorer: source_cot_total(rows, scorer, rng)
            for scorer in ["human", *judges]
        },
        "model_rankings": {
            scorer: model_rankings(rows, scorer)
            for scorer in ["human", *judges]
        },
        "model_condition_means": {
            scorer: model_condition_means(rows, scorer)
            for scorer in ["human", *judges]
        },
        "interpretation_guardrails": [
            "Qwen and Claude are the primary multimodal judges and are independent of all evaluated model vendors.",
            "DeepSeek is text-only and is a sensitivity analysis only.",
            "DeepSeek data-integration scores do not independently verify image findings.",
            "Case is the resampling and pairing unit for condition contrasts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(common), "responses": len(rows)}))


if __name__ == "__main__":
    main()
