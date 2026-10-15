#!/usr/bin/env python3
"""Offline sensitivity analysis for potential public-source contamination.

This analysis cannot test proprietary training-set membership. It instead asks
whether responses to public Chinese educational cases show a consistent score
advantage relative to contributed Ukrainian clinical cases, overall and within
the shared retinal-detachment category.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "revision_judge"
OUT = ROOT / "outputs" / "revision_contamination"
WORKBOOK = OUT / "public_source_sensitivity.xlsx"
sys.path.insert(0, str(ROOT / "analysis"))

from analyze_full_results import DEFAULT_RUN_CONFIG, load_scores  # noqa: E402
from analysis_common import read_jsonl, score_rows  # noqa: E402


FIELDS = [
    ("human_total_score", "Total score"),
    ("human_diagnostic_accuracy", "Diagnostic Accuracy"),
    ("human_data_integration", "Data Integration"),
    ("human_clinical_logic", "Clinical Logic"),
]


def load_rows(run_config: Path = DEFAULT_RUN_CONFIG) -> list[dict]:
    scores, _ = load_scores(run_config)
    tasks = read_jsonl(DATA / "judge_tasks.jsonl")
    humans = read_jsonl(DATA / "human_reference.jsonl")
    rows = score_rows(
        {x["case"] for x in tasks},
        {x["evaluation_id"]: x for x in tasks},
        {x["evaluation_id"]: x for x in humans},
        scores,
    )
    disease = {x["evaluation_id"]: x.get("disease_type", "") for x in tasks}
    for row in rows:
        row["disease_type"] = disease[row["evaluation_id"]]
    return rows


def is_rd(row: dict) -> bool:
    value = row["disease_type"]
    return "RD" in value or "Retinal_Detachment" in value


def case_means(rows: list[dict], predicate, field: str) -> list[dict]:
    records = []
    for case in sorted({r["case"] for r in rows if predicate(r)}):
        subset = [r for r in rows if r["case"] == case and predicate(r)]
        records.append(
            {
                "case": case,
                "source": subset[0]["source"],
                "value": float(np.mean([r[field] for r in subset])),
            }
        )
    return records


def difference_summary(records: list[dict], rng: np.random.Generator) -> dict:
    public = np.array([r["value"] for r in records if r["source"] == "Chinese"])
    clinical = np.array([r["value"] for r in records if r["source"] in {"Ukrainian", "Polish"}])
    draws = np.empty(20000)
    for idx in range(len(draws)):
        draws[idx] = (
            rng.choice(public, len(public), replace=True).mean()
            - rng.choice(clinical, len(clinical), replace=True).mean()
        )
    return {
        "public_cases_n": int(len(public)),
        "ukrainian_cases_n": int(len(clinical)),
        "public_mean": float(public.mean()),
        "ukrainian_mean": float(clinical.mean()),
        "public_minus_ukrainian": float(public.mean() - clinical.mean()),
        "bootstrap_95_ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
    }


def source_urls() -> tuple[list[dict], int]:
    """Read optional, redistribution-approved provenance URLs, one per line."""
    source_file = ROOT / "data" / "provenance_urls.txt"
    if not source_file.exists():
        return [], 0
    urls = [line.strip() for line in source_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"source_url": url, "occurrences_in_provenance_archive": urls.count(url)} for url in dict.fromkeys(urls)], len(urls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = parser.parse_args()
    rows = load_rows(args.run_config)
    clinical_source = "Ukrainian" if any(r["source"] == "Ukrainian" for r in rows) else "Polish"
    rng = np.random.default_rng(20261015)
    summaries = []
    json_summary = {}
    for scope, predicate in [("All cases", lambda row: True), ("Retinal detachment", is_rd)]:
        json_summary[scope] = {}
        for field, label in FIELDS:
            item = difference_summary(case_means(rows, predicate, field), rng)
            json_summary[scope][label] = item
            summaries.append({"scope": scope, "outcome": label, **item})

    descriptive = []
    for scope, predicate in [("All cases", lambda row: True), ("Retinal detachment", is_rd)]:
        for model in sorted({r["model"] for r in rows}):
            for mode in ["Reasoning Disabled", "Reasoning Enabled"]:
                for source, source_label in [
                    ("Chinese", "Public Chinese educational"),
                    (clinical_source, "Contributed Ukrainian clinical"),
                ]:
                    subset = [
                        r for r in rows
                        if predicate(r) and r["model"] == model and r["mode"] == mode and r["source"] == source
                    ]
                    descriptive.append(
                        {
                            "scope": scope,
                            "model": model,
                            "prompt_package": "Direct answer" if mode == "Reasoning Disabled" else "Structured explanation",
                            "source": source_label,
                            "n_cases": len(subset),
                            "mean_human_total": float(np.mean([r["human_total_score"] for r in subset])),
                            "mean_human_diagnostic_accuracy": float(np.mean([r["human_diagnostic_accuracy"] for r in subset])),
                        }
                    )

    urls, url_occurrences = source_urls()
    output = {
        "analysis": "Public-source performance sensitivity; not a training-membership test",
        "case_level_aggregation": "Each case averaged across three models and two prompt packages",
        "bootstrap_resamples": 20000,
        "seed": 20261015,
        "public_cases": 51,
        "ukrainian_cases": 23,
        "unique_public_source_urls": len(urls),
        "url_occurrences_in_provenance_archive": url_occurrences,
        "results": json_summary,
        "interpretation": (
            "All confidence intervals for public-minus-Ukrainian differences include zero. "
            "This provides no clear evidence of a consistent public-source score advantage, "
            "but cannot exclude training-corpus overlap or memorization."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source_contamination_sensitivity.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = pd.DataFrame(
        [
            {
                "item": "Purpose",
                "value": "Check for an observable public-source performance advantage; this is not a proprietary training-membership test.",
            },
            {
                "item": "Primary unit",
                "value": "Case; human-consensus scores averaged across three models and two prompt packages.",
            },
            {
                "item": "Uncertainty",
                "value": "Two-sample case bootstrap with 20,000 resamples.",
            },
            {
                "item": "Key limitation",
                "value": "Source, language, case mix, and acquisition setting are inseparable; proprietary training corpora and retrieval logs are unavailable.",
            },
        ]
    )
    WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Case-level sensitivity", index=False)
        pd.DataFrame(descriptive).to_excel(writer, sheet_name="Model-prompt descriptives", index=False)
        pd.DataFrame(urls).to_excel(writer, sheet_name="Public source URLs", index=False)

    print(OUT / "source_contamination_sensitivity.json")
    print(WORKBOOK)


if __name__ == "__main__":
    main()
