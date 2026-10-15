#!/usr/bin/env python3
"""Shared statistical functions for the public full-cohort analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "revision_judge"
DOMAINS = ["diagnostic_accuracy", "data_integration", "clinical_logic"]
MEASURES = DOMAINS + ["total_score"]
MODE_DIRECT = "Reasoning Disabled"
MODE_COT = "Reasoning Enabled"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def qwk(a, b) -> float | None:
    if len(a) < 2:
        return None
    value = cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3])
    return None if np.isnan(value) else float(value)


def rho(a, b) -> dict:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return {"rho": None, "p": None}
    r = spearmanr(a, b)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


def agreement_metrics(a, b) -> dict:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {
        "n": int(len(aa)),
        **rho(aa, bb),
        "mae": float(np.mean(np.abs(aa - bb))),
        "mean_signed_difference_first_minus_second": float(np.mean(aa - bb)),
        "exact": float(np.mean(aa == bb)),
        "within_0_5": float(np.mean(np.abs(aa - bb) <= 0.5)),
        "within_1": float(np.mean(np.abs(aa - bb) <= 1.0)),
    }


def bootstrap_mean_ci(values, rng, n_boot=10000) -> list[float]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [None, None]
    means = np.mean(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def paired_summary(deltas, rng) -> dict:
    x = np.asarray(deltas, dtype=float)
    if not len(x):
        return {"n": 0}
    if np.allclose(x, 0):
        p = 1.0
    else:
        p = float(wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
    return {
        "n": int(len(x)),
        "mean_delta": float(np.mean(x)),
        "median_delta": float(np.median(x)),
        "bootstrap_95_ci": bootstrap_mean_ci(x, rng),
        "wilcoxon_p_unadjusted": p,
        "improved_tied_worse": [int(np.sum(x > 0)), int(np.sum(x == 0)), int(np.sum(x < 0))],
    }


def complete_cases(selected_cases, task_by_id, score_by_id):
    ids_by_case = defaultdict(list)
    for eid, t in task_by_id.items():
        if t["case"] in selected_cases:
            ids_by_case[t["case"]].append(eid)
    return sorted(
        case for case, ids in ids_by_case.items()
        if len(ids) == 6 and all(eid in score_by_id for eid in ids)
    )


def score_rows(case_set, task_by_id, human_by_id, judge_scores):
    rows = []
    for eid, task in task_by_id.items():
        if task["case"] not in case_set:
            continue
        row = {
            "evaluation_id": eid,
            "case": task["case"],
            "source": task["source"],
            "model": task["model"],
            "mode": task["mode"],
        }
        h = human_by_id[eid]
        for domain in DOMAINS:
            row[f"human_{domain}"] = h["consensus"][domain]
            row[f"rater1_{domain}"] = h["rater1"][domain]
            row[f"rater2_{domain}"] = h["rater2"][domain]
        row["human_total_score"] = h["consensus_total"]
        for judge, score_map in judge_scores.items():
            s = score_map[eid]["score"]
            for measure in MEASURES:
                row[f"{judge}_{measure}"] = s[measure]
        rows.append(row)
    return rows


def cot_effects(rows, scorer, rng):
    index = {(r["case"], r["model"], r["mode"]): r for r in rows}
    result = {}
    models = sorted({r["model"] for r in rows})
    cases = sorted({r["case"] for r in rows})
    for model in models:
        result[model] = {}
        for measure in MEASURES:
            deltas = []
            for case in cases:
                enabled = index.get((case, model, MODE_COT))
                disabled = index.get((case, model, MODE_DIRECT))
                if enabled and disabled:
                    deltas.append(enabled[f"{scorer}_{measure}"] - disabled[f"{scorer}_{measure}"])
            result[model][measure] = paired_summary(deltas, rng)
    return result


def source_cot_total(rows, scorer, rng):
    result = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        # Average the three model-specific paired deltas within each case.
        idx = {(r["case"], r["model"], r["mode"]): r for r in subset}
        deltas = []
        for case in sorted({r["case"] for r in subset}):
            ds = []
            for model in sorted({r["model"] for r in subset}):
                a, b = idx.get((case, model, MODE_COT)), idx.get((case, model, MODE_DIRECT))
                if a and b:
                    ds.append(a[f"{scorer}_total_score"] - b[f"{scorer}_total_score"])
            if ds:
                deltas.append(float(np.mean(ds)))
        result[source] = paired_summary(deltas, rng)
    return result


def model_rankings(rows, scorer):
    out = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r[f"{scorer}_total_score"] for r in rows if r["model"] == model]
        out.append({"model": model, "mean_total": float(np.mean(subset)), "n": len(subset)})
    return sorted(out, key=lambda x: (-x["mean_total"], x["model"]))


def model_condition_means(rows, scorer):
    out = {}
    for model in sorted({r["model"] for r in rows}):
        out[model] = {}
        for mode in [MODE_DIRECT, MODE_COT]:
            subset = [
                r[f"{scorer}_total_score"] for r in rows
                if r["model"] == model and r["mode"] == mode
            ]
            out[model][mode] = {"mean_total": float(np.mean(subset)), "n": len(subset)}
    return out


