#!/usr/bin/env python3
"""Agreement metrics used by the public full-cohort analysis."""

from __future__ import annotations

from itertools import combinations

from analysis_common import (
    DOMAINS,
    MEASURES,
    agreement_metrics,
    qwk,
)


def human_agreement(rows, judges):
    out = {}
    for judge in judges:
        out[judge] = {}
        for domain in DOMAINS:
            scores = [r[f"{judge}_{domain}"] for r in rows]
            consensus = [r[f"human_{domain}"] for r in rows]
            metrics = agreement_metrics(scores, consensus)
            metrics["qwk_vs_rater1"] = qwk(scores, [r[f"rater1_{domain}"] for r in rows])
            metrics["qwk_vs_rater2"] = qwk(scores, [r[f"rater2_{domain}"] for r in rows])
            out[judge][domain] = metrics
        out[judge]["total_score"] = agreement_metrics(
            [r[f"{judge}_total_score"] for r in rows],
            [r["human_total_score"] for r in rows],
        )
    return out


def pairwise_agreement(rows, judges):
    out = {}
    for first, second in combinations(judges, 2):
        pair = f"{first}_vs_{second}"
        out[pair] = {}
        for measure in MEASURES:
            a = [r[f"{first}_{measure}"] for r in rows]
            b = [r[f"{second}_{measure}"] for r in rows]
            metrics = agreement_metrics(a, b)
            if measure in DOMAINS:
                metrics["qwk"] = qwk(a, b)
            out[pair][measure] = metrics
    return out


