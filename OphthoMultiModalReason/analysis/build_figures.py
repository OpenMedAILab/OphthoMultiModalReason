#!/usr/bin/env python3
"""Build public, aggregate figures from authorized local evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "analysis"
sys.path.insert(0, str(CODE))

from analyze_full_results import DEFAULT_RUN_CONFIG, load_scores  # noqa: E402
from analysis_common import DATA, read_jsonl, score_rows  # noqa: E402


OUT = ROOT / "outputs" / "figures"
WORKBOOK = ROOT / "outputs" / "independent_judge_scores.xlsx"
SUMMARY = {}

SCORERS = ["human", "qwen", "claude", "deepseek_text"]
SCORER_LABELS = {
    "human": "Ophthalmologist\nconsensus",
    "qwen": "Qwen3.8-Max\n(multimodal)",
    "claude": "Claude Sonnet 5.5\n(multimodal)",
    "deepseek_text": "DeepSeek-V4-Pro\n(text only)",
}
MODELS = ["GPT-5.2", "Gemini 2.5 Pro", "Grok 4"]
COLORS = {"GPT-5.2": "#3973ac", "Gemini 2.5 Pro": "#57a773", "Grok 4": "#e38b48"}


def load_rows(run_config: Path):
    scores, _ = load_scores(run_config)
    tasks = read_jsonl(DATA / "judge_tasks.jsonl")
    humans = read_jsonl(DATA / "human_reference.jsonl")
    rows = score_rows(
        {x["case"] for x in tasks},
        {x["evaluation_id"]: x for x in tasks},
        {x["evaluation_id"]: x for x in humans},
        scores,
    )
    if len(rows) != 444:
        raise RuntimeError(f"Expected 444 rows, found {len(rows)}")
    return rows, scores


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def workflow_figure():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4, edgecolor=color, facecolor="#f8fafc"
        )
        ax.add_patch(patch)
        ax.text(x + 0.15, y + h - 0.28, title, weight="bold", color=color, fontsize=11, va="top")
        ax.text(x + 0.15, y + h - 0.72, body, fontsize=8.7, va="top", linespacing=1.35)

    box(0.25, 3.55, 3.35, 2.05, "A  Benchmark cohort",
        "74 de-identified cases\n51 Chinese educational cases\n23 Ukrainian clinical cases\nClinical vignette + multimodal images", "#2b6777")
    box(4.55, 3.55, 3.35, 2.05, "B  Frozen model responses",
        "GPT-5.2, Gemini 2.5 Pro, Grok 4\nTwo fixed prompt packages\n74 cases × 3 models × 2 conditions\n444 responses", "#3f6c9e")
    box(8.85, 3.55, 3.35, 2.05, "C  Human reference",
        "Two ophthalmologists\nThree-domain 1–3 rubric\nDiagnostic accuracy\nData integration; Clinical logic", "#9a5b39")

    box(1.25, 0.35, 4.55, 2.05, "D  Independent multimodal re-scoring",
        "Qwen3.8-Max + Claude Sonnet 5.5\nSame three-domain rubric as ophthalmologists\nAll images and clinical context provided\nBlinded to model and prompt condition", "#4c956c")
    box(6.7, 0.35, 4.55, 2.05, "E  Sensitivity and analysis",
        "DeepSeek-V4-Pro: text-only sensitivity analysis\nCase-paired CoT–Direct contrasts\nCase bootstrap 95% confidence intervals\nAgreement, ranking, and source-stratified analyses", "#7768ae")

    for x1, y1, x2, y2 in [(3.6, 4.58, 4.55, 4.58), (7.9, 4.58, 8.85, 4.58),
                            (6.25, 3.55, 4.0, 2.4), (10.5, 3.55, 9.0, 2.4)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#52616b"))
    ax.text(6.25, 6.02, "Revised evaluation workflow", ha="center", fontsize=14, weight="bold")
    return save(fig, "Figure1_revised_workflow.png")


def performance_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    axes = axes.ravel()
    width = 0.34
    x = np.arange(len(MODELS))
    for ax, scorer in zip(axes, SCORERS):
        disabled, enabled = [], []
        for model in MODELS:
            disabled.append(np.mean([r[f"{scorer}_total_score"] for r in rows if r["model"] == model and r["mode"] == "Reasoning Disabled"]))
            enabled.append(np.mean([r[f"{scorer}_total_score"] for r in rows if r["model"] == model and r["mode"] == "Reasoning Enabled"]))
        ax.bar(x - width / 2, disabled, width, label="Direct-answer package", color="#a9b7c6")
        ax.bar(x + width / 2, enabled, width, label="Structured-explanation package", color="#3f7cac")
        for pos, value in zip(x - width / 2, disabled):
            ax.text(pos, value + 0.08, f"{value:.2f}", ha="center", fontsize=7.5)
        for pos, value in zip(x + width / 2, enabled):
            ax.text(pos, value + 0.08, f"{value:.2f}", ha="center", fontsize=7.5)
        ax.set_title(SCORER_LABELS[scorer].replace("\n", " "), weight="bold")
        ax.set_ylim(0, 9.6)
        ax.set_ylabel("Mean total score (0–9)")
        ax.set_xticks(x, ["GPT-5.2", "Gemini 2.5 Pro", "Grok 4"], rotation=12)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Total scores under the two fixed prompt packages", fontsize=14, weight="bold", y=1.01)
    return save(fig, "Figure3_prompt_package_scores.png")


def agreement_figure():
    domains = ["diagnostic_accuracy", "data_integration", "clinical_logic", "total_score"]
    domain_labels = ["Diagnostic\naccuracy", "Data\nintegration", "Clinical\nlogic", "Total\nscore"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    x = np.arange(len(domains))
    width = 0.24
    judges = ["qwen", "claude", "deepseek_text"]
    colors = ["#4c956c", "#3f6c9e", "#7768ae"]
    for offset, judge, color in zip([-width, 0, width], judges, colors):
        vals = [SUMMARY["human_agreement"][judge][d]["rho"] for d in domains]
        axes[0].bar(x + offset, vals, width, label=SCORER_LABELS[judge].replace("\n", " "), color=color)
    axes[0].set_xticks(x, domain_labels)
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("Spearman ρ vs human consensus")
    axes[0].set_title("A  Agreement with ophthalmologists", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].grid(axis="y", alpha=0.2)

    pairs = ["qwen_vs_claude", "qwen_vs_deepseek_text", "claude_vs_deepseek_text"]
    pair_labels = ["Qwen–Claude", "Qwen–DeepSeek", "Claude–DeepSeek"]
    vals = [SUMMARY["pairwise_judge_agreement"][p]["total_score"]["rho"] for p in pairs]
    axes[1].bar(np.arange(3), vals, color=["#2b6777", "#6b8e23", "#8b6f9c"])
    axes[1].set_xticks(np.arange(3), pair_labels, rotation=18)
    axes[1].set_ylim(0, 0.9)
    axes[1].set_ylabel("Spearman ρ for total score")
    axes[1].set_title("B  Agreement between AI evaluators", loc="left", weight="bold")
    for i, v in enumerate(vals): axes[1].text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    matrix = np.array([[next(x["mean_total"] for x in SUMMARY["model_rankings"][s] if x["model"] == m) for m in MODELS] for s in SCORERS])
    im = axes[2].imshow(matrix, cmap="Blues", vmin=5.5, vmax=8.3, aspect="auto")
    axes[2].set_xticks(np.arange(3), ["GPT-5.2", "Gemini", "Grok"])
    axes[2].set_yticks(np.arange(4), ["Human", "Qwen", "Claude", "DeepSeek"])
    axes[2].set_title("C  Mean total-score ranking", loc="left", weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[2].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Mean total score")
    fig.tight_layout()
    return save(fig, "Figure4_agreement_and_ranking.png")


def reliability_source_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    domains = ["diagnostic_accuracy", "data_integration", "clinical_logic"]
    labels = ["Diagnostic accuracy", "Data integration", "Clinical logic"]
    human_qwk = [cohen_kappa_score([r[f"rater1_{d}"] for r in rows], [r[f"rater2_{d}"] for r in rows], weights="quadratic") for d in domains]
    qwen_claude = [SUMMARY["pairwise_judge_agreement"]["qwen_vs_claude"][d]["qwk"] for d in domains]
    for ax, vals, title, color in [
        (axes[0], human_qwk, "A  Human inter-rater reliability", "#e69f70"),
        (axes[1], qwen_claude, "B  Qwen–Claude reliability", "#56b4b9"),
    ]:
        y = np.arange(3)
        ax.barh(y, vals, color=color)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Quadratic-weighted κ")
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.2)
        for i, v in enumerate(vals): ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)

    ylabels, points, lows, highs, point_colors = [], [], [], [], []
    available_sources = SUMMARY["source_stratified_cot_total"]["human"]
    clinical_source = "Ukrainian" if "Ukrainian" in available_sources else "Polish"
    for scorer in SCORERS:
        for source in ["Chinese", clinical_source]:
            item = SUMMARY["source_stratified_cot_total"][scorer][source]
            display_source = "Chinese educational" if source == "Chinese" else "Ukrainian clinical"
            ylabels.append(f"{SCORER_LABELS[scorer].replace(chr(10), ' ')} – {display_source}")
            points.append(item["mean_delta"])
            lows.append(item["mean_delta"] - item["bootstrap_95_ci"][0])
            highs.append(item["bootstrap_95_ci"][1] - item["mean_delta"])
            point_colors.append("#3973ac" if source == "Chinese" else "#e38b48")
    y = np.arange(len(points))
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].errorbar(points, y, xerr=np.vstack([lows, highs]), fmt="none", ecolor="#555555", capsize=3, lw=1)
    axes[2].scatter(points, y, c=point_colors, s=32, zorder=3)
    axes[2].set_yticks(y, ylabels, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Structured-explanation minus direct-answer score (95% CI)")
    axes[2].set_title("C  Source-stratified sensitivity", loc="left", weight="bold")
    axes[2].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return save(fig, "Figure5_reliability_and_source_sensitivity.png")


def human_prompt_source_figures(rows):
    """Build Figure 6 and Supplementary Figure S1 from human-consensus scores."""
    data = pd.DataFrame(rows)
    data["source_label"] = data["source"].map(
        lambda value: "Chinese educational cases" if value == "Chinese" else "Ukrainian clinical cases"
    )
    sources = ["Chinese educational cases", "Ukrainian clinical cases"]
    modes = ["Reasoning Disabled", "Reasoning Enabled"]
    mode_labels = {"Reasoning Disabled": "Direct-answer", "Reasoning Enabled": "Structured-explanation"}
    domains = [
        ("human_diagnostic_accuracy", "Diagnostic\nAccuracy"),
        ("human_data_integration", "Data Integration &\nKey Information Recognition"),
        ("human_clinical_logic", "Clinical\nLogic"),
    ]

    fig = plt.figure(figsize=(8.0, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.92])
    colors = {"Direct-answer": "#4E9CB8", "Structured-explanation": "#9DBDC4"}
    for row_index, source in enumerate(sources):
        ax = fig.add_subplot(grid[row_index, :])
        x = np.arange(len(MODELS))
        width = 0.36
        for mode_index, mode in enumerate(modes):
            values, deviations = [], []
            for model in MODELS:
                subset = data[(data.source_label == source) & (data.model == model) & (data["mode"] == mode)]
                values.append(subset.human_total_score.mean())
                deviations.append(subset.human_total_score.std(ddof=1))
            label = mode_labels[mode]
            bars = ax.bar(x + (mode_index - 0.5) * width, values, width, yerr=deviations,
                          capsize=3, color=colors[label], label=label, edgecolor="white")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                        f"{value:.2f}", ha="center", fontsize=8)
        ax.set_title(f"{source}: human mean total score", weight="bold")
        ax.set_ylabel("Total score (0–9), mean ± SD")
        ax.set_xticks(x, MODELS)
        ax.set_ylim(0, 10.2)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2, loc="upper left")

    matrices, all_changes = {}, []
    for source in sources:
        matrix = np.zeros((len(MODELS), len(domains)))
        for i, model in enumerate(MODELS):
            for j, (column, _) in enumerate(domains):
                direct = data[(data.source_label == source) & (data.model == model) &
                              (data["mode"] == "Reasoning Disabled")][column].mean()
                structured = data[(data.source_label == source) & (data.model == model) &
                                  (data["mode"] == "Reasoning Enabled")][column].mean()
                matrix[i, j] = 100 * (structured - direct) / direct
        matrices[source] = matrix
        all_changes.extend(matrix.ravel())
    limit = max(abs(min(all_changes)), abs(max(all_changes)), 1)
    for column_index, source in enumerate(sources):
        ax = fig.add_subplot(grid[2, column_index])
        matrix = matrices[source]
        image_handle = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(source, weight="bold")
        ax.set_xticks(range(len(domains)), [label for _, label in domains], fontsize=7)
        ax.set_yticks(range(len(MODELS)), MODELS, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=8)
        if column_index == 1:
            fig.colorbar(image_handle, ax=ax, fraction=0.046, pad=0.04, label="Relative difference (%)")
    fig.suptitle("Human expert scores by prompt package and source", weight="bold")
    figure6 = save(fig, "Figure6_prompt_package_and_source.png")

    angles = np.linspace(0, 2 * math.pi, len(domains), endpoint=False).tolist()
    angles += angles[:1]
    radar = plt.figure(figsize=(10.5, 8.1), constrained_layout=True)
    for source_index, source in enumerate(sources):
        for model_index, model in enumerate(MODELS):
            ax = radar.add_subplot(2, 3, source_index * 3 + model_index + 1, polar=True)
            for mode, color in [("Reasoning Disabled", "#3D6FB6"), ("Reasoning Enabled", "#E6863B")]:
                subset = data[(data.source_label == source) & (data.model == model) & (data["mode"] == mode)]
                values = [subset[column].mean() for column, _ in domains]
                values += values[:1]
                ax.plot(angles, values, color=color, linewidth=1.8, label=mode_labels[mode])
                ax.fill(angles, values, color=color, alpha=0.12)
            ax.set_xticks(angles[:-1], ["Diagnostic\nAccuracy", "Data\nIntegration", "Clinical\nLogic"], fontsize=7)
            ax.set_ylim(1, 3)
            ax.set_yticks([1, 2, 3], ["1", "2", "3"], fontsize=6)
            ax.set_title(f"{model}\n{source}", fontsize=9, weight="bold")
            if source_index == 0 and model_index == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(-0.38, 1.28), frameon=False, fontsize=8)
    radar.suptitle("Human expert domain scores under the two prompt packages", weight="bold")
    radar_path = save(radar, "Supplementary_Figure_S1_human_radar.png")
    return [figure6, radar_path]


def export_workbook(rows, score_maps):
    WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(rows)
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        base.to_excel(writer, sheet_name="All numeric scores", index=False)
        for judge, input_mode in [("qwen", "multimodal"), ("claude", "multimodal"), ("deepseek_text", "text-only")]:
            records = []
            for row in rows:
                item = score_maps[judge][row["evaluation_id"]]
                score = item["score"]
                records.append(
                    {
                        "evaluation_id": row["evaluation_id"], "case": row["case"], "source": row["source"],
                        "model": row["model"], "mode": row["mode"], "input_mode": input_mode,
                        "diagnostic_accuracy": score["diagnostic_accuracy"],
                        "diagnostic_accuracy_rationale": score.get("diagnostic_accuracy_rationale", ""),
                        "data_integration": score["data_integration"],
                        "data_integration_rationale": score.get("data_integration_rationale", ""),
                        "clinical_logic": score["clinical_logic"],
                        "clinical_logic_rationale": score.get("clinical_logic_rationale", ""),
                        "total_score": score["total_score"],
                    }
                )
            pd.DataFrame(records).to_excel(writer, sheet_name=judge[:31], index=False)
        ranking_rows = []
        for scorer, values in SUMMARY["model_rankings"].items():
            for item in values: ranking_rows.append({"evaluator": scorer, **item})
        pd.DataFrame(ranking_rows).to_excel(writer, sheet_name="Model rankings", index=False)
        effect_rows = []
        for scorer, models in SUMMARY["cot_minus_direct"].items():
            for model, measures in models.items():
                item = measures["total_score"]
                effect_rows.append({
                    "evaluator": scorer, "model": model, "n_cases": item["n"],
                    "mean_delta": item["mean_delta"], "ci_low": item["bootstrap_95_ci"][0],
                    "ci_high": item["bootstrap_95_ci"][1], "wilcoxon_p_unadjusted": item["wilcoxon_p_unadjusted"],
                })
        pd.DataFrame(effect_rows).to_excel(writer, sheet_name="CoT contrasts", index=False)
    return WORKBOOK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "full_analysis_summary.json")
    args = parser.parse_args()
    global SUMMARY
    SUMMARY = json.loads(args.summary.read_text(encoding="utf-8"))
    setup_style()
    rows, scores = load_rows(args.run_config)
    outputs = [workflow_figure(), performance_figure(rows), agreement_figure(), reliability_source_figure(rows)]
    outputs.extend(human_prompt_source_figures(rows))
    outputs.append(export_workbook(rows, scores))
    for path in outputs: print(path)


if __name__ == "__main__":
    main()
