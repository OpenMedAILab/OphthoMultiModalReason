# OphthoMultiModalReason

**An open-ended multimodal benchmark of clinical reasoning in ophthalmology.**

How do large language models combine clinical information and ophthalmic images,
and how closely do independent AI evaluators agree with ophthalmologists?
OphthoMultiModalReason compares three diagnostic models under two fixed prompt
packages, using specialist ratings and independent evaluation with the same
three-domain rubric.

| Benchmark | Diagnostic models | Prompt packages | Model responses |
| :---: | :---: | :---: | :---: |
| 74 cases · 348 images | GPT-5.2 · Gemini 2.5 Pro · Grok 4 | Direct answer · Structured explanation | 444 |

[Study overview](#study-overview) · [Results](#results-at-a-glance) ·
[Quick start](#installation) · [Reproduction](#reproduction-workflow) ·
[Data availability](#data-availability) · [Citation](#citation)

## Study overview

![Figure 1. Benchmark workflow: multimodal cases, three diagnostic models, two ophthalmologists, independent same-rubric evaluators, and sensitivity analyses.](assets/figures/Figure1_revised_workflow.png)

*Figure 1 from the revised manuscript. The benchmark includes 51 public Chinese
educational cases and 23 contributed Ukrainian clinical cases. Each case is
assessed by three models under both prompt packages.*

Two ophthalmologists provide the human reference using **Diagnostic Accuracy**,
**Data Integration and Key Information Recognition**, and **Clinical Logic**,
each scored from 1 to 3. Qwen3.8-Max and Claude Sonnet 5.5 independently apply
the same rubric with clinical text and images. DeepSeek-V4-Pro provides a
separate **text-only sensitivity analysis**.

Evaluator roles and decoding settings are recorded in
[`configs/final_evaluator_panel.json`](configs/final_evaluator_panel.json).

## Results at a glance

### Performance across prompt packages

![Figure 3. Mean total scores for three diagnostic models under direct-answer and structured-explanation prompts, assessed by ophthalmologists and three independent AI evaluators.](assets/figures/Figure3_prompt_package_scores.png)

*Figure 3 from the revised manuscript. Mean three-domain total scores are shown
for each model and prompt package; the maximum total score is 9. DeepSeek
received text only.*

GPT-5.2 ranked first, followed by Gemini 2.5 Pro and Grok 4, under human
consensus and all three AI evaluators when scores were pooled across prompt
packages. For GPT-5.2, the structured-explanation package yielded higher total
scores under human consensus and both primary multimodal evaluators. Effects
for the other models depended on the evaluator.

### Agreement with ophthalmologists

![Figure 4. Domain-level agreement with ophthalmologists, correlations between AI evaluators, and pooled model rankings.](assets/figures/Figure4_agreement_and_ranking.png)

*Figure 4 from the revised manuscript. Panels show agreement with human
consensus, agreement between AI evaluators, and mean total-score rankings.*

Qwen and Claude had total-score Spearman correlations of **0.701** and
**0.680** with human consensus. Agreement was stronger for Diagnostic Accuracy
and weaker for Clinical Logic, supporting AI evaluation as a robustness
analysis alongside specialist assessment.

These comparisons concern two fixed prompt packages, which differ in requested
structure and explanation. Human review was not fully masked to model and
prompt condition; the results do not establish an isolated causal effect of
reasoning.

## What this repository provides

- Diagnostic response generation with the two fixed prompt templates.
- Blinded independent evaluation using the same rubric as the ophthalmologists.
- Full-cohort model rankings, agreement metrics, and paired prompt comparisons.
- Source-stratified and public-source sensitivity analyses.
- Figure and analysis-workbook generation, input schemas, and offline tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Do not commit `.env`. Paid API calls require an additional explicit
`--confirm-paid-run` flag.

## Quick smoke test

```bash
python -m unittest discover -s tests -v
python scripts/generate_responses.py \
  --dataset examples/synthetic_dataset \
  --dry-run
```

The smoke test performs no network calls and uses no clinical data.

## Reproduction workflow

### 1. Generate diagnostic responses

```bash
python scripts/generate_responses.py \
  --dataset data/Dataset_74 \
  --output outputs/evaluation_results_full.json \
  --confirm-paid-run
```

The script submits all images for one case in a single OpenRouter request,
uses temperature 0.1, retries failed requests up to five times, and saves after
every successful response.

### 2. Prepare blinded same-rubric evaluation tasks

```bash
python scripts/rescore_same_rubric.py prepare \
  --results outputs/evaluation_results_full.json \
  --dataset data/Dataset_74 \
  --rater1 data/human_scores_rater1.xlsx \
  --rater2 data/human_scores_rater2.xlsx \
  --output data/revision_judge

python scripts/rescore_same_rubric.py validate \
  --output data/revision_judge
```

### 3. Run independent evaluators

Representative commands are shown below. They intentionally require local
credentials and the explicit paid-run confirmation.

```bash
# Qwen3.8-Max, multimodal
python scripts/rescore_same_rubric.py run \
  --output data/revision_judge \
  --provider bailian-token-plan \
  --judge-model qwen3.8-max \
  --run-label qwen_full \
  --thinking-mode disabled \
  --image-detail high \
  --confirm-paid-run

# Claude Sonnet 5.5, multimodal
python scripts/rescore_same_rubric.py run \
  --output data/revision_judge \
  --provider openrouter \
  --judge-model anthropic/claude-sonnet-5.5 \
  --run-label claude_full \
  --reasoning-effort medium \
  --image-detail high \
  --confirm-paid-run

# DeepSeek-V4-Pro, text-only sensitivity analysis
python scripts/rescore_same_rubric.py run \
  --output data/revision_judge \
  --provider bailian-token-plan \
  --judge-model deepseek-v4-pro \
  --run-label deepseek_text_full \
  --input-mode text-only \
  --thinking-mode disabled \
  --confirm-paid-run
```

The published analysis used several resumable run labels. Their mapping is
retained in `configs/published_run_map.json`, so frozen result directories can
be analyzed without merging or rewriting raw result files. A clean single-run
mapping matching the example commands is provided in
`configs/single_run_map.example.json`.

### 4. Full-cohort analysis

```bash
python analysis/analyze_full_results.py \
  --run-config configs/single_run_map.example.json
python analysis/analyze_source_sensitivity.py \
  --run-config configs/single_run_map.example.json
python analysis/build_figures.py \
  --run-config configs/single_run_map.example.json
```

The full analysis requires 74 complete cases and 444 case-model-condition
records for every evaluator. It fails closed if that completeness gate is not
met. The sensitivity script evaluates observable source-associated performance
signals; it is not a test of proprietary training-set membership.

## Repository map

- `scripts/generate_responses.py`: fixed prompt templates and diagnostic
  response generation.
- `scripts/rescore_same_rubric.py`: task freezing, validation, dry-run, and
  independent evaluator execution.
- `analysis/analyze_full_results.py`: final 74-case analysis.
- `analysis/analyze_source_sensitivity.py`: source-stratified sensitivity
  analysis.
- `analysis/build_figures.py`: aggregate workflow, performance, agreement, and
  reliability figures plus a local analysis workbook.
- `analysis/analysis_common.py`, `analysis/agreement_metrics.py`: reusable
  statistics and agreement functions.
- `assets/figures/`: aggregate manuscript Figures 1, 3, and 4 used on this page.
- `tests/`: offline unit and smoke tests.
- `examples/`: synthetic, non-clinical fixture.

## Data availability

This repository distributes code, configuration metadata, schemas, tests, a
synthetic example, and the aggregate manuscript figures displayed above.
Clinical images, clinical vignettes, reference diagnoses, individual clinician
ratings, model responses, evaluator rationales, manuscripts, reviewer files,
and API credentials are excluded.

Authorized users can place their own equivalent inputs under `data/` using
[`docs/data_schema.md`](docs/data_schema.md).

No clinical or clinician-level data are distributed in this code package.
Availability of de-identified study data remains subject to institutional
approval, data-use agreements, consent, copyright, and privacy restrictions.
Public Chinese educational case images and source text are also not
redistributed; source copyright remains with the original publishers. The
release includes aggregate figures, but does not redistribute case source media.

## License

A code license must be selected by the authors before the first public push.
See [`LICENSE_PENDING.md`](LICENSE_PENDING.md).

## Citation

Please cite the accompanying manuscript. Machine-readable citation metadata are
provided in [`CITATION.cff`](CITATION.cff); journal and DOI fields should be
updated after publication.
