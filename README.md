# OphthoMultiModalReason

Reproduction code for an open-ended multimodal benchmark of clinical reasoning
by large language models in ophthalmology. The repository implements the two
fixed diagnostic prompt packages, blinded same-rubric independent evaluation,
full-cohort statistical analysis, and the public-source sensitivity analysis
reported in the revised manuscript.

## Public-release boundary

This repository contains code, configuration metadata, schemas, tests, and a
fully synthetic example only. It intentionally excludes clinical images,
clinical vignettes, ground-truth diagnoses, individual clinician ratings,
model outputs, evaluator rationales, manuscripts, reviewer files, and API
credentials. Authorized users can place their own equivalent inputs under
`data/` using the schemas in [`docs/data_schema.md`](docs/data_schema.md).

## Evaluated and evaluating models

Diagnostic responses were generated with GPT-5.2, Gemini 2.5 Pro, and Grok 4
under direct-answer and structured-explanation prompt packages. The final
independent-evaluator panel comprised Qwen3.8-Max and Claude Sonnet 5.5 as
multimodal evaluators and DeepSeek-V4-Pro as a text-only sensitivity evaluator.
Exact evaluator roles and decoding settings are recorded in
[`configs/final_evaluator_panel.json`](configs/final_evaluator_panel.json).

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

- `scripts/generate_responses.py`: exact fixed prompt packages and diagnostic
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
- `tests/`: offline unit and smoke tests.
- `examples/`: synthetic, non-clinical fixture.

## Data availability

No clinical or clinician-level data are distributed in this code package.
Availability of de-identified study data remains subject to institutional
approval, data-use agreements, consent, copyright, and privacy restrictions.
Public Chinese educational case images and source text are also not
redistributed; source copyright remains with the original publishers. The
release contains code and synthetic fixtures only, not source media.

## License

A code license must be selected by the authors before the first public push.
See [`LICENSE_PENDING.md`](LICENSE_PENDING.md).

## Citation

Please cite the accompanying manuscript. Machine-readable citation metadata are
provided in [`CITATION.cff`](CITATION.cff); journal and DOI fields should be
updated after publication.
