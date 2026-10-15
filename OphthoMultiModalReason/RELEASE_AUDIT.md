# Public release audit

## Paper analysis coverage matrix

| Paper component | Public entry point | Input boundary | Test/status |
|---|---|---|---|
| Diagnostic response generation | `scripts/generate_responses.py` | Authorized case folders and images | Exact fixed prompts; offline dry-run passed |
| Blinded same-rubric evaluation | `scripts/rescore_same_rubric.py` | Frozen responses, images, and two rater workbooks | Unit tested; paid-run confirmation required |
| Qwen and Claude multimodal evaluation | `scripts/rescore_same_rubric.py run` | Private frozen task package | Configuration documented |
| DeepSeek text-only sensitivity evaluation | `scripts/rescore_same_rubric.py run --input-mode text-only` | Private frozen task package | Image exclusion unit tested |
| Full 74-case analysis | `analysis/analyze_full_results.py` | Private task, human-reference, and evaluator result files | Completeness gate requires 74 cases/444 records |
| Agreement and paired prompt contrasts | `analysis/analysis_common.py`, `analysis/agreement_metrics.py` | In-memory score rows | Public statistical implementation |
| Public-source sensitivity | `analysis/analyze_source_sensitivity.py` | Private score rows; optional redistribution-approved URL list | Explicitly not a membership-inference test |
| Aggregate Figures 1 and 3–6; Supplementary Figure S1 | `analysis/build_figures.py` | Authorized local score artifacts | Public plotting implementation |
| Qualitative Figure 2 | Not distributed | De-identified Ukrainian clinical images and frozen text outputs | Intentionally excluded because clinical source media are outside the public-release boundary |

## Intentionally excluded

- clinical images, case vignettes, and reference diagnoses;
- individual ophthalmologist scores and review packages;
- model responses and evaluator rationales;
- `.env` and all credentials;
- manuscript, supplement, response letter, and reviewer files;
- raw public-source provenance documents and URLs without redistribution approval;
- qualitative Figure 2 source images, which are de-identified Ukrainian clinical images and remain subject to the study's institutional, consent, privacy, and data-use restrictions.

## Verification

- 11 offline unit tests passed.
- Synthetic dry-run passed with one case and six planned model-condition calls.
- File-name and content safety dry-run passed.
- No `.env`, Office/PDF files, clinical data, bytecode, hard-coded credentials,
  private keys, or absolute local paths are included.

## Remaining release decision

The authors must select a software license before public push. Until then, the
release verdict is `release_after_fixes`. After license selection, the package
is `complete_enough_to_release` within the documented private-data boundary.
