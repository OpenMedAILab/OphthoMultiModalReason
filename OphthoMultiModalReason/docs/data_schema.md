# Private input schemas

These schemas document local inputs expected by the public code. Do not commit
real clinical data to the public repository.

## Case folders

```text
data/Dataset_74/
  <source_or_disease_group>/
    <case_id>/
      info.txt
      <one or more images>
```

Each `info.txt` contains `Part A` (model-visible clinical information) followed
by `Part B` (reference diagnosis). Supported image extensions are JPG, JPEG,
PNG, BMP, TIF, and TIFF. Image filenames may encode modality and laterality.

## Generated response JSON

`generate_responses.py` writes a JSON array with one object per
case-model-condition combination:

```json
{
  "case": "Case_001",
  "disease_type": "synthetic_group",
  "model": "GPT-5.2",
  "mode": "Reasoning Disabled",
  "diagnosis": "model response",
  "score": null,
  "ground_truth": "reference diagnosis"
}
```

## Human-rating workbooks

Both rater workbooks contain one row per case-model-condition record and the
following columns:

- `病例`
- `推理模型`
- `有无推理`
- `诊断准确性（3/2/1）`
- `数据整合与关键信息识别（3/2/1）`
- `临床逻辑性 （3/2/1）`

Scores must be integers from 1 to 3. The preparation command computes the
prespecified two-rater consensus without changing either source workbook.

## Frozen evaluation artifacts

`rescore_same_rubric.py prepare` creates:

- `judge_tasks.jsonl`: blinded evaluator tasks;
- `human_reference.jsonl`: rater scores and consensus;
- `image_manifest.jsonl`: relative paths and hashes;
- `dataset_manifest.json`: counts and source-file hashes;
- `judge_protocol.json`: rubric and scoring protocol;
- selection manifests used for resumable execution.

Individual result files are stored below `results/<run_label>/<model_slug>/`.
They may contain model outputs or evaluator rationales and are therefore not
included in this public package.
