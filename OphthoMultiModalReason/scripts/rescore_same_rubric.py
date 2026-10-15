#!/usr/bin/env python3
"""Prepare and run blinded, same-rubric LLM judging for the TVST revision.

The default commands (prepare, validate, dry-run) are fully offline. The `run`
command requires both a judge model ID and the explicit `--confirm-paid-run`
flag. Original model responses and source data are never modified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "evaluation_results_full.json"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "Dataset_74"
DEFAULT_RATER1 = (
    PROJECT_ROOT
    / "data"
    / "Supplementary"
    / "Supplementary_Data_2_Human_Expert_Scores_Rater_1.xlsx"
)
DEFAULT_RATER2 = (
    PROJECT_ROOT
    / "data"
    / "Supplementary"
    / "Supplementary_Data_3_Human_Expert_Scores_Rater_2.xlsx"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "revision_judge"
DEFAULT_ENV = PROJECT_ROOT / ".env"

EXPECTED_RECORDS = 444
EXPECTED_CASES = 74
EXPECTED_IMAGES = 348
RUBRIC_VERSION = "human_three_domain_supp_table_s2_v1"

CASE_COL = "病例"
MODEL_COL = "推理模型"
MODE_COL = "有无推理"
HUMAN_COLUMNS = {
    "diagnostic_accuracy": "诊断准确性（3/2/1）",
    "data_integration": "数据整合与关键信息识别（3/2/1）",
    "clinical_logic": "临床逻辑性 （3/2/1）",
}

RUBRIC_TEXT = """Three-domain human-expert rubric (use exactly this 1-3 scale):

1. Diagnostic Accuracy
- 3: Main diagnosis correct and secondary diagnoses correct or clinically reasonable.
- 2: Main diagnosis broadly correct but subtype misclassified or complication missed.
- 1: Main diagnosis incorrect.

2. Data Integration & Key Information Recognition
- 3: Multiple key textual and/or imaging features identified and integrated; fully supports the diagnostic conclusion.
- 2: Partial key features; incomplete, predominantly single-modality, or insufficiently supportive.
- 1: Failure to identify or integrate diagnostically relevant information.

3. Clinical Logic
- 3: Overall reasoning coherent and consistent with medical standards; no identifiable safety concern.
- 2: Clinical logic broadly valid but contains overgeneralization, imprecision, or partial mismatch.
- 1: Clinical reasoning flawed or unsafe.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(*parts: str, length: int = 16) -> str:
    return sha256_bytes("\x1f".join(parts).encode("utf-8"))[:length]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing secrets or overriding env."""
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


PROVIDER_ENV = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    "bailian-token-plan": ("BAILIAN_TOKEN_PLAN_API_KEY", "BAILIAN_TOKEN_PLAN_URL"),
    "bailian-token-plan-2": ("BAILIAN_TOKEN_PLAN_API_KEY2", "BAILIAN_TOKEN_PLAN_URL"),
}


def credential_status(env_path: Path, provider: str = "openrouter") -> dict[str, bool]:
    load_env_file(env_path)
    if provider not in PROVIDER_ENV:
        raise ValueError(f"Unknown provider: {provider}")
    key_name, url_name = PROVIDER_ENV[provider]
    return {name: bool(os.environ.get(name, "").strip()) for name in (key_name, url_name)}


def normalize_mode(value: Any) -> str:
    text = str(value).strip()
    mapping = {
        "CoT": "Reasoning Enabled",
        "Direct": "Reasoning Disabled",
        "Reasoning-Enabled": "Reasoning Enabled",
        "Reasoning-Disabled": "Reasoning Disabled",
        "Reasoning Enabled": "Reasoning Enabled",
        "Reasoning Disabled": "Reasoning Disabled",
    }
    if text not in mapping:
        raise ValueError(f"Unknown prompt condition: {text!r}")
    return mapping[text]


def record_key(case: Any, model: Any, mode: Any) -> tuple[str, str, str]:
    return str(case).strip(), str(model).strip(), normalize_mode(mode)


def find_case_folders(dataset_root: Path) -> dict[str, Path]:
    folders: dict[str, Path] = {}
    for info_path in sorted(dataset_root.rglob("info.txt")):
        case = info_path.parent.name
        if case in folders:
            raise ValueError(f"Duplicate case folder basename: {case}")
        folders[case] = info_path.parent
    return folders


def read_part_a(info_path: Path) -> str:
    text = info_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if "Part B" not in text:
        raise ValueError(f"Missing Part B delimiter: {info_path}")
    part_a, _ = text.split("Part B", 1)
    part_a = re.sub(r"^\s*Part A\s*", "", part_a, count=1).strip()
    if not part_a:
        raise ValueError(f"Empty Part A: {info_path}")
    if re.search(r"【(?:主诊断|Main Diagnosis)】", part_a, flags=re.IGNORECASE):
        raise ValueError(f"Ground-truth label leaked into Part A: {info_path}")
    return part_a


def image_paths(case_folder: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in case_folder.iterdir() if p.is_file() and p.suffix.lower() in allowed)


def load_rater(path: Path, rater: str) -> dict[tuple[str, str, str], dict[str, int]]:
    frame = pd.read_excel(path)
    expected = {CASE_COL, MODEL_COL, MODE_COL, *HUMAN_COLUMNS.values()}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{rater} missing columns: {sorted(missing)}")
    scores: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in frame.to_dict(orient="records"):
        key = record_key(row[CASE_COL], row[MODEL_COL], row[MODE_COL])
        if key in scores:
            raise ValueError(f"Duplicate {rater} key: {key}")
        item: dict[str, int] = {}
        for domain, column in HUMAN_COLUMNS.items():
            value = row[column]
            if pd.isna(value) or int(value) != value or int(value) not in {1, 2, 3}:
                raise ValueError(f"Invalid {rater} score for {key}, {domain}: {value!r}")
            item[domain] = int(value)
        scores[key] = item
    return scores


def source_label(disease_type: str) -> str:
    if disease_type.startswith("CN_"):
        return "Chinese"
    if disease_type.startswith("INT_"):
        return "Ukrainian"
    return "Unknown"


def select_quantile_cases(
    case_rows: list[dict[str, Any]], count: int, used_diseases: set[str]
) -> list[dict[str, Any]]:
    ordered = sorted(case_rows, key=lambda x: (x["image_count"], x["case"]))
    if count <= 0:
        return []
    anchors = [round(i * (len(ordered) - 1) / max(count - 1, 1)) for i in range(count)]
    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()
    for anchor in anchors:
        candidates = sorted(
            enumerate(ordered),
            key=lambda pair: (
                pair[1]["disease_type"] in used_diseases,
                abs(pair[0] - anchor),
                pair[1]["case"],
            ),
        )
        choice = next(row for _, row in candidates if row["case"] not in used_cases)
        selected.append(choice)
        used_cases.add(choice["case"])
        used_diseases.add(choice["disease_type"])
    return selected


def prepare(args: argparse.Namespace) -> None:
    results_path = args.results.resolve()
    dataset_root = args.dataset.resolve()
    output = args.output.resolve()
    records = read_json(results_path)
    if not isinstance(records, list):
        raise ValueError("Main results JSON must contain a list")

    folders = find_case_folders(dataset_root)
    rater1 = load_rater(args.rater1.resolve(), "Rater 1")
    rater2 = load_rater(args.rater2.resolve(), "Rater 2")

    image_manifest: list[dict[str, Any]] = []
    case_data: dict[str, dict[str, Any]] = {}
    for case, folder in sorted(folders.items()):
        images = image_paths(folder)
        if not images:
            raise ValueError(f"No images for case: {case}")
        image_items = []
        for image in images:
            relative = image.relative_to(PROJECT_ROOT).as_posix()
            item = {
                "case": case,
                "path": relative,
                "bytes": image.stat().st_size,
                "sha256": sha256_file(image),
            }
            image_manifest.append(item)
            image_items.append(item)
        case_data[case] = {
            "clinical_info": read_part_a(folder / "info.txt"),
            "images": image_items,
        }

    tasks: list[dict[str, Any]] = []
    human_reference: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = record_key(record.get("case"), record.get("model"), record.get("mode"))
        if key in seen:
            raise ValueError(f"Duplicate model-output key: {key}")
        seen.add(key)
        case, model, mode = key
        if case not in case_data:
            raise ValueError(f"No case folder for model output: {case}")
        if key not in rater1 or key not in rater2:
            raise ValueError(f"Human rating missing for: {key}")
        diagnosis = str(record.get("diagnosis") or "").strip()
        ground_truth = str(record.get("ground_truth") or "").strip()
        if not ground_truth:
            raise ValueError(f"Ground truth missing for: {key}")
        blind_id = "E" + stable_id(case, model, mode)
        task_core = {
            "evaluation_id": blind_id,
            "case": case,
            "disease_type": str(record.get("disease_type") or "").strip(),
            "source": source_label(str(record.get("disease_type") or "").strip()),
            "model": model,
            "mode": mode,
            "clinical_info": case_data[case]["clinical_info"],
            "ground_truth": ground_truth,
            "candidate_response": diagnosis,
            "is_empty_response": not bool(diagnosis),
            "images": case_data[case]["images"],
            "rubric_version": RUBRIC_VERSION,
        }
        task_core["task_sha256"] = sha256_bytes(canonical_json(task_core).encode("utf-8"))
        tasks.append(task_core)

        s1, s2 = rater1[key], rater2[key]
        consensus = {domain: (s1[domain] + s2[domain]) / 2 for domain in HUMAN_COLUMNS}
        human_reference.append(
            {
                "evaluation_id": blind_id,
                "case": case,
                "model": model,
                "mode": mode,
                "rater1": s1,
                "rater2": s2,
                "consensus": consensus,
                "consensus_total": sum(consensus.values()),
            }
        )

    tasks.sort(key=lambda x: (x["case"], x["model"], x["mode"]))
    human_reference.sort(key=lambda x: x["evaluation_id"])

    case_rows = []
    for case in sorted(folders):
        sample = next(task for task in tasks if task["case"] == case)
        case_rows.append(
            {
                "case": case,
                "source": sample["source"],
                "disease_type": sample["disease_type"],
                "image_count": len(case_data[case]["images"]),
            }
        )
    used_diseases: set[str] = set()
    pilot_cases = select_quantile_cases(
        [x for x in case_rows if x["source"] == "Chinese"], 4, used_diseases
    ) + select_quantile_cases(
        [x for x in case_rows if x["source"] == "Ukrainian"], 2, used_diseases
    )
    pilot_case_names = {x["case"] for x in pilot_cases}
    pilot_ids = [x["evaluation_id"] for x in tasks if x["case"] in pilot_case_names]
    pilot_id_set = set(pilot_ids)
    remaining_ids = [x["evaluation_id"] for x in tasks if x["evaluation_id"] not in pilot_id_set]

    # Quarter-stage analysis uses complete case clusters. Nineteen of 74 cases
    # (25.7%) preserves all six model-condition pairs per selected case. The
    # selection includes the pilot cases, covers all disease groups, and matches
    # the overall source ratio as closely as possible (13 Chinese / 6 Ukrainian).
    quarter_cases_by_name = {x["case"]: x for x in pilot_cases}
    represented_diseases = {x["disease_type"] for x in pilot_cases}
    for disease in sorted({x["disease_type"] for x in case_rows} - represented_diseases):
        candidates = sorted(
            [x for x in case_rows if x["disease_type"] == disease],
            key=lambda x: (x["image_count"], x["case"]),
        )
        choice = candidates[len(candidates) // 2]
        quarter_cases_by_name[choice["case"]] = choice
        represented_diseases.add(disease)
    source_targets = {"Chinese": 13, "Ukrainian": 6}
    for source, target in source_targets.items():
        current = sum(x["source"] == source for x in quarter_cases_by_name.values())
        need = target - current
        if need < 0:
            raise ValueError(f"Disease coverage exceeds quarter source target for {source}")
        candidates = [
            x
            for x in case_rows
            if x["source"] == source and x["case"] not in quarter_cases_by_name
        ]
        for choice in select_quantile_cases(candidates, need, represented_diseases):
            quarter_cases_by_name[choice["case"]] = choice
    quarter_cases = sorted(quarter_cases_by_name.values(), key=lambda x: (x["source"], x["case"]))
    if len(quarter_cases) != 19:
        raise ValueError(f"Expected 19 quarter-stage cases, found {len(quarter_cases)}")
    quarter_case_names = set(quarter_cases_by_name)
    quarter_ids = [x["evaluation_id"] for x in tasks if x["case"] in quarter_case_names]
    quarter_additional_ids = [x for x in quarter_ids if x not in pilot_id_set]
    remaining_after_quarter_ids = [
        x["evaluation_id"] for x in tasks if x["case"] not in quarter_case_names
    ]

    write_jsonl(output / "judge_tasks.jsonl", tasks)
    write_jsonl(output / "human_reference.jsonl", human_reference)
    write_jsonl(output / "image_manifest.jsonl", image_manifest)
    write_json(
        output / "pilot_selection.json",
        {
            "selection_version": "six_case_source_image_burden_v1",
            "cases": pilot_cases,
            "evaluation_ids": pilot_ids,
            "expected_calls_per_judge": len(pilot_ids)
            - sum(x["is_empty_response"] for x in tasks if x["evaluation_id"] in pilot_ids),
        },
    )
    write_json(
        output / "remaining_after_pilot_selection.json",
        {
            "selection_version": "all_tasks_excluding_pilot_v1",
            "evaluation_ids": remaining_ids,
            "tasks": len(remaining_ids),
            "empty_response_tasks": sum(
                x["is_empty_response"] for x in tasks if x["evaluation_id"] in set(remaining_ids)
            ),
            "expected_paid_calls_per_judge": len(remaining_ids)
            - sum(x["is_empty_response"] for x in tasks if x["evaluation_id"] in set(remaining_ids)),
        },
    )
    write_json(
        output / "quarter_selection.json",
        {
            "selection_version": "nineteen_complete_cases_stratified_v1",
            "cases": quarter_cases,
            "evaluation_ids": quarter_ids,
            "tasks": len(quarter_ids),
            "source_case_counts": dict(Counter(x["source"] for x in quarter_cases)),
            "disease_groups": len({x["disease_type"] for x in quarter_cases}),
        },
    )
    write_json(
        output / "quarter_additional_after_pilot_selection.json",
        {
            "selection_version": "quarter_stage_excluding_completed_pilot_v1",
            "evaluation_ids": quarter_additional_ids,
            "tasks": len(quarter_additional_ids),
            "empty_response_tasks": sum(
                x["is_empty_response"] for x in tasks if x["evaluation_id"] in set(quarter_additional_ids)
            ),
            "expected_paid_calls_per_judge": len(quarter_additional_ids)
            - sum(x["is_empty_response"] for x in tasks if x["evaluation_id"] in set(quarter_additional_ids)),
        },
    )
    write_json(
        output / "remaining_after_quarter_selection.json",
        {
            "selection_version": "all_complete_cases_excluding_quarter_stage_v1",
            "evaluation_ids": remaining_after_quarter_ids,
            "tasks": len(remaining_after_quarter_ids),
            "cases": len({
                x["case"] for x in tasks
                if x["evaluation_id"] in set(remaining_after_quarter_ids)
            }),
            "empty_response_tasks": sum(
                x["is_empty_response"]
                for x in tasks
                if x["evaluation_id"] in set(remaining_after_quarter_ids)
            ),
            "expected_paid_calls_per_judge": len(remaining_after_quarter_ids)
            - sum(
                x["is_empty_response"]
                for x in tasks
                if x["evaluation_id"] in set(remaining_after_quarter_ids)
            ),
        },
    )
    write_json(
        output / "dataset_manifest.json",
        {
            "manifest_version": 1,
            "rubric_version": RUBRIC_VERSION,
            "source_files": {
                str(results_path.relative_to(PROJECT_ROOT)): sha256_file(results_path),
                str(args.rater1.resolve().relative_to(PROJECT_ROOT)): sha256_file(args.rater1.resolve()),
                str(args.rater2.resolve().relative_to(PROJECT_ROOT)): sha256_file(args.rater2.resolve()),
            },
            "counts": {
                "tasks": len(tasks),
                "cases": len({x["case"] for x in tasks}),
                "images": len(image_manifest),
                "empty_responses": sum(x["is_empty_response"] for x in tasks),
                "pilot_cases": len(pilot_cases),
                "pilot_tasks": len(pilot_ids),
                "remaining_tasks_after_pilot": len(remaining_ids),
                "quarter_stage_cases": len(quarter_cases),
                "quarter_stage_tasks": len(quarter_ids),
                "quarter_stage_additional_tasks": len(quarter_additional_ids),
                "remaining_cases_after_quarter": len(folders) - len(quarter_cases),
                "remaining_tasks_after_quarter": len(remaining_after_quarter_ids),
            },
            "rubric_sha256": sha256_bytes(RUBRIC_TEXT.encode("utf-8")),
        },
    )
    print(f"Prepared {len(tasks)} tasks in {output}")
    print(f"Cases: {len(folders)}; images: {len(image_manifest)}; pilot tasks: {len(pilot_ids)}")
    print("No API requests were made.")


def validate_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    domains = ["diagnostic_accuracy", "data_integration", "clinical_logic"]
    normalized: dict[str, Any] = {}
    for domain in domains:
        value = payload.get(domain)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ValueError(f"{domain} must be an integer")
        value = int(value)
        if value not in {1, 2, 3}:
            raise ValueError(f"{domain} out of range: {value}")
        normalized[domain] = value
    normalized["total_score"] = sum(normalized[x] for x in domains)
    for field in [
        "diagnostic_accuracy_rationale",
        "data_integration_rationale",
        "clinical_logic_rationale",
    ]:
        normalized[field] = str(payload.get(field, "")).strip()
    return normalized


def parse_score_response(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    if cleaned != candidates[0]:
        candidates.append(cleaned)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    errors = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("response JSON is not an object")
            return validate_score_payload(parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError("Could not parse valid score JSON: " + " | ".join(errors[:3]))


def build_prompt(task: dict[str, Any], images_available: bool = True) -> str:
    evidence_note = (
        "Use the clinical information, reference diagnosis, attached images, and candidate response as evidence."
        if images_available
        else "This is a prespecified text-only sensitivity analysis: images are not supplied. Use only the clinical information, reference diagnosis, and candidate response; do not claim to have independently verified image findings."
    )
    return f"""Evaluate one de-identified ophthalmology model response. Apply the rubric literally and independently. Do not infer or reward the identity of the model or its prompting condition. {evidence_note}

{RUBRIC_TEXT}

Clinical information (Part A):
{task['clinical_info']}

Reference diagnosis (Part B):
{task['ground_truth']}

Candidate response:
{task['candidate_response']}

Return JSON only, with exactly these fields:
{{
  "diagnostic_accuracy": 1,
  "diagnostic_accuracy_rationale": "brief evidence-based rationale",
  "data_integration": 1,
  "data_integration_rationale": "brief evidence-based rationale",
  "clinical_logic": 1,
  "clinical_logic_rationale": "brief evidence-based rationale",
  "total_score": 3
}}
All three scores must be integers from 1 to 3. The total must equal their sum."""


def validate_artifacts(output: Path, verify_hashes: bool = False) -> dict[str, Any]:
    tasks = read_jsonl(output / "judge_tasks.jsonl")
    humans = read_jsonl(output / "human_reference.jsonl")
    images = read_jsonl(output / "image_manifest.jsonl")
    manifest = read_json(output / "dataset_manifest.json")
    pilot = read_json(output / "pilot_selection.json")

    ids = [x["evaluation_id"] for x in tasks]
    if len(tasks) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} unique tasks, found {len(tasks)}/{len(set(ids))}")
    human_ids = {x["evaluation_id"] for x in humans}
    if human_ids != set(ids):
        raise ValueError("Human reference IDs do not exactly match task IDs")
    cases = Counter(x["case"] for x in tasks)
    if len(cases) != EXPECTED_CASES or set(cases.values()) != {6}:
        raise ValueError(f"Expected {EXPECTED_CASES} cases with six tasks each")
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} images, found {len(images)}")
    for item in images:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Image size changed: {path}")
        if verify_hashes and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Image hash changed: {path}")
    for task in tasks:
        if task["rubric_version"] != RUBRIC_VERSION:
            raise ValueError(f"Rubric mismatch: {task['evaluation_id']}")
        if not task["is_empty_response"]:
            build_prompt(task)
    pilot_ids = pilot.get("evaluation_ids", [])
    if len(pilot.get("cases", [])) != 6 or len(pilot_ids) != 36 or not set(pilot_ids) <= set(ids):
        raise ValueError("Pilot must contain six complete cases / 36 tasks")
    if manifest["counts"]["tasks"] != len(tasks):
        raise ValueError("Dataset manifest count mismatch")
    return {
        "tasks": len(tasks),
        "cases": len(cases),
        "images": len(images),
        "empty_responses": sum(x["is_empty_response"] for x in tasks),
        "pilot_tasks": len(pilot_ids),
    }


def validate_command(args: argparse.Namespace) -> None:
    summary = validate_artifacts(args.output.resolve(), verify_hashes=args.verify_image_hashes)
    print("Artifact validation passed:", canonical_json(summary))
    if args.check_env:
        status = credential_status(args.env.resolve(), args.provider)
        if not all(status.values()):
            missing = [key for key, present in status.items() if not present]
            raise ValueError(f"Missing required .env entries: {missing}")
        print("Credential variables are present; values were not displayed.")
    print("No API requests were made.")


def load_selection(output: Path, selection_path: Path | None) -> set[str] | None:
    if selection_path is None:
        return None
    selection = read_json(selection_path.resolve())
    ids = selection.get("evaluation_ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError("Selection JSON must contain an evaluation_ids list")
    return set(ids)


def chosen_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = args.output.resolve()
    tasks = read_jsonl(output / "judge_tasks.jsonl")
    selected = load_selection(output, args.selection)
    if selected is not None:
        tasks = [task for task in tasks if task["evaluation_id"] in selected]
        if len(tasks) != len(selected):
            raise ValueError("Selection includes unknown or duplicate evaluation IDs")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def dry_run(args: argparse.Namespace) -> None:
    summary = validate_artifacts(args.output.resolve(), verify_hashes=False)
    status = credential_status(args.env.resolve(), args.provider)
    if not all(status.values()):
        missing = [key for key, present in status.items() if not present]
        raise ValueError(f"Missing required .env entries: {missing}")
    tasks = chosen_tasks(args)
    total_image_bytes = 0
    api_eligible = 0
    for task in tasks:
        if task["is_empty_response"]:
            continue
        api_eligible += 1
        prompt = build_prompt(task, images_available=args.input_mode == "multimodal")
        if not prompt.strip():
            raise ValueError(f"Empty prompt: {task['evaluation_id']}")
        if args.input_mode == "multimodal":
            for image in task["images"]:
                total_image_bytes += (PROJECT_ROOT / image["path"]).stat().st_size
    print(
        "Dry-run passed:",
        canonical_json(
            {
                **summary,
                "selected_tasks": len(tasks),
                "api_eligible_tasks": api_eligible,
                "input_mode": args.input_mode,
                "repeated_image_bytes_before_base64": total_image_bytes,
                "env_variables_present": True,
            }
        ),
    )
    print("No API requests were made.")


def image_data_url(path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image type: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_messages(
    task: dict[str, Any], image_detail: str, input_mode: str = "multimodal"
) -> list[dict[str, Any]]:
    images_available = input_mode == "multimodal"
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_prompt(task, images_available=images_available)}
    ]
    if images_available:
        for index, image in enumerate(task["images"], start=1):
            path = PROJECT_ROOT / image["path"]
            content.append({"type": "text", "text": f"Attached image {index}: {path.name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(path), "detail": image_detail},
                }
            )
    return [
        {
            "role": "system",
            "content": "You are an expert ophthalmology evaluator. Return valid JSON only.",
        },
        {"role": "user", "content": content},
    ]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        raise ValueError("Judge model ID produced an empty output slug")
    return slug


def atomic_result(path: Path, value: dict[str, Any]) -> None:
    write_json(path, value)


def run_command(args: argparse.Namespace) -> None:
    if not args.confirm_paid_run:
        raise SystemExit("Refusing API calls without --confirm-paid-run")
    if not args.judge_model.strip():
        raise ValueError("--judge-model is required")
    validate_artifacts(args.output.resolve(), verify_hashes=False)
    status = credential_status(args.env.resolve(), args.provider)
    if not all(status.values()):
        missing = [key for key, present in status.items() if not present]
        raise ValueError(f"Missing required .env entries: {missing}")

    from openai import OpenAI

    key_name, url_name = PROVIDER_ENV[args.provider]
    client_kwargs = {
        "api_key": os.environ[key_name],
        "base_url": os.environ[url_name],
    }
    if args.provider == "openrouter":
        client_kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/zi-ran",
            "X-Title": "TVST Same-Rubric Judge Validation",
        }
    client = OpenAI(**client_kwargs)
    tasks = chosen_tasks(args)
    judge_dir = (
        args.output.resolve()
        / "results"
        / safe_slug(args.run_label)
        / safe_slug(args.judge_model)
    )
    judge_dir.mkdir(parents=True, exist_ok=True)
    selection_sha256 = None
    if args.selection is not None:
        selection_sha256 = sha256_file(args.selection.resolve())
    run_parameters = {
        "provider": args.provider,
        "judge_model": args.judge_model,
        "input_mode": args.input_mode,
        "thinking_mode": args.thinking_mode,
        "temperature": 0,
        "image_detail": args.image_detail,
        "max_completion_tokens": args.max_completion_tokens,
        "reasoning_effort": args.reasoning_effort,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": sha256_bytes(RUBRIC_TEXT.encode("utf-8")),
        "task_manifest_sha256": sha256_file(args.output.resolve() / "judge_tasks.jsonl"),
        "selection_sha256": selection_sha256,
        "selected_task_count": len(tasks),
        "run_label": args.run_label,
    }
    config_path = judge_dir / "run_config.json"
    if config_path.exists():
        previous = read_json(config_path)
        previous_parameters = previous.get("parameters")
        if previous_parameters != run_parameters:
            raise ValueError(
                "Existing run_config.json does not match requested parameters; "
                "use a different judge model/output directory or restore the frozen configuration"
            )
    else:
        write_json(
            config_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": run_parameters,
            },
        )
    completed = skipped = skipped_errors = failed = 0
    for position, task in enumerate(tasks, start=1):
        result_path = judge_dir / f"{task['evaluation_id']}.json"
        existing_error_path = judge_dir / f"{task['evaluation_id']}.error.json"
        if result_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if existing_error_path.exists() and args.skip_existing_errors and not args.overwrite:
            skipped_errors += 1
            print(
                f"[{position}/{len(tasks)}] {task['evaluation_id']}: retained prior error",
                file=sys.stderr,
            )
            continue
        if task["is_empty_response"]:
            score = {
                "diagnostic_accuracy": 1,
                "data_integration": 1,
                "clinical_logic": 1,
                "total_score": 3,
                "diagnostic_accuracy_rationale": "Prespecified minimum score for an empty response.",
                "data_integration_rationale": "Prespecified minimum score for an empty response.",
                "clinical_logic_rationale": "Prespecified minimum score for an empty response.",
            }
            atomic_result(
                result_path,
                {
                    "evaluation_id": task["evaluation_id"],
                    "task_sha256": task["task_sha256"],
                    "judge_model": args.judge_model,
                    "rubric_version": RUBRIC_VERSION,
                    "score": score,
                    "raw_response": None,
                    "attempts": 0,
                    "run_parameters": run_parameters,
                    "prespecified_empty_response_rule": True,
                },
            )
            completed += 1
            continue

        messages = build_messages(task, args.image_detail, args.input_mode)
        last_error = ""
        last_raw_response = None
        for attempt in range(1, args.max_retries + 1):
            try:
                extra_body = None
                if args.provider == "openrouter" and args.reasoning_effort != "none":
                    extra_body = {"reasoning": {"effort": args.reasoning_effort}}
                elif args.provider.startswith("bailian-token-plan") and args.thinking_mode != "default":
                    extra_body = {"enable_thinking": args.thinking_mode == "enabled"}
                response = client.chat.completions.create(
                    model=args.judge_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_completion_tokens,
                    timeout=args.timeout,
                    extra_body=extra_body,
                )
                raw = response.choices[0].message.content or ""
                last_raw_response = raw
                score = parse_score_response(raw)
                usage = None
                if getattr(response, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
                atomic_result(
                    result_path,
                    {
                        "evaluation_id": task["evaluation_id"],
                        "task_sha256": task["task_sha256"],
                        "judge_model": args.judge_model,
                        "rubric_version": RUBRIC_VERSION,
                        "score": score,
                        "raw_response": raw,
                        "response_id": getattr(response, "id", None),
                        "response_model": getattr(response, "model", None),
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "attempts": attempt,
                        "usage": usage,
                        "run_parameters": run_parameters,
                        "prespecified_empty_response_rule": False,
                    },
                )
                stale_error_path = judge_dir / f"{task['evaluation_id']}.error.json"
                if stale_error_path.exists():
                    stale_error_path.unlink()
                completed += 1
                print(f"[{position}/{len(tasks)}] {task['evaluation_id']}: saved")
                break
            except Exception as exc:  # API errors and schema failures are both retried.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    delay = min(args.retry_base_seconds * (2 ** (attempt - 1)), 60)
                    time.sleep(delay + random.random())
        else:
            failed += 1
            error_path = judge_dir / f"{task['evaluation_id']}.error.json"
            atomic_result(
                error_path,
                {
                    "evaluation_id": task["evaluation_id"],
                    "task_sha256": task["task_sha256"],
                    "judge_model": args.judge_model,
                    "error": last_error,
                    "raw_response": last_raw_response,
                    "attempts": args.max_retries,
                },
            )
            print(f"[{position}/{len(tasks)}] {task['evaluation_id']}: failed", file=sys.stderr)
            if failed >= args.max_failures:
                print(
                    f"Stopping after {failed} failed task(s); max_failures={args.max_failures}",
                    file=sys.stderr,
                )
                break
        if args.inter_call_seconds > 0:
            time.sleep(args.inter_call_seconds)
    print(
        canonical_json(
            {
                "completed": completed,
                "skipped": skipped,
                "skipped_errors": skipped_errors,
                "failed": failed,
            }
        )
    )
    if failed:
        raise SystemExit(2)


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def add_provider(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_ENV),
        default="openrouter",
        help="API-compatible inference provider and matching .env credential pair",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Build frozen tasks and manifests (offline)")
    p_prepare.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p_prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p_prepare.add_argument("--rater1", type=Path, default=DEFAULT_RATER1)
    p_prepare.add_argument("--rater2", type=Path, default=DEFAULT_RATER2)
    add_common_paths(p_prepare)
    p_prepare.set_defaults(func=prepare)

    p_validate = sub.add_parser("validate", help="Validate prepared artifacts (offline)")
    add_common_paths(p_validate)
    add_provider(p_validate)
    p_validate.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_validate.add_argument("--check-env", action="store_true")
    p_validate.add_argument("--verify-image-hashes", action="store_true")
    p_validate.set_defaults(func=validate_command)

    p_dry = sub.add_parser("dry-run", help="Construct and validate selected calls (offline)")
    add_common_paths(p_dry)
    add_provider(p_dry)
    p_dry.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_dry.add_argument("--selection", type=Path)
    p_dry.add_argument("--limit", type=int)
    p_dry.add_argument("--input-mode", choices=["multimodal", "text-only"], default="multimodal")
    p_dry.set_defaults(func=dry_run)

    p_run = sub.add_parser("run", help="Execute paid OpenRouter judge calls")
    add_common_paths(p_run)
    add_provider(p_run)
    p_run.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_run.add_argument("--selection", type=Path)
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--judge-model", required=True)
    p_run.add_argument("--run-label", default="main")
    p_run.add_argument("--input-mode", choices=["multimodal", "text-only"], default="multimodal")
    p_run.add_argument(
        "--thinking-mode",
        choices=["default", "disabled", "enabled"],
        default="default",
        help="Bailian thinking toggle; recorded and ignored by OpenRouter",
    )
    p_run.add_argument("--image-detail", choices=["low", "auto", "high"], default="high")
    p_run.add_argument("--max-completion-tokens", type=int, default=800)
    p_run.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max", "xhigh"],
        default="none",
        help="OpenRouter reasoning effort; use 'none' for non-reasoning models",
    )
    p_run.add_argument("--max-retries", type=int, default=5)
    p_run.add_argument("--max-failures", type=int, default=1)
    p_run.add_argument("--retry-base-seconds", type=float, default=3.0)
    p_run.add_argument("--inter-call-seconds", type=float, default=0.5)
    p_run.add_argument("--timeout", type=float, default=180.0)
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument(
        "--skip-existing-errors",
        action="store_true",
        help="Retain audited error files as missing results and continue remaining tasks",
    )
    p_run.add_argument("--confirm-paid-run", action="store_true")
    p_run.set_defaults(func=run_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
