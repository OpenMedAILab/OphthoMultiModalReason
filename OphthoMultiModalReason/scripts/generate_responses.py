#!/usr/bin/env python3
"""Generate the frozen model responses used by the ophthalmology benchmark.

The script never sends a request unless ``--confirm-paid-run`` is supplied.
Clinical inputs and images remain local except for the explicit API requests.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path


MODELS = {
    "GPT-5.2": "openai/gpt-5.2",
    "Grok 4": "x-ai/grok-4",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
}
MODES = ["Reasoning Disabled", "Reasoning Enabled"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def split_case(info_path: Path) -> tuple[str, str]:
    text = info_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if "Part B" not in text:
        raise ValueError(f"Missing Part B delimiter: {info_path}")
    part_a, part_b = text.split("Part B", 1)
    part_a = re.sub(r"^\s*Part A\s*", "", part_a, count=1).strip()
    if not part_a or not part_b.strip():
        raise ValueError(f"Empty Part A or Part B: {info_path}")
    return part_a, part_b.strip()


def parse_image_info(path: Path) -> tuple[str, str]:
    name = path.stem.lower()
    modality = next((x for x in ["fundus", "ffa", "oct", "mri", "bscan"] if x in name), "unknown")
    if re.search(r"[_\-.](l|left)(?:[_\-.]|$)", name) or "os" in name:
        laterality = "left"
    elif re.search(r"[_\-.](r|right)(?:[_\-.]|$)", name) or "od" in name:
        laterality = "right"
    else:
        laterality = "unknown"
    return modality, laterality


def prompt_text(clinical_info: str, mode: str) -> str:
    if mode == "Reasoning Enabled":
        return (
            f"【患者临床信息（Part A）】：\n{clinical_info}\n\n"
            "请作为眼科专家，运用思维链（Chain-of-Thought）进行详细推导。\n"
            "**请仔细观察下方【影像清单】及每张图的文件名，严格区分左眼/右眼。**\n"
            "**请严格使用 Markdown 的 ## 二级标题格式输出，结构如下：**\n\n"
            "## 1.【多模态特征提取】\n(结合影像特征和病史描述证据)\n"
            "## 2.【逻辑推理】\n(解释推导过程)\n"
            "## 3.【主诊断】\n(仅给出最重要的1个诊断结论)\n"
            "## 4.【次要诊断】\n(列出并存的其他诊断或并发症，若无则填“无”)\n"
            "## 5.【鉴别诊断】\n(列出2例最容易混淆的备选疾病，并说明理由)\n"
            "## 6.【治疗方案】\n(详细建议)\n"
            "## 7.【风险与随访】"
        )
    return (
        f"【患者临床信息（Part A）】：\n{clinical_info}\n\n"
        "请作为眼科专家，直接给出观察到的特征和最终结论。\n"
        "**请仔细观察下方【影像清单】及每张图的文件名，严格区分左眼/右眼。**\n"
        "**绝对禁止**运用思维链、进行解释、推理、深度思考（Deep Thinking）或给出'因为...所以...'的理由。仅罗列事实。\n"
        "请直接按列表输出：\n"
        "1. 【多模态特征提取】(仅列出观察到的关键特征，不要解释)\n"
        "2. 【主诊断】\n(仅给出最重要的1个诊断结论)\n"
        "3. 【次要诊断】\n"
        "4. 【鉴别诊断】\n(仅列出2例最容易混淆的备选疾病名称)\n"
        "5. 【治疗方案】\n"
        "6. 【风险与随访】"
    )


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def message_content(clinical_info: str, images: list[Path], mode: str) -> list[dict]:
    inventory = []
    for index, path in enumerate(images, 1):
        modality, laterality = parse_image_info(path)
        inventory.append(f"Image {index}: {path.name}; modality={modality}; laterality={laterality}")
    content: list[dict] = [{"type": "text", "text": prompt_text(clinical_info, mode)}]
    if inventory:
        content.append({"type": "text", "text": "Image inventory:\n" + "\n".join(inventory)})
    for path in images:
        content.extend([
            {"type": "text", "text": f"Current image file: {path.name}"},
            {"type": "image_url", "image_url": {"url": data_url(path), "detail": "low"}},
        ])
    return content


def request_with_retry(client, model_id: str, content: list[dict], retries: int) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("Unreachable")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Root containing case folders with info.txt")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_results_full.json"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-paid-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    info_files = sorted(args.dataset.rglob("info.txt"))
    if args.limit is not None:
        info_files = info_files[: args.limit]
    planned = len(info_files) * len(args.models) * len(MODES)
    if args.dry_run:
        print(json.dumps({"cases": len(info_files), "planned_requests": planned, "models": args.models}))
        return
    if not args.confirm_paid_run:
        raise SystemExit("Refusing paid API calls without --confirm-paid-run")

    load_env(args.env)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
    completed = {(x["case"], x["model"], x["mode"]) for x in existing}
    for info_path in info_files:
        case_dir = info_path.parent
        clinical_info, ground_truth = split_case(info_path)
        images = sorted(p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for model_name in args.models:
            for mode in MODES:
                key = (case_dir.name, model_name, mode)
                if key in completed:
                    continue
                diagnosis = request_with_retry(
                    client, MODELS[model_name], message_content(clinical_info, images, mode), args.max_retries
                )
                existing.append({
                    "case": case_dir.name,
                    "disease_type": case_dir.parent.name,
                    "model": model_name,
                    "mode": mode,
                    "diagnosis": diagnosis,
                    "score": None,
                    "ground_truth": ground_truth,
                })
                write_json_atomic(args.output, existing)
                completed.add(key)
    print(json.dumps({"output": str(args.output), "records": len(existing)}))


if __name__ == "__main__":
    main()
