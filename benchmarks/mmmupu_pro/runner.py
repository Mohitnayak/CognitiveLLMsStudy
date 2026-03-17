"""
MMMU-Pro runner: load dataset from HuggingFace (MMMU/MMMU_Pro), run Ollama VLM
on each sample, extract answer (A/B/C/.../J), and compute accuracy.
Supports configs: standard (4 options), standard (10 options), vision.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

# Configs on Hugging Face: "standard (4 options)", "standard (10 options)", "vision"
MMMU_PRO_CONFIGS = ("standard (4 options)", "standard (10 options)", "vision")

MMMU_PRO_POST_PROMPT = (
    "\nAnswer with the option's letter only (e.g. A, B, C). "
    "Enclose the letter within ``."
)

VISION_PROMPT = (
    "The image contains a multiple-choice question. "
    "Answer with only the option letter (A, B, C, etc.) enclosed in ``."
)


def _letter_from_option_index(index: int) -> str:
    """Map 0->A, 1->B, ... 9->J."""
    if index < 0 or index > 9:
        return ""
    return chr(ord("A") + index)


def _normalize_gt_answer(answer: str, options: list | None, max_options: int = 10) -> str:
    """
    Normalize ground truth to a single letter A–J.
    If answer is already a single letter, return it. If it's option text, find index in options.
    """
    if not answer:
        return ""
    s = answer.strip().upper()
    # Single letter A–J
    if len(s) == 1 and s in "ABCDEFGHIJ":
        return s
    if len(s) > 1 and s[0] in "ABCDEFGHIJ":
        return s[0]
    # Match full option text to get letter
    if options and isinstance(options, list):
        for i, opt in enumerate(options):
            if i >= max_options:
                break
            if opt and str(opt).strip().upper() == s:
                return _letter_from_option_index(i)
            if opt and str(opt).strip().upper().startswith(s[:20]):
                return _letter_from_option_index(i)
    return s[0] if s else ""


def extract_single_choice_mmmupu(pred: str, gt_letter: str, max_option: str = "J") -> float:
    """
    Extract single-choice letter (A through max_option, default J) from model prediction.
    Returns 1.0 if prediction matches ground truth letter, 0.0 otherwise.
    """
    if not pred or gt_letter is None:
        return 0.0
    pred = pred.strip()
    # Try ``...`` first
    m = re.search(r"``([^`]*)``", pred)
    if m:
        pred = m.group(1).strip()
    else:
        m = re.search(r"`([^`]*)`", pred)
        if m:
            pred = m.group(1).strip()
    m = re.search(r"\{([^}]*)\}", pred)
    if m:
        pred = m.group(1).strip()
    # Single letter A–J with word boundary
    letters = "ABCDEFGHIJ"[: ord(max_option.upper()) - ord("A") + 1]
    pattern = r"\b[" + letters + r"]\b(?!\s[a-zA-Z])"
    m = re.search(pattern, pred, re.IGNORECASE)
    if not m:
        return 0.0
    pred_letter = m.group().upper()
    return 1.0 if pred_letter == gt_letter.upper() else 0.0


def _extract_pred_letter_mmmupu(pred: str, max_option: str = "J") -> str:
    """Extract single-choice letter (A–max_option) from model prediction; returns "" if none found."""
    if not pred:
        return ""
    pred = pred.strip()
    m = re.search(r"``([^`]*)``", pred)
    if m:
        pred = m.group(1).strip()
    else:
        m = re.search(r"`([^`]*)`", pred)
        if m:
            pred = m.group(1).strip()
    m = re.search(r"\{([^}]*)\}", pred)
    if m:
        pred = m.group(1).strip()
    letters = "ABCDEFGHIJ"[: ord(max_option.upper()) - ord("A") + 1]
    pattern = r"\b[" + letters + r"]\b(?!\s[a-zA-Z])"
    m = re.search(pattern, pred, re.IGNORECASE)
    if not m:
        return ""
    return m.group().upper()


def _get_image_paths_from_row(row: dict, tmpdir: Path, sample_id: str) -> list[str]:
    """Collect image paths from row (image_1, image_2, ... or image for vision)."""
    paths: list[str] = []
    # Vision: single "image" key
    if "image" in row and row["image"] is not None:
        img = row["image"]
        p = tmpdir / f"{sample_id}_0.png"
        _save_image(img, p)
        paths.append(str(p))
        return paths
    # Standard: image_1, image_2, ... (or "images" as list)
    if "images" in row and row["images"] is not None and isinstance(row["images"], list):
        for n, img in enumerate(row["images"]):
            if img is None:
                continue
            p = tmpdir / f"{sample_id}_{n}.png"
            try:
                _save_image(img, p)
                paths.append(str(p))
            except Exception:
                continue
        if paths:
            return paths
    for key in sorted(row.keys()):
        if not key.startswith("image_") or row[key] is None:
            continue
        try:
            num = key.replace("image_", "")
            p = tmpdir / f"{sample_id}_{num}.png"
            _save_image(row[key], p)
            paths.append(str(p))
        except Exception:
            continue
    return paths


def _save_image(img: Any, path: Path) -> None:
    if hasattr(img, "save"):
        img.save(path)
    elif isinstance(img, bytes):
        path.write_bytes(img)
    elif isinstance(img, dict) and "bytes" in img:
        path.write_bytes(img["bytes"])
    else:
        path.write_bytes(img if isinstance(img, (bytes, bytearray)) else b"")


def run_mmmupu_pro(
    model: str,
    *,
    config: str = "standard (10 options)",
    limit: int | None = None,
    results_dir: Path | None = None,
    output_name: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Load MMMU-Pro from HuggingFace, run Ollama model on each sample, compute accuracy.

    Args:
        model: Ollama model name (e.g. llava, qwen2.5-vl).
        config: Dataset config: "standard (4 options)", "standard (10 options)", or "vision".
        limit: If set, run only the first N samples.
        results_dir: Where to save result JSON/CSV; uses repo results/ if None.
        output_name: Base name for result files (default: mmmupu_pro_<config_slug>_<model>).

    Returns:
        (list of per-sample result dicts, metrics dict with total, correct, exact_match).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "MMMU-Pro requires the 'datasets' package. Install with: pip install datasets"
        ) from None

    from src.ollama_client import chat

    if config not in MMMU_PRO_CONFIGS:
        raise ValueError(
            f"config must be one of {MMMU_PRO_CONFIGS!r}, got {config!r}"
        )

    repo_root = Path(__file__).resolve().parent.parent.parent
    if results_dir is None:
        results_dir = repo_root / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Native output directory for per-example rows
    native_dir = repo_root / "results" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    system_name = os.environ.get("COG_SYSTEM", "baseline")
    run_id = int(os.environ.get("COG_RUN_ID", "1"))
    config_slug = config.replace(" ", "_").replace("(", "").replace(")", "")
    native_path = (
        native_dir
        / f"mmmu_pro_{config_slug}_{model.replace('/', '_')}_{system_name}_run{run_id}.jsonl"
    )
    native_file = native_path.open("w", encoding="utf-8")

    dataset = load_dataset("MMMU/MMMU_Pro", config, trust_remote_code=True)
    # Dataset may be split as train/validation/test; use first available
    if isinstance(dataset, dict):
        split_name = next(iter(dataset.keys()))
        dataset = dataset[split_name]
    else:
        split_name = "default"
    total = len(dataset)
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, total)))
        total = len(dataset)

    max_options = 10 if "10" in config else 4
    max_letter = "J" if max_options == 10 else "D"
    is_vision = config == "vision"

    results: list[dict] = []
    correct = 0
    start_time = time.perf_counter()
    log_interval = 50 if total > 100 else (10 if total > 20 else 1)
    print(f"[MMMU-Pro] config={config}, {total} samples (model={model})")

    with tempfile.TemporaryDirectory(prefix="mmmupu_pro_images_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx in range(len(dataset)):
            row = dataset[idx]
            sample_id = str(row.get("id", idx))
            question = row.get("question", "") or ""
            options = row.get("options")
            if not isinstance(options, list):
                options = []
            answer_raw = row.get("answer", "")
            gt_letter = _normalize_gt_answer(answer_raw, options, max_options)
            subject = row.get("subject", "")

            image_paths = _get_image_paths_from_row(row, tmpdir_path, sample_id)

            if is_vision:
                prompt = VISION_PROMPT
            else:
                choices = "\n".join(
                    f"{_letter_from_option_index(i)}. {opt}"
                    for i, opt in enumerate(options[:max_options])
                )
                prompt = f"{question.strip()}\n\nOptions:\n{choices}\n{MMMU_PRO_POST_PROMPT}"

            msg: dict = {"role": "user", "content": prompt}
            if image_paths:
                msg["images"] = image_paths

            t0 = time.perf_counter()
            try:
                resp = chat(model=model, messages=[msg])
                response_text = (
                    resp.get("message", {}).get("content", "")
                    if isinstance(resp, dict)
                    else ""
                )
            except Exception as e:
                response_text = ""
                print(f"  [MMMU-Pro] Sample {sample_id} error: {e}")
            latency_s = time.perf_counter() - t0

            score = extract_single_choice_mmmupu(response_text, gt_letter, max_letter)
            if score > 0:
                correct += 1

            row_result = {
                "id": sample_id,
                "subject": subject,
                "answer": answer_raw,
                "gt_letter": gt_letter,
                "response": response_text,
                "correct": bool(score),
                "model": model,
                "config": config,
            }
            results.append(row_result)

            pred_letter = _extract_pred_letter_mmmupu(response_text, max_letter)
            native_row = {
                "benchmark": "mmmu_pro",
                "item_id": sample_id,
                "base_model": model,
                "system": system_name,
                "run_id": run_id,
                "pred_raw": response_text,
                "pred_final": pred_letter or "",
                "gold": row_result["gt_letter"],
                "correct": int(bool(score)),
                "latency_s": latency_s,
                "subject": subject,
                "config": config,
            }
            native_file.write(json.dumps(native_row, ensure_ascii=False) + "\n")

            n_done = idx + 1
            if n_done % log_interval == 0 or n_done == total:
                elapsed = time.perf_counter() - start_time
                acc = correct / n_done if n_done else 0.0
                eta_sec = (elapsed / n_done) * (total - n_done) if n_done else 0
                eta_str = f", ETA ~{eta_sec / 60:.1f} min" if n_done < total and elapsed > 0 else ""
                print(
                    f"[MMMU-Pro] {n_done}/{total} | "
                    f"correct={correct} | acc={acc:.1%} | "
                    f"elapsed={elapsed / 60:.1f} min{eta_str}"
                )

    native_file.close()

    metrics = {
        "total": total,
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
    }

    out_name = output_name or f"mmmupu_pro_{config_slug}_{model.replace('/', '_')}"
    results_path = results_dir / f"{out_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    import pandas as pd
    pd.DataFrame(results).to_csv(
        results_dir / f"{out_name}_results.csv", index=False, encoding="utf-8"
    )
    metrics_path = results_dir / f"{out_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return results, metrics
