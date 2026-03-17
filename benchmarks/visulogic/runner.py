"""
VisuLogic runner: load data.jsonl from a local VisuLogic dataset folder,
run an Ollama VLM on each sample with the corresponding image, and compute
accuracy (exact match on option letter A–D).

Dataset layout expected (matching the Hugging Face VisuLogic dataset):

    VISULOGIC_DATA_ROOT/
      data.jsonl
      images/
        00000.png
        00001.png
        ...

Each row in data.jsonl has:
  - image_path: e.g. "images/00000.png"
  - label: correct option letter ("A", "B", "C", "D")
  - question: question text (includes the option descriptions)
  - tag: reasoning category
  - id: string id
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from src.ollama_client import chat


VISULOGIC_POST_PROMPT = (
    "\nAnswer with the option's letter (A, B, C, or D) only. "
    "Enclose the letter within ``."
)


def extract_single_choice_visulogic(pred: str, gt_letter: str) -> float:
    """
    Extract single-choice letter (A–D) from model prediction.
    Returns 1.0 if prediction matches ground truth letter, 0.0 otherwise.
    """
    if not pred or not gt_letter:
        return 0.0
    text = pred.strip()
    # Try ``...`` first
    m = re.search(r"``([^`]*)``", text)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"`([^`]*)`", text)
        if m:
            text = m.group(1).strip()
    # Try {...}
    m = re.search(r"\{([^}]*)\}", text)
    if m:
        text = m.group(1).strip()
    # Final: single letter A–D with word boundary
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", text, re.IGNORECASE)
    if not m:
        return 0.0
    pred_letter = m.group().upper()
    return 1.0 if pred_letter == gt_letter.strip().upper() else 0.0


def _extract_pred_letter_visulogic(pred: str) -> str:
    """Extract single-choice letter (A–D) from model prediction; returns "" if none found."""
    if not pred:
        return ""
    text = pred.strip()
    m = re.search(r"``([^`]*)``", text)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"`([^`]*)`", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{([^}]*)\}", text)
    if m:
        text = m.group(1).strip()
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", text, re.IGNORECASE)
    if not m:
        return ""
    return m.group().upper()


def _load_visulogic_rows(data_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_visulogic(
    model: str,
    *,
    data_root: Path,
    limit: int | None = None,
    results_dir: Path | None = None,
    output_name: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Run VisuLogic evaluation using an Ollama VLM.

    Args:
        model: Ollama model name (e.g. llava, qwen3-vl, minicpm-v).
        data_root: Path to VisuLogic data folder containing data.jsonl and images/.
        limit: If set, run only the first N samples.
        results_dir: Where to save result JSON/CSV; uses repo results/ if None.
        output_name: Base name for result files (default: visulogic_<model>).
    """
    data_root = Path(data_root)
    data_path = data_root / "data.jsonl"
    images_root = data_root
    if not data_path.is_file():
        msg = (
            f"VisuLogic data.jsonl not found at {data_path}. "
            "VISULOGIC_DATA_ROOT is set to the folder that must contain data.jsonl and images/. "
            "Download the dataset from https://huggingface.co/datasets/VisuLogic/VisuLogic, "
            "place data.jsonl and the images/ folder there, then set VISULOGIC_DATA_ROOT to that folder "
            "(use your real path, not the example from the docs)."
        )
        raise FileNotFoundError(msg)

    rows = _load_visulogic_rows(data_path)
    total = len(rows)
    if total == 0:
        raise RuntimeError(f"No rows found in {data_path}")
    if limit is not None and limit > 0:
        rows = rows[: min(limit, total)]
        total = len(rows)

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
    model_safe = model.replace("/", "_").replace(":", "_")
    native_path = (
        native_dir
        / f"visulogic_{model_safe}_{system_name}_run{run_id}.jsonl"
    )

    results: list[dict] = []
    correct = 0
    start_time = time.perf_counter()
    log_interval = 50 if total > 100 else (10 if total > 20 else 1)
    print(f"[VisuLogic] Starting evaluation: {total} samples (model={model})")

    with native_path.open("w", encoding="utf-8") as native_file:
        for idx, row in enumerate(rows):
            sample_id = row.get("id", str(idx))
            question = row.get("question", "") or ""
            label = row.get("label", "") or ""
            tag = row.get("tag", "") or ""
            image_rel = row.get("image_path", "") or ""
            image_path = images_root / image_rel

            prompt = question.strip() + VISULOGIC_POST_PROMPT
            msg: dict[str, Any] = {"role": "user", "content": prompt}
            if image_path.is_file():
                msg["images"] = [str(image_path)]

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
                print(f"  [VisuLogic] Sample {sample_id} error: {e}")
            latency_s = time.perf_counter() - t0

            score = extract_single_choice_visulogic(response_text, label)
            if score > 0:
                correct += 1

            row_result = {
                "id": sample_id,
                "tag": tag,
                "label": label,
                "response": response_text,
                "correct": bool(score),
                "model": model,
            }
            results.append(row_result)

            pred_letter = _extract_pred_letter_visulogic(response_text)
            native_row = {
                "benchmark": "visulogic",
                "item_id": sample_id,
                "base_model": model,
                "system": system_name,
                "run_id": run_id,
                "pred_raw": response_text,
                "pred_final": pred_letter or "",
                "gold": label,
                "correct": int(bool(score)),
                "latency_s": latency_s,
                "tag": tag,
            }
            native_file.write(json.dumps(native_row, ensure_ascii=False) + "\n")

            n_done = idx + 1
            if n_done % log_interval == 0 or n_done == total:
                elapsed = time.perf_counter() - start_time
                acc = correct / n_done if n_done else 0.0
                eta_sec = (elapsed / n_done) * (total - n_done) if n_done else 0
                eta_str = (
                    f", ETA ~{eta_sec / 60:.1f} min"
                    if n_done < total and elapsed > 0
                    else ""
                )
                print(
                    f"[VisuLogic] {n_done}/{total} | "
                    f"correct={correct} | acc={acc:.1%} | "
                    f"elapsed={elapsed / 60:.1f} min{eta_str}"
                )

    metrics = {
        "total": total,
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
    }

    out_name = output_name or f"visulogic_{model.replace('/', '_')}"
    results_path = results_dir / f"{out_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(
            results_dir / f"{out_name}_results.csv",
            index=False,
            encoding="utf-8",
        )
    except Exception:
        # pandas is optional; JSON is always written
        pass
    metrics_path = results_dir / f"{out_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return results, metrics

