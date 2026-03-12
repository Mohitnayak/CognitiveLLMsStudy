"""
MMSI-Bench runner: load dataset from HuggingFace, run Ollama VLM on each sample,
extract answer (A/B/C/D), and compute accuracy. Kept separate from ivispar and
other benchmarks.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

# MMSI-Bench post-prompt and extraction (aligned with official repo)
MMSI_POST_PROMPT = (
    "\nAnswer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)


def extract_single_choice_with_word_boundary(pred: str, gt: str) -> float:
    """
    Extract single-choice letter (A-D) from model prediction. Matches MMSI-Bench logic.
    Returns 1.0 if prediction matches ground truth letter, 0.0 otherwise.
    """
    if not pred or not gt:
        return 0.0
    # Try ``...`` first
    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1).strip()
    else:
        pattern_2 = r"`([^`]*)`"
        match = re.search(pattern_2, pred)
        if match:
            pred = match.group(1).strip()
    # Try {...}
    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1).strip()
    # Final: single letter A-D with word boundary
    pattern_3 = r"\b[A-D]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred, re.IGNORECASE)
    if not match:
        return 0.0
    pred_letter = match.group().upper()
    answer = gt.strip().upper()
    # Ground truth may be full choice text; take first character if it's A-D
    if len(answer) > 1 and answer[0] in "ABCD":
        answer = answer[0]
    return 1.0 if pred_letter == answer else 0.0


def run_mmsi_bench(
    model: str,
    *,
    limit: int | None = None,
    split: str = "train",
    results_dir: Path | None = None,
    output_name: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Load MMSI-Bench from HuggingFace, run Ollama model on each sample, compute accuracy.

    Args:
        model: Ollama model name (e.g. llava, qwen2.5-vl).
        limit: If set, run only the first N samples (for quick tests).
        split: Dataset split to use (default "train").
        results_dir: Where to save result JSON/CSV; uses repo results/ if None.
        output_name: Base name for result files (default: mmsi_bench_<model>).

    Returns:
        (list of per-sample result dicts, metrics dict with total, correct, exact_match).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "MMSI-Bench requires the 'datasets' package. Install with: pip install datasets"
        ) from None

    from src.ollama_client import chat

    repo_root = Path(__file__).resolve().parent.parent.parent
    if results_dir is None:
        results_dir = repo_root / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("RunsenXu/MMSI-Bench", split=split, trust_remote_code=True)
    except Exception:
        # Some dataset versions only expose one split (e.g. "train" or first key)
        ds_dict = load_dataset("RunsenXu/MMSI-Bench", trust_remote_code=True)
        split = next(iter(ds_dict.keys()))
        dataset = ds_dict[split]
    total = len(dataset)
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, total)))
        total = len(dataset)

    results: list[dict] = []
    correct = 0
    start_time = time.perf_counter()
    # Log every N samples; more frequent when total is small
    log_interval = 50 if total > 100 else (10 if total > 20 else 1)
    print(f"[MMSI-Bench] Starting evaluation: {total} samples (model={model})")

    with tempfile.TemporaryDirectory(prefix="mmsi_bench_images_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx in range(len(dataset)):
            row = dataset[idx]
            sample_id = row.get("id", idx)
            question = row.get("question", "")
            answer = row.get("answer", "")
            question_type = row.get("question_type", "")
            images_data = row.get("images")

            image_paths: list[str] = []
            if images_data is not None:
                for n, img in enumerate(images_data):
                    path = tmpdir_path / f"{sample_id}_{n}.jpg"
                    if hasattr(img, "save"):
                        img.save(path)
                    elif isinstance(img, bytes):
                        path.write_bytes(img)
                    elif isinstance(img, dict) and "bytes" in img:
                        path.write_bytes(img["bytes"])
                    else:
                        path.write_bytes(img if isinstance(img, (bytes, bytearray)) else b"")
                    image_paths.append(str(path))

            prompt = question.strip() + MMSI_POST_PROMPT
            msg: dict = {"role": "user", "content": prompt}
            if image_paths:
                msg["images"] = image_paths

            try:
                resp = chat(model=model, messages=[msg])
                response_text = (
                    resp.get("message", {}).get("content", "")
                    if isinstance(resp, dict)
                    else ""
                )
            except Exception as e:
                response_text = ""
                print(f"  [MMSI-Bench] Sample {sample_id} error: {e}")

            score = extract_single_choice_with_word_boundary(response_text, answer)
            if score > 0:
                correct += 1

            results.append({
                "id": sample_id,
                "question_type": question_type,
                "answer": answer,
                "response": response_text,
                "correct": bool(score),
                "model": model,
            })

            # Progress log every N samples or on last sample
            n_done = idx + 1
            if n_done % log_interval == 0 or n_done == total:
                elapsed = time.perf_counter() - start_time
                acc = correct / n_done if n_done else 0.0
                eta_sec = (elapsed / n_done) * (total - n_done) if n_done else 0
                eta_str = f", ETA ~{eta_sec / 60:.1f} min" if n_done < total and elapsed > 0 else ""
                print(
                    f"[MMSI-Bench] {n_done}/{total} | "
                    f"correct={correct} | acc={acc:.1%} | "
                    f"elapsed={elapsed / 60:.1f} min{eta_str}"
                )

    metrics = {
        "total": total,
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
    }

    # Save results
    out_name = output_name or f"mmsi_bench_{model.replace('/', '_')}"
    import json
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
