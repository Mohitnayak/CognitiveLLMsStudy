"""
Evaluation helpers: run a set of prompts through the Ollama client, collect answers,
compute basic metrics, and save results to the results/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Default results directory (repo root / results)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_prompt_set(
    model: str,
    prompts: list[dict[str, Any]],
    *,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
    results_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run a list of prompt items through the model and collect responses.

    Each item in `prompts` can be:
      - {"prompt": "text"} for text-only
      - {"prompt": "text", "image_path": "path"} or "image_base64": "..." for VLMs

    Uses src.ollama_client.chat or chat_with_image by default.

    Returns:
        List of dicts with keys: prompt (or prompt key), response, model, and any extra keys.
    """
    from src.ollama_client import chat, chat_with_image

    chat_fn = chat_fn or chat
    out: list[dict[str, Any]] = []
    for item in prompts:
        prompt_text = item.get("prompt", item.get("content", ""))
        image_path = item.get("image_path")
        image_base64 = item.get("image_base64")
        if image_path or image_base64:
            resp = chat_with_image(
                model=model,
                content=prompt_text,
                image_path=image_path,
                image_base64=image_base64,
            )
        else:
            resp = chat(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
            )
        content = (
            resp.get("message", {}).get("content", "")
            if isinstance(resp, dict)
            else getattr(getattr(resp, "message", None), "content", "")
        )
        record = {
            **{k: v for k, v in item.items() if k not in ("image_path", "image_base64")},
            "response": content,
            "model": model,
        }
        out.append(record)
    return out


def compute_metrics(
    results: list[dict[str, Any]],
    *,
    answer_key: str = "expected_answer",
    response_key: str = "response",
) -> dict[str, Any]:
    """
    Compute simple metrics over results that have expected_answer and response.

    Returns dict with e.g. exact_match (fraction), total, correct.
    """
    if not results:
        return {"total": 0, "correct": 0, "exact_match": 0.0}
    total = len(results)
    correct = 0
    for r in results:
        expected = r.get(answer_key, "")
        actual = (r.get(response_key) or "").strip()
        if expected and actual and _normalize(actual) == _normalize(expected):
            correct += 1
    return {
        "total": total,
        "correct": correct,
        "exact_match": correct / total if total else 0.0,
    }


def _normalize(s: str) -> str:
    return " ".join(s.lower().split()).strip()


def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
    *,
    name: str = "eval",
    results_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """
    Save results to results/ as JSON and optional CSV; optionally save metrics as JSON.

    Returns:
        (path_to_results_file, path_to_metrics_file or None)
    """
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(results)
    csv_path = results_dir / f"{name}_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    metrics_path = None
    if metrics is not None:
        metrics_path = results_dir / f"{name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return results_path, metrics_path
