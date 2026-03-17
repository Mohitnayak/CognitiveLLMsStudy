"""
Shared schema and helpers for native benchmark output rows.

All benchmark runners (MMSI-Bench, MMMU-Pro, VisuLogic) write one JSONL row per
example to results/native/. This module defines the common row schema and
a helper to append a row so runners can optionally use it for consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Schema (log one row per example)
# ---------------------------------------------------------------------------
# Required in every row:
#   benchmark: str (e.g. "mmsi", "mmmu_pro", "visulogic")
#   item_id: str
#   base_model: str
#   system: str (e.g. "baseline", "cog")
#   run_id: int
#   pred_raw: str
#   pred_final: str (extracted option letter or "")
#   gold: str
#   correct: int (0 or 1)
#   latency_s: float
#
# Optional (for cognitive metrics when present):
#   confidence: float in [0, 1]
#   abstained: int (0 or 1)
#   cost_usd: float
#
# Benchmark-specific (include when available):
#   MMMU-Pro: subject, config
#   VisuLogic: tag
#   MMSI-Bench: question_type, failure_mode
# ---------------------------------------------------------------------------


REQUIRED_KEYS = frozenset({
    "benchmark", "item_id", "base_model", "system", "run_id",
    "pred_raw", "pred_final", "gold", "correct", "latency_s",
})


def ensure_native_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of row with required keys present (fill missing with defaults).
    Does not strip extra keys; use for validation before writing.
    """
    out = dict(row)
    for key in REQUIRED_KEYS:
        if key not in out:
            if key == "correct":
                out[key] = 0
            elif key == "run_id":
                out[key] = 1
            elif key in ("pred_raw", "pred_final", "gold"):
                out[key] = ""
            elif key == "latency_s":
                out[key] = 0.0
            else:
                out[key] = ""
    return out


def write_native_row(fp: TextIO, row: dict[str, Any], ensure_keys: bool = True) -> None:
    """
    Append one native row as a JSON line to fp.
    If ensure_keys is True, fill missing required keys with defaults.
    """
    to_write = ensure_native_row(row) if ensure_keys else row
    fp.write(json.dumps(to_write, ensure_ascii=False) + "\n")


def native_output_path(
    repo_root: Path,
    benchmark: str,
    model: str,
    system: str,
    run_id: int,
    *,
    config_slug: str | None = None,
) -> Path:
    """
    Build the standard native JSONL path for a run.
    config_slug: for MMMU-Pro only (e.g. "standard_10_options", "vision").
    """
    native_dir = repo_root / "results" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    if config_slug:
        name = f"{benchmark}_{config_slug}_{safe_model}_{system}_run{run_id}.jsonl"
    else:
        name = f"{benchmark}_{safe_model}_{system}_run{run_id}.jsonl"
    return native_dir / name
