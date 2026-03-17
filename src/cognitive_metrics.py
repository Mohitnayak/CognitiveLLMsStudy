from __future__ import annotations

"""
Post-processing utilities for native benchmark outputs.

Reads JSONL files from results/native/ and computes:
- official-style accuracy (overall and by subcategory fields)
- basic cognitive-friendly aggregates (per-system, per-base-model deltas)
- calibration (Brier, ECE) when confidence is present
- abstention metrics when abstained is present
- consistency (answer agreement across run_id) when multiple runs exist
- benchmark-specific derived metrics
"""

from pathlib import Path
from typing import Iterable, Dict, Any, List

import json
from collections import defaultdict


def load_native_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def accuracy_overall(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(int(r.get("correct", 0)) for r in rows)
    return correct / len(rows)


def accuracy_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        label = str(r.get(key, ""))
        if not label:
            label = "UNKNOWN"
        buckets[label].append(int(r.get("correct", 0)))
    out: dict[str, float] = {}
    for label, vals in buckets.items():
        out[label] = sum(vals) / len(vals) if vals else 0.0
    return out


def group_by_system_and_model(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        system = str(r.get("system", "baseline"))
        base_model = str(r.get("base_model", r.get("model", "")))
        grouped[(system, base_model)].append(r)
    return grouped


def compute_system_deltas(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    """
    Compute per-base-model delta between systems named "baseline" and others.
    """
    grouped = group_by_system_and_model(rows)
    # Organize by base_model then system
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for (system, base_model), rs in grouped.items():
        acc = accuracy_overall(rs)
        by_model[base_model][system] = acc

    deltas: dict[str, dict[str, float]] = {}
    for base_model, sys_accs in by_model.items():
        baseline = sys_accs.get("baseline")
        if baseline is None:
            continue
        for system, acc in sys_accs.items():
            if system == "baseline":
                continue
            deltas.setdefault(base_model, {})[system] = acc - baseline

    all_deltas: list[float] = []
    improved = 0
    for base_model, sys_d in deltas.items():
        for _, d in sys_d.items():
            all_deltas.append(d)
            if d > 0:
                improved += 1

    mean_delta = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
    return {
        "benchmark": benchmark,
        "per_base_model": deltas,
        "mean_delta": mean_delta,
        "num_improvements": improved,
        "num_deltas": len(all_deltas),
    }


# ---------------------------------------------------------------------------
# Calibration (require confidence in [0, 1] per row)
# ---------------------------------------------------------------------------


def compute_brier(rows: list[dict[str, Any]]) -> float | None:
    """
    Brier score: (1/N) * sum (p_i - y_i)^2, y_i = 1 if correct else 0.
    Returns None if no row has a numeric 'confidence' key.
    """
    pairs: list[tuple[float, int]] = []
    for r in rows:
        c = r.get("confidence")
        if c is None:
            continue
        try:
            p = float(c)
        except (TypeError, ValueError):
            continue
        y = 1 if int(r.get("correct", 0)) else 0
        pairs.append((p, y))
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def compute_ece(rows: list[dict[str, Any]], n_bins: int = 10) -> float | None:
    """
    Expected Calibration Error: bin by confidence, compare avg confidence to
    empirical accuracy in each bin, weighted average of |acc - conf|.
    Returns None if no row has 'confidence'.
    """
    pairs: list[tuple[float, int]] = []
    for r in rows:
        c = r.get("confidence")
        if c is None:
            continue
        try:
            p = float(c)
        except (TypeError, ValueError):
            continue
        y = 1 if int(r.get("correct", 0)) else 0
        pairs.append((p, y))
    if not pairs:
        return None
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1) if p < 1.0 else n_bins - 1
        bins[idx].append((p, y))
    ece = 0.0
    total = len(pairs)
    for b in bins:
        if not b:
            continue
        n = len(b)
        avg_conf = sum(x[0] for x in b) / n
        avg_acc = sum(x[1] for x in b) / n
        ece += (n / total) * abs(avg_conf - avg_acc)
    return ece


# ---------------------------------------------------------------------------
# Abstention (require abstained 0/1 per row; optional confidence)
# ---------------------------------------------------------------------------


def compute_abstention_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    coverage = answered / total, answered_accuracy = correct among answered,
    high_confidence_error_rate = wrong among rows with confidence >= threshold.
    Returns None if no row has 'abstained' key.
    """
    if not rows:
        return None
    has_abstained = any("abstained" in r for r in rows)
    if not has_abstained:
        return None
    answered = [r for r in rows if int(r.get("abstained", 0)) == 0]
    total = len(rows)
    coverage = len(answered) / total if total else 0.0
    if not answered:
        return {
            "coverage": coverage,
            "answered_accuracy": 0.0,
            "answered_count": 0,
            "total": total,
        }
    correct_answered = sum(int(r.get("correct", 0)) for r in answered)
    answered_accuracy = correct_answered / len(answered)
    # High-confidence (e.g. >= 0.8) error rate: among answered with conf >= 0.8, fraction wrong
    high_conf_threshold = 0.8
    high_conf_rows = [
        r for r in answered
        if r.get("confidence") is not None and float(r.get("confidence", 0)) >= high_conf_threshold
    ]
    if high_conf_rows:
        high_conf_wrong = sum(1 for r in high_conf_rows if int(r.get("correct", 0)) == 0)
        high_conf_error_rate = high_conf_wrong / len(high_conf_rows)
    else:
        high_conf_error_rate = None
    out: dict[str, Any] = {
        "coverage": coverage,
        "answered_accuracy": answered_accuracy,
        "answered_count": len(answered),
        "total": total,
    }
    if high_conf_error_rate is not None:
        out["high_confidence_error_rate"] = high_conf_error_rate
    return out


# ---------------------------------------------------------------------------
# Consistency (multiple run_id per item: agreement of pred_final across runs)
# ---------------------------------------------------------------------------


def compute_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Group by (item_id, system, base_model). For groups with >1 run_id,
    compute fraction of (run pairs) that agree on pred_final.
    Also report per-item agreement rate and mean agreement over items.
    """
    # key: (item_id, system, base_model) -> list of (run_id, pred_final)
    groups: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get("item_id", "")),
            str(r.get("system", "baseline")),
            str(r.get("base_model", r.get("model", ""))),
        )
        run_id = int(r.get("run_id", 1))
        pred = str(r.get("pred_final", "")).strip()
        groups[key].append((run_id, pred))

    multi_run = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi_run:
        return {
            "items_with_multiple_runs": 0,
            "mean_agreement": None,
            "per_item_agreement": {},
        }

    per_item_agreement: dict[str, float] = {}
    for (item_id, system, base_model), run_preds in multi_run.items():
        preds = [p for _, p in sorted(run_preds, key=lambda x: x[0])]
        n = len(preds)
        same = sum(1 for p in preds if p == preds[0])
        per_item_agreement[f"{item_id}|{system}|{base_model}"] = same / n if n else 0.0

    mean_agreement = sum(per_item_agreement.values()) / len(per_item_agreement) if per_item_agreement else None
    return {
        "items_with_multiple_runs": len(multi_run),
        "mean_agreement": mean_agreement,
        "per_item_agreement": per_item_agreement,
    }


# ---------------------------------------------------------------------------
# Benchmark-specific derived metrics
# ---------------------------------------------------------------------------


def mmmu_pro_vision_dependence_gain(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Vision Dependence Gain = (Cog gain on vision-only) - (Cog gain on standard).
    Requires rows from both configs with system baseline vs other; uses system deltas per config.
    """
    by_config = defaultdict(list)
    for r in rows:
        cfg = r.get("config", "")
        if cfg:
            by_config[cfg].append(r)
    if "vision" not in by_config or not by_config:
        return None
    # Need at least standard (4 or 10) and vision
    standard_configs = [c for c in by_config if "standard" in c.lower()]
    if not standard_configs:
        return None
    # Use first standard config for "standard" delta
    standard_rows = by_config[standard_configs[0]]
    vision_rows = by_config["vision"]
    delta_standard = compute_system_deltas(standard_rows, "mmmu_pro")
    delta_vision = compute_system_deltas(vision_rows, "mmmu_pro")
    # Compare mean_delta: vision_dependence_gain = mean_delta_vision - mean_delta_standard
    mean_std = delta_standard.get("mean_delta")
    mean_vis = delta_vision.get("mean_delta")
    if mean_std is None and mean_vis is None:
        return None
    mean_std = mean_std if mean_std is not None else 0.0
    mean_vis = mean_vis if mean_vis is not None else 0.0
    return {
        "vision_dependence_gain": mean_vis - mean_std,
        "mean_delta_vision": mean_vis,
        "mean_delta_standard": mean_std,
    }


def visulogic_wrong_rate_by_tag(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Wrong rate (1 - accuracy) by tag. High-confidence wrong rate by tag if confidence present."""
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tag = str(r.get("tag", "")) or "UNKNOWN"
        by_tag[tag].append(r)
    out: dict[str, float] = {}
    for tag, rlist in by_tag.items():
        acc = accuracy_overall(rlist)
        out[tag] = 1.0 - acc
    return out


def mmsi_delta_by_question_type(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Per question_type (or failure_mode if present): system delta (Cog - baseline).
    Returns dict of question_type -> { system -> accuracy } and per_type_deltas.
    """
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = "failure_mode" if any(r.get("failure_mode") for r in rows) else "question_type"
    for r in rows:
        t = str(r.get(key, "")) or "UNKNOWN"
        by_type[t].append(r)
    if not by_type:
        return None
    per_type_deltas: dict[str, dict[str, float]] = {}
    for qtype, rlist in by_type.items():
        d = compute_system_deltas(rlist, "mmsi")
        per_base = d.get("per_base_model", {})
        if per_base:
            per_type_deltas[qtype] = {
                base: (sum(sys_d.values()) / len(sys_d)) if sys_d else 0.0
                for base, sys_d in per_base.items()
            }
    return {
        "per_question_type_deltas": per_type_deltas,
        "key_used": key,
    }

