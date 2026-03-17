#!/usr/bin/env python3
"""
Scan results/native/ for native benchmark outputs and compute:
- overall accuracy per (benchmark, system, base_model)
- basic per-subcategory accuracies (where applicable)
- simple system deltas (LLM+Cog vs baseline) per benchmark.

This script does not run any benchmarks; it only reads existing native outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cognitive_metrics import (
    load_native_rows,
    accuracy_overall,
    accuracy_by_key,
    compute_system_deltas,
    compute_brier,
    compute_ece,
    compute_abstention_metrics,
    compute_consistency,
    mmmu_pro_vision_dependence_gain,
    visulogic_wrong_rate_by_tag,
    mmsi_delta_by_question_type,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute JSON summaries and cognitive metrics from native outputs.")
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=Path("results") / "native",
        help="Directory containing native JSONL files (default: results/native/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "reports",
        help="Directory to write summary JSON files (default: results/reports/)",
    )
    args = parser.parse_args()

    native_dir: Path = args.native_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not native_dir.is_dir():
        print(f"No native output directory found at {native_dir}")
        return

    files = sorted(native_dir.glob("*.jsonl"))
    if not files:
        print(f"No native JSONL files found in {native_dir}")
        return

    print(f"Found {len(files)} native files in {native_dir}")

    # Load all rows once; for larger setups you could stream/group per file.
    rows = load_native_rows(files)
    if not rows:
        print("No rows loaded from native outputs.")
        return

    # Group by benchmark
    by_benchmark: Dict[str, list[Dict[str, Any]]] = {}
    for r in rows:
        b = str(r.get("benchmark", "unknown"))
        by_benchmark.setdefault(b, []).append(r)

    native_summary: Dict[str, Any] = {}
    cognitive_summary: Dict[str, Any] = {}

    for benchmark, brs in by_benchmark.items():
        overall = accuracy_overall(brs)
        entry: Dict[str, Any] = {"overall_accuracy": overall}

        # Add simple per-subcategory breakdowns using known keys.
        if benchmark == "mmmu_pro":
            entry["accuracy_by_subject"] = accuracy_by_key(brs, "subject")
            entry["accuracy_by_config"] = accuracy_by_key(brs, "config")
        elif benchmark == "visulogic":
            entry["accuracy_by_tag"] = accuracy_by_key(brs, "tag")
        elif benchmark == "mmsi":
            entry["accuracy_by_question_type"] = accuracy_by_key(brs, "question_type")

        native_summary[benchmark] = entry

        # Cognitive metrics: deltas, calibration, abstention, consistency, benchmark-specific
        cog_entry: Dict[str, Any] = dict(compute_system_deltas(brs, benchmark))
        brier = compute_brier(brs)
        if brier is not None:
            cog_entry["brier_score"] = brier
        ece = compute_ece(brs)
        if ece is not None:
            cog_entry["ece"] = ece
        abst = compute_abstention_metrics(brs)
        if abst is not None:
            cog_entry["abstention"] = abst
        cons = compute_consistency(brs)
        if cons.get("items_with_multiple_runs", 0) > 0:
            cog_entry["consistency"] = cons
        if benchmark == "mmmu_pro":
            vdg = mmmu_pro_vision_dependence_gain(brs)
            if vdg is not None:
                cog_entry["vision_dependence_gain"] = vdg
        elif benchmark == "visulogic":
            cog_entry["wrong_rate_by_tag"] = visulogic_wrong_rate_by_tag(brs)
        elif benchmark == "mmsi":
            mmsi_derived = mmsi_delta_by_question_type(brs)
            if mmsi_derived is not None:
                cog_entry["delta_by_question_type"] = mmsi_derived
        cognitive_summary[benchmark] = cog_entry

    with (out_dir / "native_summary.json").open("w", encoding="utf-8") as f:
        json.dump(native_summary, f, indent=2, ensure_ascii=False)

    with (out_dir / "cognitive_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(cognitive_summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote native summary to {out_dir / 'native_summary.json'}")
    print(f"Wrote cognitive metrics to {out_dir / 'cognitive_metrics.json'}")


if __name__ == "__main__":
    main()

