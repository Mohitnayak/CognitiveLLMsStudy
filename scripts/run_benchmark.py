#!/usr/bin/env python3
"""
Entrypoint to run benchmarks (demo, iVISPAR, MMSI-Bench) via Ollama and record evaluation results.
Uses src.ollama_client and src.evaluation; for iVISPAR launches the full experiment if IVISPAR_ROOT is set;
for MMSI-Bench loads the dataset from Hugging Face and runs locally (no clone required).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow importing from src
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    RESULTS_DIR,
    compute_metrics,
    run_prompt_set,
    save_results,
)
from src.ollama_client import list_models


def run_ivispar_experiment(ivispar_root: Path, model: str) -> None:
    """Run iVISPAR experiment via their run_experiment (requires OllamaAgent added to iVISPAR)."""
    source_dir = ivispar_root / "Source" / "Experiment"
    if not source_dir.is_dir():
        print(f"IVISPAR Source/Experiment not found at {source_dir}")
        return
    params_path = REPO_ROOT / "benchmarks" / "ivispar" / "params_ollama.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    # Override ollama_model with --model
    for agent_cfg in params.get("agents", {}).values():
        if agent_cfg.get("agent_type") == "OllamaAgent":
            agent_cfg["ollama_model"] = model
            break
    source_exp = ivispar_root / "Source" / "Experiment"
    os.chdir(source_exp)
    if str(source_exp) not in sys.path:
        sys.path.insert(0, str(source_exp))
    try:
        import experiment_utilities as util
        from run_experiment import run_experiment
    except ImportError as e:
        print(f"Cannot import iVISPAR experiment modules: {e}")
        print()
        print("Use iVISPAR's conda environment so all deps are available:")
        print("  1. conda activate conda_env_iVISPAR")
        print("     (If it doesn't exist: cd iVISPAR && conda env create -f Resources/environment.yml && pip install ollama)")
        print("  2. set IVISPAR_ROOT=c:\\path\\to\\iVISPAR   (or your actual clone path)")
        print("  3. python scripts/run_benchmark.py --benchmark ivispar --model llava")
        print()
        print("See benchmarks/ivispar/README.md for full setup.")
        return
    print(f"Running iVISPAR with model '{model}' (experiment_id: {params.get('experiment_id', 'Ollama')})...")
    asyncio.run(
        run_experiment(
            games=params["games"],
            agents=params["agents"],
            envs=params["envs"],
            experiment_id=params.get("experiment_id"),
        )
    )
    print("iVISPAR experiment finished. Results are in iVISPAR/Data/Experiments/")
    print("Use iVISPAR Source/Evaluate scripts to compute metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ollama benchmark and evaluation")
    parser.add_argument(
        "--model",
        "-m",
        default="llava",
        help="Ollama model name (e.g. llava, qwen2-vl, qwen3-vl)",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        default="demo",
        choices=["demo", "ivispar", "mmsi_bench"],
        help="Benchmark to run (demo = small prompt set for testing)",
    )
    parser.add_argument(
        "--output-name",
        "-o",
        default=None,
        help="Base name for result files (default: <benchmark>_<model>)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="For mmsi_bench only: run first N samples (default: full dataset)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        models = list_models()
        print("Available models:")
        for m in models:
            name = getattr(m, "name", m.get("name") if isinstance(m, dict) else str(m))
            print(f"  - {name}")
        return

    # Demo: small set of text prompts to verify pipeline
    if args.benchmark == "demo":
        prompts = [
            {"prompt": "What is 2 + 2? Reply with one number only.", "expected_answer": "4"},
            {"prompt": "Say 'hello' and nothing else.", "expected_answer": "hello"},
        ]
        print(f"Running benchmark '{args.benchmark}' with model '{args.model}' ({len(prompts)} items)...")
        results = run_prompt_set(args.model, prompts)
        metrics = compute_metrics(results)
        out_name = args.output_name or f"{args.benchmark}_{args.model.replace('/', '_')}"
        results_path, metrics_path = save_results(results, metrics, name=out_name, results_dir=RESULTS_DIR)
        print(f"Results: {results_path}")
        if metrics_path:
            print(f"Metrics: {metrics_path}")
        print(f"Exact match: {metrics.get('exact_match', 0):.2%} ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
        return

    if args.benchmark == "ivispar":
        ivispar_root = os.environ.get("IVISPAR_ROOT")
        if not ivispar_root or not Path(ivispar_root).is_dir():
            print("iVISPAR benchmark requires the iVISPAR repo and IVISPAR_ROOT to be set.")
            print("1. Clone: git clone https://github.com/SharkyBamboozle/iVISPAR.git")
            print("2. Add OllamaAgent to iVISPAR (see benchmarks/ivispar/README.md)")
            print("3. Set IVISPAR_ROOT to the clone path.")
            print("   PowerShell: $env:IVISPAR_ROOT = \"C:\\path\\to\\iVISPAR\"")
            print("   CMD:        set IVISPAR_ROOT=C:\\path\\to\\iVISPAR")
            print("4. Run again: python scripts/run_benchmark.py --benchmark ivispar --model llava")
            return
        run_ivispar_experiment(Path(ivispar_root), args.model)
        return

    if args.benchmark == "mmsi_bench":
        from benchmarks.mmsi_bench.runner import run_mmsi_bench
        out_name = args.output_name or f"mmsi_bench_{args.model.replace('/', '_')}"
        limit_str = f" (first {args.limit} samples)" if args.limit else ""
        print(f"Running MMSI-Bench with model '{args.model}'{limit_str}...")
        results, metrics = run_mmsi_bench(
            args.model,
            limit=args.limit,
            results_dir=RESULTS_DIR,
            output_name=out_name,
        )
        print(f"Results saved to {RESULTS_DIR / (out_name + '_results.json')}")
        print(f"Accuracy: {metrics['exact_match']:.2%} ({metrics['correct']}/{metrics['total']})")
        return


if __name__ == "__main__":
    main()
