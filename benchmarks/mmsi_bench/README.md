# MMSI-Bench integration

[MMSI-Bench](https://github.com/InternRobotics/MMSI-Bench) is a **multi-image spatial intelligence** VQA benchmark (ICLR 2026). Each sample has multiple images and a multiple-choice question (A/B/C/D). This integration uses the dataset from Hugging Face and runs evaluation with your local Ollama vision model **without modifying any existing iVISPAR or other benchmark code**.

## Requirements

- Ollama installed and running
- A vision-capable model (e.g. LLaVA, Qwen2.5-VL): `ollama pull llava`
- Python dependency: `pip install datasets` (or use the project `requirements.txt`, which includes it for MMSI-Bench)

## Quick start

From the repo root:

```bash
# Full benchmark (1000 samples; can take a while)
python scripts/run_benchmark.py --benchmark mmsi_bench --model llava

# Quick test: first 5 samples
python scripts/run_benchmark.py --benchmark mmsi_bench --model llava --limit 5
```

Results and metrics are written under `results/`:

- `mmsi_bench_<model>_results.json` / `.csv` — per-sample answers and correctness
- `mmsi_bench_<model>_metrics.json` — total count, correct count, accuracy (exact_match)

Native per-example rows are also written to **`results/native/`** (one JSONL file per run, e.g. `mmsi_<model>_<system>_run<id>.jsonl`). Use `--system` and `--run-id` to label runs. Then run `python scripts/run_cognitive_report.py` to generate accuracy summaries and cognitive metrics in `results/reports/`.

## Options

| Option | Description |
|--------|-------------|
| `--benchmark mmsi_bench` | Run MMSI-Bench (no effect on ivispar or demo). |
| `--model`, `-m` | Ollama model name (e.g. `llava`, `qwen2.5-vl`). |
| `--limit`, `-n` | Run only the first N samples (e.g. `--limit 10`). |
| `--output-name`, `-o` | Base name for result files (default: `mmsi_bench_<model>`). |
| `--system` | System label for native output (default: `baseline`). |
| `--run-id` | Run index for native output (default: `1`). |

## How it works

1. **Dataset**: Loads [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) via the `datasets` library (no clone of the MMSI-Bench repo required).
2. **Inference**: For each sample, images are decoded to temporary files and sent to Ollama with the question and the same post-prompt used in the official benchmark (answer with the option letter in `` ``).
3. **Answer extraction**: Uses the same regex logic as the official MMSI-Bench (single letter A–D with word boundary).
4. **Metrics**: Accuracy = correct / total (exact match of the chosen letter to the ground truth).

This runner is self-contained under `benchmarks/mmsi_bench/` and does not touch `benchmarks/ivispar/` or the iVISPAR experiment scripts.

## References

- [MMSI-Bench GitHub](https://github.com/InternRobotics/MMSI-Bench)
- [MMSI-Bench on Hugging Face](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)
- [Paper (ICLR 2026)](https://arxiv.org/abs/2505.23764)
