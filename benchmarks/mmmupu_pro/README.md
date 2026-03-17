# MMMU-Pro integration

[MMMU-Pro](https://arxiv.org/abs/2409.02813) is a **multi-discipline multimodal understanding** benchmark: a more robust variant of MMMU with vision-only questions and up to 10 options per question. This integration uses the dataset from [Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro) and runs evaluation with your local Ollama vision model.

## Requirements

- Ollama installed and running
- A vision-capable model (e.g. LLaVA, Qwen2.5-VL): `ollama pull llava`
- Python: `pip install datasets` (included in project `requirements.txt`)

## Quick start

From the repo root:

```bash
# Standard (10 options) – default, full dataset
python scripts/run_benchmark.py --benchmark mmmupu_pro --model llava

# Standard (4 options)
python scripts/run_benchmark.py --benchmark mmmupu_pro --model llava --config "standard (4 options)"

# Vision-only (question embedded in image)
python scripts/run_benchmark.py --benchmark mmmupu_pro --model llava --config vision

# Quick test: first 5 samples
python scripts/run_benchmark.py --benchmark mmmupu_pro --model llava --limit 5
```

Results are written under `results/`:

- `mmmupu_pro_<config>_<model>_results.json` / `.csv` — per-sample answers and correctness
- `mmmupu_pro_<config>_<model>_metrics.json` — total, correct, accuracy (exact_match)

Native per-example rows are also written to **`results/native/`** (e.g. `mmmu_pro_<config>_<model>_<system>_run<id>.jsonl`). Use `--system` and `--run-id` to label runs; then run `python scripts/run_cognitive_report.py` for summaries and cognitive metrics in `results/reports/`.

## Options

| Option | Description |
|--------|-------------|
| `--benchmark mmmupu_pro` | Run MMMU-Pro. |
| `--config` | `"standard (4 options)"`, `"standard (10 options)"`, or `vision`. |
| `--model`, `-m` | Ollama model name (e.g. `llava`, `qwen2.5-vl`). |
| `--limit`, `-n` | Run only the first N samples. |
| `--output-name`, `-o` | Base name for result files. |
| `--system` | System label for native output (default: `baseline`). |
| `--run-id` | Run index for native output (default: `1`). |

## References

- [MMMU-Pro on Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro)
- [Paper (arXiv:2409.02813)](https://arxiv.org/abs/2409.02813)
- [MMMU Benchmark homepage](https://mmmu-benchmark.github.io/)
