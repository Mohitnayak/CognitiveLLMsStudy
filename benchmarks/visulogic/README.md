# VisuLogic integration

[VisuLogic](https://visulogic-benchmark.github.io/VisuLogic) is a benchmark for **visual reasoning** in multimodal large language models. Each sample has one image, a multiple-choice question, and an answer label (A/B/C/D). This integration uses the official dataset from [Hugging Face](https://huggingface.co/datasets/VisuLogic/VisuLogic) and runs evaluation with your local Ollama vision model.

## Requirements

- Ollama installed and running
- A vision-capable model (e.g. LLaVA, Qwen3-VL, MiniCPM-V): `ollama pull llava`
- Python dependencies:
  - `pip install datasets` (already included in the project requirements)
  - For CSV output: `pip install pandas` (optional; JSON is always saved)

## Prepare the VisuLogic dataset

**Option A – Download via script (recommended)**

From the repo root (requires `huggingface_hub`, e.g. `pip install huggingface_hub` or use the project environment):

```bash
python scripts/download_visulogic.py
```

This downloads the dataset to `data/visulogic/` by default. To use another folder:

```bash
python scripts/download_visulogic.py --out-dir C:\data\VisuLogic
```

Then set `VISULOGIC_DATA_ROOT` to that folder (e.g. `data/visulogic` when using the default; use the full path, e.g. `C:\Users\...\CogntiveLLMsStudy\data\visulogic`).

**Option B – Manual download**

1. Go to the [VisuLogic dataset page](https://huggingface.co/datasets/VisuLogic/VisuLogic), download `data.jsonl` and `images.zip`.
2. Create a folder and unzip so you have `data.jsonl` and `images/` (e.g. `00000.png`, ...) inside it.
3. Set `VISULOGIC_DATA_ROOT` to that folder, e.g. in PowerShell:

   ```powershell
   $env:VISULOGIC_DATA_ROOT = "C:\path\to\folder\containing\data.jsonl\and\images"
   ```

   Or in CMD:

   ```cmd
   set VISULOGIC_DATA_ROOT=C:\path\to\folder\containing\data.jsonl\and\images
   ```

## Running the VisuLogic benchmark

From the repo root:

```bash
python scripts/run_benchmark.py --benchmark visulogic --model llava
```

Options:

- `--limit N` – run only the first N samples, e.g. `--limit 50` for a quick test.
- `--output-name NAME` – custom base name for result files (default: `visulogic_<model>`).
- `--system` – system label for native output (default: `baseline`).
- `--run-id` – run index for native output (default: `1`).

Results are written under `results/`:

- `visulogic_<model>_results.json` / `.csv` – per-sample answers and correctness
- `visulogic_<model>_metrics.json` – total, correct, accuracy (`exact_match`)

Native per-example rows are also written to **`results/native/`** (e.g. `visulogic_<model>_<system>_run<id>.jsonl`). Use `--system` and `--run-id` to label runs; then run `python scripts/run_cognitive_report.py` for summaries and cognitive metrics in `results/reports/`.

## Notes

- The integration expects the same JSONL format as the official dataset, with fields: `image_path`, `label`, `question`, `tag`, `id`.
- The question text already includes the option descriptions (A/B/C/D). The runner appends a short post-prompt to make the model answer with the option letter only, enclosed in backticks.
- Metrics are identical in spirit to the official evaluation: simple accuracy over the four-way multiple choice (exact match on the option letter).

