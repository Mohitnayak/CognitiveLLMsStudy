# Ollama LLM Benchmark & Evaluation

A Python project for setting up LLMs through [Ollama](https://ollama.com) and evaluating them on benchmarks, including **iVISPAR** (Interactive Visual-Spatial Reasoning), **MMSI-Bench** (Multi-Image Spatial Intelligence), and **MMMU-Pro** (Multi-discipline Multimodal Understanding), with evaluation tooling.

## Prerequisites

- **Conda** or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Ollama](https://ollama.com) installed and running locally
- (Optional) GPU for vision-language model benchmarks

## Setup

1. Create the conda environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate cogntive-llms
   ```

3. (Optional) If not using conda or to refresh dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project structure

| Path | Description |
|------|-------------|
| `src/` | Core modules: Ollama client wrapper and evaluation helpers |
| `scripts/` | Entrypoints: run chat, run benchmarks |
| `benchmarks/` | Benchmark integrations (e.g. iVISPAR, MMSI-Bench, MMMU-Pro, VisuLogic) |
| `results/` | Output for logs and metrics (gitignored) |

## Usage

### Pulling models

For **iVISPAR (multimodal)** use vision models. Recommended:

- **Qwen3-VL**: `ollama pull qwen3-vl`
- **LLaVA**: `ollama pull llava`

Do **not** use text-only models (e.g. `llama3.2`) for image-based tasks.

### Running the benchmark

- **Demo** (quick smoke test):  
  `python scripts/run_benchmark.py --benchmark demo`
- **iVISPAR** (full VLM experiment): clone iVISPAR, add the Ollama agent (see `benchmarks/ivispar/README.md`), set `IVISPAR_ROOT` to the clone path, then run:  
  `python scripts/run_benchmark.py --benchmark ivispar --model llava`
- **MMSI-Bench** (multi-image spatial intelligence):  
  `python scripts/run_benchmark.py --benchmark mmsi_bench --model llava`  
  Optional: `--limit 5` for a quick test. See `benchmarks/mmsi_bench/README.md`.
- **MMMU-Pro** (multi-discipline multimodal understanding):  
  `python scripts/run_benchmark.py --benchmark mmmupu_pro --model llava`  
  Use `--config "standard (10 options)"` (default), `"standard (4 options)"`, or `vision`. Optional: `--limit 5`. See `benchmarks/mmmupu_pro/README.md`.
 - **VisuLogic** (visual reasoning, 1 image + 4-way multiple choice):  
  `python scripts/run_benchmark.py --benchmark visulogic --model llava`  
  Requires `VISULOGIC_DATA_ROOT` pointing to a folder with `data.jsonl` and `images/` from the official dataset. See `benchmarks/visulogic/README.md`.

Example chat with a model:

```bash
python scripts/run_ollama_chat.py
```

### Native evaluation outputs and cognitive report

Benchmark runs (MMSI-Bench, MMMU-Pro, VisuLogic) write **native per-example rows** to a single place: **`results/native/`**. One JSONL file is created per run, e.g.:

- `results/native/mmsi_<model>_<system>_run<id>.jsonl`
- `results/native/mmmu_pro_<config>_<model>_<system>_run<id>.jsonl`
- `results/native/visulogic_<model>_<system>_run<id>.jsonl`

Each line is one example with: `benchmark`, `item_id`, `base_model`, `system`, `run_id`, `pred_raw`, `pred_final`, `gold`, `correct`, `latency_s`, plus benchmark-specific fields (`subject`/`config` for MMMU-Pro, `tag` for VisuLogic, `question_type` for MMSI-Bench). Use `--system` and `--run-id` when running benchmarks to label runs (e.g. baseline vs cognitive layer, or multiple runs for consistency).

A **separate script** reads these native outputs and produces JSON summaries and cognitive metrics (no benchmark execution):

```bash
python scripts/run_cognitive_report.py
```

Optional: `--native-dir results/native` (default) and `--out-dir results/reports`. This writes `results/reports/native_summary.json` (overall and per-subcategory accuracy) and `results/reports/cognitive_metrics.json`. The latter includes system deltas (LLM+Cog vs baseline), and when the native rows contain the right fields: calibration (Brier, ECE), abstention metrics, consistency across runs, and benchmark-specific metrics (e.g. MMMU-Pro vision-dependence gain, VisuLogic wrong rate by tag, MMSI delta by question type). See `src/cognitive_metrics.py` and `src/native_logging.py` for the schema and available metrics.

## Benchmarks

### iVISPAR

[iVISPAR](https://github.com/SharkyBamboozle/iVISPAR) is an interactive visual-spatial reasoning benchmark for **Vision-Language Models (VLMs)**. It uses sliding-tile-style puzzles in 3D, 2D, and text modalities. Use **LLaVA** or **Qwen3-VL** with Ollama.

#### How to set up iVISPAR

1. **Clone iVISPAR** (this project does not include the iVISPAR repo):

   ```bash
   git clone https://github.com/SharkyBamboozle/iVISPAR.git
   cd iVISPAR
   ```

2. **Create iVISPAR’s conda environment** and install Ollama:

   ```bash
   conda env create -f Resources/environment.yml
   conda activate conda_env_iVISPAR
   pip install ollama
   ```

3. **Create a dummy API key file** (required by the experiment; Ollama does not use it):

   ```bash
   mkdir -p Data/API-keys
   touch Data/API-keys/api-keys.txt
   ```
   On Windows: `mkdir Data\API-keys` then create an empty `Data\API-keys\api-keys.txt`.

4. **Add the Ollama agent to iVISPAR** (so experiments can call your local Ollama):
   - **`iVISPAR/Source/Experiment/agent_systems.py`**: add `import ollama` at the top if missing; paste the full contents of **`benchmarks/ivispar/ollama_agent_snippet.py`** at the end (the `OllamaAgent` class).
   - **`iVISPAR/Source/Experiment/init_experiment_components.py`**: in `init_agent()`, add the `OllamaAgent` branch **before** the `else: raise ValueError(...)` line, as in **`benchmarks/ivispar/init_experiment_ollama_patch.txt`**.

5. **Set `IVISPAR_ROOT`** to the path of your iVISPAR clone (use your actual path):

   ```bash
   # Windows (PowerShell or CMD)
   set IVISPAR_ROOT=C:\path\to\iVISPAR

   # Linux/macOS
   export IVISPAR_ROOT=/path/to/iVISPAR
   ```

#### How to run iVISPAR

**You must use the iVISPAR conda environment** (it has `ollama` and iVISPAR deps) **and set `IVISPAR_ROOT`** before running.

**PowerShell** (use this syntax — `set` does not work in PowerShell):

```powershell
conda activate conda_env_iVISPAR
$env:IVISPAR_ROOT = "C:\Users\Administrator\Desktop\CogntiveLLMsStudy\iVISPAR"
cd C:\Users\Administrator\Desktop\CogntiveLLMsStudy
python scripts/run_benchmark.py --benchmark ivispar --model llava
```

**CMD** (Command Prompt):

```cmd
conda activate conda_env_iVISPAR
set IVISPAR_ROOT=C:\Users\Administrator\Desktop\CogntiveLLMsStudy\iVISPAR
cd C:\Users\Administrator\Desktop\CogntiveLLMsStudy
python scripts/run_benchmark.py --benchmark ivispar --model llava
```

- The iVISPAR web app opens at http://localhost:8000. Wait for the Unity app to fully load (progress bar gone).
- When the terminal shows **"Please enter the remote client id:"**, copy the client ID from the browser (e.g. "Copy ID" in the app), **paste it in the terminal**, and press Enter.
- **Keep the browser tab open and in the foreground** until the run finishes. Results go to `iVISPAR/Data/Experiments/`.

Other vision models (e.g. Qwen3-VL):

```bash
python scripts/run_benchmark.py --benchmark ivispar --model qwen3-vl
```

**If the command fails:** `No module named 'ollama'` means you are not in `conda_env_iVISPAR` — run `conda activate conda_env_iVISPAR` first. "IVISPAR_ROOT to be set" means set the env var as in the second line above (adjust the path if iVISPAR is elsewhere).

For troubleshooting (port 1984, client disconnects, WebGL errors), see **`benchmarks/ivispar/README.md`**.

### MMSI-Bench

[MMSI-Bench](https://github.com/InternRobotics/MMSI-Bench) (ICLR 2026) is a multi-image spatial intelligence VQA benchmark: 1,000 multiple-choice questions, each with multiple images. The integration loads the dataset from Hugging Face and runs your Ollama vision model; results are saved under `results/`. It is fully separate from iVISPAR and does not modify any existing benchmark code. See `benchmarks/mmsi_bench/README.md` for usage.

### MMMU-Pro

[MMMU-Pro](https://arxiv.org/abs/2409.02813) is a more robust multi-discipline multimodal understanding benchmark (vision-only setting and up to 10 options). The integration loads [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) from Hugging Face and runs your Ollama vision model; results are saved under `results/`. Configs: `standard (4 options)`, `standard (10 options)`, or `vision`. See `benchmarks/mmmupu_pro/README.md` for usage.

### VisuLogic

[VisuLogic](https://visulogic-benchmark.github.io/VisuLogic) is a visual reasoning benchmark with 1,000 multiple-choice (A/B/C/D) questions, each paired with an image. The integration reads the official `data.jsonl` and `images/` from a local folder (downloaded from [VisuLogic/VisuLogic](https://huggingface.co/datasets/VisuLogic/VisuLogic)) and runs your Ollama vision model; results are saved under `results/`. See `benchmarks/visulogic/README.md` for dataset setup and usage.

## License

MIT
