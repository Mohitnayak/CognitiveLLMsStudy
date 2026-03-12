# Ollama LLM Benchmark & Evaluation

A Python project for setting up LLMs through [Ollama](https://ollama.com) and evaluating them on benchmarks, including **iVISPAR** (Interactive Visual-Spatial Reasoning) and **MMSI-Bench** (Multi-Image Spatial Intelligence), with evaluation tooling.

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
| `benchmarks/` | Benchmark integrations (e.g. iVISPAR, MMSI-Bench) |
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

Example chat with a model:

```bash
python scripts/run_ollama_chat.py
```

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

From this repo, in a terminal where `IVISPAR_ROOT` is set:

```bash
conda activate conda_env_iVISPAR
set IVISPAR_ROOT=C:\path\to\iVISPAR
cd C:\path\to\CogntiveLLMsStudy
python scripts/run_benchmark.py --benchmark ivispar --model llava
```

- The iVISPAR web app opens at http://localhost:8000. Wait for the Unity app to fully load (progress bar gone).
- When the terminal shows **"Please enter the remote client id:"**, copy the client ID from the browser (e.g. "Copy ID" in the app), **paste it in the terminal**, and press Enter.
- **Keep the browser tab open and in the foreground** until the run finishes. Results go to `iVISPAR/Data/Experiments/`.

Other vision models (e.g. Qwen3-VL):

```bash
python scripts/run_benchmark.py --benchmark ivispar --model qwen3-vl
```

For troubleshooting (port 1984, client disconnects, WebGL errors), see **`benchmarks/ivispar/README.md`**.

### MMSI-Bench

[MMSI-Bench](https://github.com/InternRobotics/MMSI-Bench) (ICLR 2026) is a multi-image spatial intelligence VQA benchmark: 1,000 multiple-choice questions, each with multiple images. The integration loads the dataset from Hugging Face and runs your Ollama vision model; results are saved under `results/`. It is fully separate from iVISPAR and does not modify any existing benchmark code. See `benchmarks/mmsi_bench/README.md` for usage.

## License

MIT
