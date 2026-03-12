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

[iVISPAR](https://github.com/SharkyBamboozle/iVISPAR) is an interactive visual-spatial reasoning benchmark for **Vision-Language Models (VLMs)**. It uses sliding-tile-style puzzles in 3D, 2D, and text modalities. This project targets VLMs and recommends **Qwen (vision)** and **LLaVA** as the primary Ollama models for running iVISPAR. See `benchmarks/ivispar/README.md` for setup and integration details.

### MMSI-Bench

[MMSI-Bench](https://github.com/InternRobotics/MMSI-Bench) (ICLR 2026) is a multi-image spatial intelligence VQA benchmark: 1,000 multiple-choice questions, each with multiple images. The integration loads the dataset from Hugging Face and runs your Ollama vision model; results are saved under `results/`. It is fully separate from iVISPAR and does not modify any existing benchmark code. See `benchmarks/mmsi_bench/README.md` for usage.

## License

MIT
