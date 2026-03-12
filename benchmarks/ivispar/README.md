# iVISPAR benchmark integration

[iVISPAR](https://github.com/SharkyBamboozle/iVISPAR) is an **Interactive Visual-Spatial Reasoning** benchmark for Vision-Language Models (VLMs). It uses sliding-tile-style puzzles in 3D, 2D, and text modalities. This folder provides an **Ollama agent** and params so you can run iVISPAR with models like **LLaVA** and **Qwen3-VL** locally.

## Quick setup summary

| Step | Action |
|------|--------|
| 1 | Clone iVISPAR: `git clone https://github.com/SharkyBamboozle/iVISPAR.git` |
| 2 | Create env: `conda env create -f Resources/environment.yml` → `conda activate conda_env_iVISPAR` → `pip install ollama` |
| 3 | Create `Data/API-keys/api-keys.txt` (empty file) |
| 4 | Add Ollama agent: paste `ollama_agent_snippet.py` into `Source/Experiment/agent_systems.py`; add `OllamaAgent` branch in `init_experiment_components.py` (see `init_experiment_ollama_patch.txt`) |
| 5 | Set `IVISPAR_ROOT` to your iVISPAR clone path |
| 6 | Run: `python scripts/run_benchmark.py --benchmark ivispar --model llava` (from this repo, with `conda_env_iVISPAR` active) |

---

## Recommended models (Ollama)

Use **multimodal** models only:

- **LLaVA**: `ollama pull llava`
- **Qwen3-VL**: `ollama pull qwen3-vl`

Ensure Ollama is running (`ollama serve` or the Ollama app) before starting experiments.

---

## 1. Clone and set up iVISPAR

```bash
git clone https://github.com/SharkyBamboozle/iVISPAR.git
cd iVISPAR
conda env create -f Resources/environment.yml
conda activate conda_env_iVISPAR
pip install ollama
```

Create a dummy API key file so the agent init does not fail (Ollama does not use it):

```bash
mkdir -p Data/API-keys
touch Data/API-keys/api-keys.txt
```

---

## 2. Add the Ollama agent to iVISPAR

### 2.1 Add `OllamaAgent` to agent_systems.py

- Open `iVISPAR/Source/Experiment/agent_systems.py`.
- Copy the **entire contents** of this project’s  
  `benchmarks/ivispar/ollama_agent_snippet.py`  
  (the class and its methods only; you can skip the first comment block if you prefer).
- Paste at the **end** of `agent_systems.py` (after the last class).
- Ensure `import ollama` is at the top of `agent_systems.py` (add it if missing).

### 2.2 Register the agent in init_experiment_components.py

- Open `iVISPAR/Source/Experiment/init_experiment_components.py`.
- In `init_agent()`, add a branch for `OllamaAgent` as shown in  
  `benchmarks/ivispar/init_experiment_ollama_patch.txt`  
  (before the `else: raise ValueError(...)`).

---

## 3. Run iVISPAR with Ollama

### Option A: From the iVISPAR repo

1. Copy our params file into iVISPAR’s `Source` and run with it:

   ```bash
   cd iVISPAR/Source
   cp /path/to/CogntiveLLMsStudy/benchmarks/ivispar/params_ollama.json .
   # Edit run_experiment.py temporarily to load params_ollama.json instead of params_experiment_ICML.json, or run via Python:
   python -c "
   import asyncio
   import experiment_utilities as util
   from run_experiment import run_experiment
   p = util.load_params_from_json('params_ollama.json')
   asyncio.run(run_experiment(
     games=p['games'],
     agents=p['agents'],
     envs=p['envs'],
     experiment_id=p.get('experiment_id')
   ))
   "
   ```

2. Or change the params filename in `run_experiment.py`’s `if __name__ == "__main__"` to `params_ollama.json` and run:

   ```bash
   python run_experiment.py
   ```

The iVISPAR web app will open in your browser; paste the client ID when prompted. Episodes will run with the Ollama agent (e.g. `llava`).

### Option B: From this project with `run_benchmark.py`

Set the path to the cloned iVISPAR repo and run:

```bash
set IVISPAR_ROOT=C:\path\to\iVISPAR   # Windows
# or: export IVISPAR_ROOT=/path/to/iVISPAR   # Linux/macOS

python scripts/run_benchmark.py --benchmark ivispar --model llava
```

If `IVISPAR_ROOT` is not set, the script prints instructions to set it and run iVISPAR manually.

---

## 4. Params and evaluation

- **params_ollama.json**: Uses `OllamaAgent` with `ollama_model: "llava"`, and `num_game_env: 1` (one episode for testing; increase for full runs). Config set is `SGP_ID_20241212_024852` if that folder exists in `Data/Configs/`.
- **Configs**: iVISPAR expects configs under `Data/Configs/<config_id>/`. Use a `config_id` that exists in your clone (e.g. `SGP_ID_20241212_024852`).
- **Results**: iVISPAR writes experiment outputs under `Data/Experiments/<experiment_id>/`. Use iVISPAR’s **Source/Evaluate** scripts to compute metrics (e.g. completed episodes, step deviation from optimal).

---

## Troubleshooting

- **Script says “Run this from an environment where iVISPAR deps are installed”**  
  Activate iVISPAR’s env: `conda activate conda_env_iVISPAR`, set `IVISPAR_ROOT` to your iVISPAR clone path, then run the benchmark again.

- **Port 1984 already in use**  
  Another process (often a previous run) is using the WebSocket port. Close other iVISPAR runs, or free the port: `netstat -ano | findstr :1984`, then `taskkill /PID <last_column_number> /F`. Then run again.

- **“Client … disconnected” / progress stays 0/10**  
  The browser tab lost the WebSocket connection. Keep the iVISPAR tab **open and in the foreground** from “Please enter the remote client id” until the run finishes. Do not refresh or close it. If it still drops, open the browser console (F12 → Console) on the iVISPAR page and check for errors when it disconnects. Try with `num_game_env: 1` in params for a single-episode test.

- **Ollama not responding**  
  Start Ollama before the run (`ollama serve` or the Ollama app) and ensure the model is pulled (e.g. `ollama pull llava`).

- **FileNotFoundError: Data/Configs/ICML**  
  That config set is missing in your clone. Edit `params_ollama.json` and set `config_id` to a folder that exists under `iVISPAR/Data/Configs/` (e.g. `SGP_ID_20241212_024852`).

- **WebGL errors in browser (drawBuffers, framebuffer release)**  
  The iVISPAR Unity build can hit WebGL limits on some GPUs/drivers. Try **Firefox**; or disable **hardware acceleration** in Chrome/Edge (Settings → System); or update your **GPU driver**. For **multimodal (vision) evaluation** you need `representation_type: "vision"`; only use `"text"` as a temporary fallback if WebGL cannot be fixed (text mode does not test VLM vision).

- **Port free, no errors, but run doesn’t progress or client disconnects**  
  - Use **http://localhost:8000** in the browser (not `http://127.0.0.1:8000`) so the WebSocket connects to the same host.  
  - Wait until the **Unity WebGL app has fully loaded** (progress bar gone, menu or scene visible), then wait another **10–15 seconds** before pasting the client ID so the app is stable.  
  - Keep the browser tab **in the foreground** and don’t refresh. After you paste the ID, the script waits 10 seconds before sending the first puzzle; keep the tab open.  
  - If the client disconnects right after you paste the ID, the Unity app may be crashing on the first message. Open the browser **Developer Console** (F12 → Console) and reproduce the run; check for **red errors** when the disconnect happens (e.g. WebSocket close, JSON parse, or Unity errors). That will show whether the problem is in the web app.  
  - Params are set to **one episode** (`num_game_env: 1`) for testing; increase again once runs are stable.

---

## Links

- [iVISPAR GitHub](https://github.com/SharkyBamboozle/iVISPAR)
- [iVISPAR How-to (experiments)](https://github.com/SharkyBamboozle/iVISPAR/blob/main/Resources/HowTo/how_to_run_experiments.md)
- [Project site](https://microcosm.ai/ivispar) / [ivispar.ai](http://ivispar.ai/)
- [Paper (ACL Anthology)](https://aclanthology.org/2025.emnlp-main.1359/)
