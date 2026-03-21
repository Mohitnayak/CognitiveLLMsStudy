# CogLLMVispar

Interactive **visual–spatial reasoning** experiments that connect **[iVISPAR](https://github.com/SharkyBamboozle/iVISPAR)** (Unity WebGL puzzle + Python driver) with multiple **agent backends**: cloud APIs, **Ollama** (local VLMs), and optional **reinforcement-learning** layers.

---

## 1. About

- **Benchmark:** iVISPAR runs *sliding-style* puzzles in 3D (and related modalities). The Python client talks to the game over **WebSockets**: each step the sim sends state (e.g. board text and/or screenshots), the agent returns a **move command**, the sim applies it, and the loop continues until the goal is reached or `max_game_length` is hit.
- **This repo** is a **monorepo**:
  - **`agent/`** — Top-level Python package: `OllamaAgent`, **`CognitiveAgent`** (multi-candidate + reranking / DQN), `OnlineRLAgent`, parsing and board utilities.
  - **`iVISPAR/`** — Patched experiment stack (`Source/Experiment`, `Data/Params`, `Data/Configs`, Unity project under `Source/iVISPAR`, WebGL build under `iVISPAR/iVISPAR/`).
- **Git hygiene:** `iVISPAR/Data/Experiments/` (run outputs) and `iVISPAR/Data/API-keys/` are **gitignored** so the repo stays small. **Puzzle datasets** live under `iVISPAR/Data/Configs/<config_id>/` and are tracked. Create `Data/API-keys/api-keys.txt` locally when using API-based agents.

---

## 2. Components — agents and how they use the benchmark

All interactive agents follow the same outer loop (`action_perception_loop.py`): **observation in → one or more model calls → action string out → sim step → next observation**. They differ in *what* computes the move.

| Kind | `agent_type` (in params JSON) | Role |
|------|----------------------------------|------|
| **API-based LLM** | `GPT4Agent`, `ClaudeAgent`, `GeminiAgent`, `DeepseekAgent`, … | Remote multimodal/text APIs. Reads keys from `api_keys_file_path`, instructions from `instruction_prompt_file_path`. One (or few) calls per step; output is parsed into `move …` commands. Implemented in `iVISPAR/Source/Experiment/agent_systems.py`. |
| **Ollama (local LLM / VLM)** | `OllamaAgent` | Calls **Ollama** HTTP API (`model_type`, `ollama_base_url`). Same observation → prompt + optional images → single completion → parsed move. Good baseline for local models (`llama3.1:8b`, `qwen3-vl`, etc.). Code: `agent/ollama_agent.py`. |
| **Cognitive layer (LLM + rerank / RL selector)** | `CognitiveAgent` | **Proposes K candidate moves** with the LLM (Ollama), then **chooses one**: `rerank_mode`: `critique` (second LLM call), `first_valid`, `random`, or **`dqn`** (PyTorch DQN over candidate *indices* using `board_data` from the sim). Still **one** move sent to Unity per step; the RL part learns *which proposal slot* to trust, not a separate game policy. Code: `agent/cognitive_agent.py`, `agent/cognitive_selector_dqn.py`. |
| **Pure online RL (no LLM)** | `OnlineRLAgent` | **No** language model: a DQN-style policy over **primitive moves** from board encoding. Baseline / ablation vs LLM-guided play. Requires **PyTorch**. Code: `agent/online_rl_agent.py`. |

**Interaction summary**

1. **Game** (`InteractivePuzzle`, …) turns each WebSocket message into an **observation** (text board lines and/or image) and logs `last_sim_dict` for RL.
2. **Agent** `act(observation, step, game?)` returns a string like `move red cube left` (or `error` / special commands).
3. **Loop** sends that string to Unity as `GameInteraction` and waits for the next message.

`CognitiveAgent` with `rerank_mode: dqn` sets `act_needs_game_reference` so the loop passes **`game`** into `act` (for rewards / `board_data`).

---

## 3. How to use

### 3.1 Environment setup

1. **Conda (recommended for iVISPAR)**  
   From repo root:
   ```bash
   conda env create -f iVISPAR/Resources/environment.yml
   conda activate conda_env_iVISPAR
   pip install ollama
   ```
2. **Python path**  
   `init_experiment_components.py` adds the **repo root** to `sys.path` so `import agent` resolves to `./agent`.
3. **Ollama**  
   Install [Ollama](https://ollama.com), run the daemon, and `ollama pull <model>` (e.g. `llama3.1:8b`, `qwen3-vl` for vision).
4. **Cognitive DQN or Online RL**  
   From repo root:
   ```bash
   pip install -r requirements-rl.txt
   ```
   (PyTorch + deps for `rerank_mode: dqn` and `OnlineRLAgent`.)

### 3.2 API keys (only for API agents)

Create a file (not committed by default):

- Path: `iVISPAR/Data/API-keys/api-keys.txt`  
- Format: whatever `GPT4Agent` / `GeminiAgent` / … expect (see upstream iVISPAR and `params_experiment.json` examples).

Point to it in the params JSON with `"api_keys_file_path": "Data/API-keys/api-keys.txt"`.

### 3.3 Run an experiment

1. **Working directory** must be the experiment package:
   ```bash
   cd iVISPAR/Source/Experiment
   ```
2. **Command-line** (`run_experiment.py`):
   ```bash
   python run_experiment.py --params params_experiment_OllamaVision.json
   ```
   `--params` is a **filename only**; the file is loaded from **`iVISPAR/Data/Params/`** (see `experiment_utilities.load_params_from_json`).

   **Useful presets (same folder):**

   | File | Purpose |
   |------|---------|
   | `params_experiment_OllamaVision.json` | **Ollama** baseline (`OllamaAgent`) |
   | `params_experiment_CognitiveAgent.json` | Cognitive + LLM critique rerank |
   | `params_experiment_CognitiveAgentDQN.json` | Cognitive + **DQN** rerank |
   | `params_experiment_OnlineRLAgent.json` | **RL-only** agent |
   | `params_experiment.json` | Example **GPT4** / API-style entries |

3. **Browser + Unity WebGL**  
   Scripts start the local WebSocket server and WebGL host. Open the app (often `http://localhost:8000`). When the terminal asks for **remote client id**, paste the ID from the WebGL UI (**Copy ID**). Keep the tab open until the run finishes.

4. **Outputs**  
   Under `iVISPAR/Data/Experiments/<experiment_id>/` (episodes, logs, obs). This path is gitignored except what you choose to track elsewhere.

### 3.4 What to edit — config files

| Goal | Where to change |
|------|------------------|
| **Which agent / model / rerank** | `iVISPAR/Data/Params/<your_params>.json` → `agents` block: `agent_type`, `model_type`, `ollama_base_url`, `n_candidates`, `rerank_mode`, `rl_*` / `rl_selector_*` fields. |
| **Puzzle dataset & episode count** | Same file → `games` → `config_id` (must match a folder under `iVISPAR/Data/Configs/`), `num_game_env` (how many `config*.json` episodes), `max_game_length`, `representation_type` (`text`, `vision`, `schematic`). |
| **Task instructions shown to the LLM** | `instruction_prompt_file_path` in the **agent** block (paths relative to `iVISPAR/`, e.g. `Data/Instructions/ICML_instruction.txt`). Game-specific copy in **game** block: `instruction_prompt_file_path` under `games`. |
| **Camera / screenshot behaviour** | Same params file → `envs` → `sim_param1` (or other env keys): `grid_label`, `camera_offset`, `auto_done_check`, etc. |
| **Fixed experiment folder name** | Set `"experiment_id": "my_run_001"` in the params JSON; default `null` uses a timestamp. |
| **Register a new agent class** | Implement in `iVISPAR/Source/Experiment/agent_systems.py` (or import from `agent/`) and add a branch in `init_experiment_components.py` → `init_agent()`. |

**Config sets (puzzle instances):**  
`config_id` selects `iVISPAR/Data/Configs/<config_id>/config*.json`. Do not confuse with **params** in `Data/Params/`.

---

## 4. References

- Upstream iVISPAR: [SharkyBamboozle/iVISPAR](https://github.com/SharkyBamboozle/iVISPAR)  
- More detail on the legacy experiment UI flow: `iVISPAR/Resources/HowTo/how_to_run_experiments.md`  
- If you previously used a nested Git clone inside `iVISPAR/`, a backup of `.git` may exist as `iVISPAR/.git_backup_ivispar_upstream` (ignored).
