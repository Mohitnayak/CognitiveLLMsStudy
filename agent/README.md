# `agent` — Ollama integration + cognitive / RL models

This folder holds **Ollama-specific** Python code for iVISPAR and optional **symbolic MDP + deep RL** code.

## `CognitiveModel/`

Same concepts as **`gridworld_rl_deep_agents.ipynb`**: `GridRearrangementEnv` (goal-conditioned MDP), DQN / Double DQN / PPO, training and evaluation helpers. See `CognitiveModel/README.md`.

## Cognitive agent (LLM proposes → rerank)

- **`cognitive_agent.py`** — `CognitiveAgent`: Ollama proposes **N** candidate `action: move ...` lines, then a second pass picks one (`rerank_mode`: **`critique`** default, or `first_valid`, `random`).
- Example params: `iVISPAR/Data/Params/params_experiment_CognitiveAgent.json` (`agent_type`: `CognitiveAgent`).
- **Compare** to `OllamaAgent` by keeping games/env/model settings identical and only swapping the agent block.

## Ollama

- **`ollama_agent.py`** — `OllamaAgent` (HTTP `/api/generate`, optional vision via `images`).
- **`__init__.py`** — re-exports `OllamaAgent`.

Shared LLM utilities (`LLMAgent`, parsing, image overlays) stay in  
`iVISPAR/Source/Experiment/agent_systems.py`; `OllamaAgent` subclasses that base class.

## Experiment parameters

Experiment JSON for Ollama runs remains under  
`iVISPAR/Data/Params/params_experiment_OllamaVision.json` (loaded by `experiment_utilities.load_params_from_json`).

## Running

Launch experiments from the repo layout that includes **both** `iVISPAR/` and `agent/` at the same parent (this repo root).  
`init_experiment_components.py` adds the repo root to `sys.path` so `import agent` resolves.
