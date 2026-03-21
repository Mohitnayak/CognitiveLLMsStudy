# CogLLMVispar — Cognitive agent + iVISPAR integration

This repository combines:

- **`agent/`** — Cognitive LLM agent (Ollama), DQN candidate selector, parsing utilities, optional RL deps (`requirements-rl.txt`).
- **`iVISPAR/`** — Patched [iVISPAR](https://github.com/SharkyBamboozle/iVISPAR) experiment stack (nested upstream Git history was removed; this tree is tracked here).

## Layout

- Experiment **outputs** under `iVISPAR/Data/Experiments/` are **gitignored** (large logs); params and source remain tracked.
- API key placeholders under `iVISPAR/Data/API-keys/` are **gitignored**.

## Setup

1. Python env with iVISPAR + Ollama dependencies (see iVISPAR `Resources/environment.yml` if you use conda).
2. `pip install -r requirements-rl.txt` if using DQN selector (`rerank_mode: dqn`).
3. Run experiments from the iVISPAR experiment entrypoints as documented upstream.

## Upstream

Original iVISPAR: [SharkyBamboozle/iVISPAR](https://github.com/SharkyBamboozle/iVISPAR).  
A backup of the previous nested `.git` metadata is kept locally as `iVISPAR/.git_backup_ivispar_upstream` (ignored by Git).
