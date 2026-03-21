# CognitiveModel

Python package factored out from **`gridworld_rl_deep_agents.ipynb`**: a **symbolic grid rearrangement MDP** and **deep RL** baselines (**DQN**, **Double DQN**, **PPO**).

## Concepts (same as the notebook)

| Concept | Module(s) |
|--------|-----------|
| MDP env (deterministic moves, goal-conditioned) | `environment.py` |
| Observations = one-hot(state) ‖ one-hot(goal) | `utils.encode_observation` |
| DQN / DDQN | `agents.DeepQAgent`, `training.train_dqn_agent` |
| PPO | `agents.PPOAgent`, `training.train_ppo_agent` |
| Replay buffer, MLP, actor–critic | `networks.py` |
| Rollouts, load/save, compare runs | `evaluation.py` |
| Matplotlib plots | `visualization.py` |

## Dependencies

```bash
pip install numpy matplotlib torch
```

## Import (from repo root `CogLLMVispar`)

```python
from agent.CognitiveModel import (
    EnvConfig,
    GridRearrangementEnv,
    DQNConfig,
    train_dqn_agent,
    generate_fixed_sample,
    rollout_agent,
)

cfg = EnvConfig(size=4, n_objects=10, seed=7)
env = GridRearrangementEnv(cfg)
env.reset(seed=123)
print(env.render_ascii())
```

## Relation to iVISPAR

This is a **standalone symbolic** puzzle for RL research. It does **not** drive the Unity WebSocket benchmark; for that, use `agent.OllamaAgent` + `iVISPAR` experiment params.
