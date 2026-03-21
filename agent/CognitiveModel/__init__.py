"""
CognitiveModel — symbolic grid rearrangement MDP + deep RL (same concepts as `gridworld_rl_deep_agents.ipynb`).

Components:
  - **MDP**: `GridRearrangementEnv`, `EnvConfig`, goal-conditioned observations (one-hot state + goal).
  - **Algorithms**: DQN, Double DQN (`DeepQAgent`), PPO (`PPOAgent`).
  - **Training / eval**: `train_dqn_agent`, `train_ppo_agent`, `rollout_agent`, `train_selected_algorithms`, etc.

This package is independent of Unity/iVISPAR; use it for RL baselines or cognitive-model experiments.
"""

from .agents import AgentType, DeepQAgent, PPOAgent
from .configs import DQNConfig, PPOConfig
from .environment import EnvConfig, GridRearrangementEnv, StepResult
from .evaluation import (
    RolloutSummary,
    comparison_rows,
    evaluate_saved_models,
    generate_fixed_sample,
    load_agent,
    print_comparison_table,
    rollout_agent,
    save_json_summary,
    train_selected_algorithms,
)
from .training import train_dqn_agent, train_ppo_agent
from .utils import encode_observation, seed_everything
from .visualization import plot_rollout_states, plot_state_and_goal, plot_training_history

__all__ = [
    "AgentType",
    "DQNConfig",
    "DeepQAgent",
    "EnvConfig",
    "GridRearrangementEnv",
    "PPOAgent",
    "PPOConfig",
    "RolloutSummary",
    "StepResult",
    "comparison_rows",
    "encode_observation",
    "evaluate_saved_models",
    "generate_fixed_sample",
    "load_agent",
    "plot_rollout_states",
    "plot_state_and_goal",
    "plot_training_history",
    "print_comparison_table",
    "rollout_agent",
    "save_json_summary",
    "seed_everything",
    "train_dqn_agent",
    "train_ppo_agent",
    "train_selected_algorithms",
]
