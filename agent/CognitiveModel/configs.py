"""Training hyperparameters for DQN / Double DQN and PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DQNConfig:
    episodes: int = 1200
    buffer_size: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    hidden_dims: Tuple[int, ...] = (256, 256)
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 250
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    epsilon_decay_fraction: float = 0.6
    max_grad_norm: float = 10.0
    seed: int = 0
    save_every: int = 0


@dataclass
class PPOConfig:
    total_timesteps: int = 80_000
    rollout_steps: int = 1024
    epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: Tuple[int, ...] = (256, 256)
    seed: int = 0
    save_every_updates: int = 0
