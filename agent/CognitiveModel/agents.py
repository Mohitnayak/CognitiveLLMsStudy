"""Deep Q-learning and PPO agent wrappers."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .configs import DQNConfig, PPOConfig
from .environment import EnvConfig
from .networks import ActorCriticNet, MLP
from .utils import ensure_device, masked_argmax


class DeepQAgent:
    def __init__(
        self, env_config: EnvConfig, train_config: DQNConfig, algo_name: str = "dqn", device: Optional[str] = None
    ):
        if algo_name not in {"dqn", "ddqn"}:
            raise ValueError("algo_name must be 'dqn' or 'ddqn'")
        self.algo_name = algo_name
        self.env_config = env_config
        self.train_config = train_config
        self.device = ensure_device(device)
        self.obs_dim = 2 * env_config.size * env_config.size * (env_config.n_objects + 1)
        self.action_dim = env_config.n_objects * 4

        self.online_net = MLP(self.obs_dim, self.action_dim, train_config.hidden_dims).to(self.device)
        self.target_net = MLP(self.obs_dim, self.action_dim, train_config.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=train_config.lr)

    def predict(self, obs_features: np.ndarray, action_mask: np.ndarray, deterministic: bool = True, epsilon: float = 0.0) -> int:
        legal_actions = np.flatnonzero(action_mask)
        if legal_actions.size == 0:
            return 0
        if (not deterministic) and (random.random() < epsilon):
            return int(random.choice(legal_actions.tolist()))
        obs_t = torch.as_tensor(obs_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
            action = masked_argmax(q_values, mask_t).item()
        return int(action)

    def save(self, path: str | Path) -> None:
        payload = {
            "algo": self.algo_name,
            "env_config": asdict(self.env_config),
            "train_config": asdict(self.train_config),
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "DeepQAgent":
        payload = torch.load(Path(path), map_location=ensure_device(device), weights_only=False)
        env_config = EnvConfig(**payload["env_config"])
        train_config = DQNConfig(**payload["train_config"])
        agent = cls(env_config, train_config, algo_name=payload["algo"], device=device)
        agent.online_net.load_state_dict(payload["online_state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        return agent


class PPOAgent:
    def __init__(self, env_config: EnvConfig, train_config: PPOConfig, device: Optional[str] = None):
        self.algo_name = "ppo"
        self.env_config = env_config
        self.train_config = train_config
        self.device = ensure_device(device)
        self.obs_dim = 2 * env_config.size * env_config.size * (env_config.n_objects + 1)
        self.action_dim = env_config.n_objects * 4

        self.policy = ActorCriticNet(self.obs_dim, self.action_dim, train_config.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=train_config.lr)

    def _distribution(self, obs_t: torch.Tensor, action_mask_t: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self.policy(obs_t)
        masked_logits = logits.masked_fill(~action_mask_t, -1e9)
        return Categorical(logits=masked_logits), value

    def act(self, obs_features: np.ndarray, action_mask: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self._distribution(obs_t, mask_t)
            if deterministic:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def save(self, path: str | Path) -> None:
        payload = {
            "algo": self.algo_name,
            "env_config": asdict(self.env_config),
            "train_config": asdict(self.train_config),
            "policy_state_dict": self.policy.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "PPOAgent":
        payload = torch.load(Path(path), map_location=ensure_device(device), weights_only=False)
        env_config = EnvConfig(**payload["env_config"])
        train_config = PPOConfig(**payload["train_config"])
        agent = cls(env_config, train_config, device=device)
        agent.policy.load_state_dict(payload["policy_state_dict"])
        return agent


AgentType = DeepQAgent | PPOAgent
