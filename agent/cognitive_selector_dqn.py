"""
DQN over LLM candidate indices Q(s, k) with Bellman TD + replay + target net.

State s is encoded from Unity ``board_data`` only; action k is which proposed
``action: move ...`` line to execute (fixed K = n_candidates, padded mask).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agent.CognitiveModel.networks import MLP, ReplayBuffer


class CognitiveSelectorBrain:
    """Singleton per process; K must match first agent that creates it."""

    _instance: Optional["CognitiveSelectorBrain"] = None
    _key: Optional[tuple] = None

    def __init__(self, n_candidates: int, hp: Dict[str, Any]):
        self.hp = hp
        self.K = int(n_candidates)
        self.max_objects = int(hp.get("rl_selector_max_objects", hp.get("rl_max_objects", 16)))
        self.grid_size = int(hp.get("rl_selector_grid_size", hp.get("rl_grid_size", 4)))
        self.obs_dim = self.max_objects * 5
        self.n_actions = self.K
        hidden = tuple(hp.get("rl_selector_hidden_dims", hp.get("rl_hidden_dims", [256, 256])))
        dev = hp.get("rl_selector_device", hp.get("rl_device", None))
        self.device = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q = MLP(self.obs_dim, self.n_actions, hidden_dims=hidden).to(self.device)
        self.target = MLP(self.obs_dim, self.n_actions, hidden_dims=hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(
            self.q.parameters(), lr=float(hp.get("rl_selector_lr", hp.get("rl_lr", 1e-3)))
        )

        cap = int(hp.get("rl_selector_buffer_size", hp.get("rl_buffer_size", 50_000)))
        self.buffer = ReplayBuffer(cap, self.obs_dim, self.n_actions)
        self.global_step = 0
        self.grad_steps = 0

        ckpt = hp.get("rl_selector_checkpoint_path")
        if ckpt:
            path = os.path.abspath(str(ckpt))
            if os.path.isfile(path):
                self._load_checkpoint(path)

    @classmethod
    def get(cls, n_candidates: int, hp: Dict[str, Any]) -> "CognitiveSelectorBrain":
        key = (int(n_candidates), int(hp.get("rl_selector_max_objects", hp.get("rl_max_objects", 16))))
        if cls._instance is None:
            cls._instance = cls(n_candidates, hp)
            cls._key = key
        elif cls._key != key:
            logging.warning(
                "CognitiveSelectorBrain already built with K=%s; ignoring new K=%s (reuse singleton).",
                cls._key,
                key,
            )
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
        cls._key = None

    def _load_checkpoint(self, path: str) -> None:
        try:
            blob = torch.load(path, map_location=self.device)
            self.q.load_state_dict(blob["q"])
            self.target.load_state_dict(blob["target"])
            self.optim.load_state_dict(blob["optim"])
            self.global_step = int(blob.get("global_step", 0))
            self.grad_steps = int(blob.get("grad_steps", 0))
            logging.info("CognitiveSelectorBrain loaded %s", path)
        except Exception as e:
            logging.warning("CognitiveSelectorBrain load failed %s: %s", path, e)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "target": self.target.state_dict(),
                "optim": self.optim.state_dict(),
                "global_step": self.global_step,
                "grad_steps": self.grad_steps,
            },
            path,
        )
        logging.info("CognitiveSelectorBrain saved %s", path)

    def epsilon(self) -> float:
        h = self.hp
        decay = int(h.get("rl_selector_epsilon_decay_steps", h.get("rl_epsilon_decay_steps", 20_000)))
        t = min(self.global_step, decay)
        lo = float(h.get("rl_selector_epsilon_end", h.get("rl_epsilon_end", 0.05)))
        hi = float(h.get("rl_selector_epsilon_start", h.get("rl_epsilon_start", 1.0)))
        if decay <= 0:
            return lo
        return hi + (lo - hi) * (t / decay)

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        legal_idx = np.flatnonzero(legal_mask)
        eps = self.epsilon()
        if legal_idx.size == 0:
            return 0
        if np.random.random() < eps:
            return int(np.random.choice(legal_idx))
        with torch.no_grad():
            qv = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            qv = qv.squeeze(0).cpu().numpy().copy()
            qv[~legal_mask] = -1e9
            return int(np.argmax(qv))

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done, next_legal_mask)
        self.global_step += 1

        hp = self.hp
        starts = int(hp.get("rl_selector_learning_starts", hp.get("rl_learning_starts", 256)))
        every = int(hp.get("rl_selector_train_every", hp.get("rl_train_every", 4)))
        batch = int(hp.get("rl_selector_batch_size", hp.get("rl_batch_size", 128)))
        gamma = float(hp.get("rl_selector_gamma", hp.get("rl_gamma", 0.99)))

        if len(self.buffer) < starts or self.global_step % every != 0:
            return

        b = self.buffer.sample(batch, self.device)
        q_all = self.q(b["obs"])
        q_sa = q_all.gather(1, b["actions"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nq = self.target(b["next_obs"])
            nq = nq.masked_fill(~b["next_masks"], -1e9)
            max_nq = nq.max(dim=1)[0]
            target = b["rewards"] + gamma * (1.0 - b["dones"]) * max_nq

        loss = F.mse_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.grad_steps += 1

        tau = int(hp.get("rl_selector_target_update_every", hp.get("rl_target_update_every", 500)))
        if self.grad_steps % tau == 0:
            self.target.load_state_dict(self.q.state_dict())
