"""
Online DQN agent for iVISPAR text-mode InteractivePuzzle.

Uses `board_data` from Unity JSON (`game.last_sim_dict`). Legal actions are a
masked subset: one grid step into an empty cell (MVP; may not match slide physics).

Requires PyTorch. A single shared policy is kept across episodes in one process
so training continues across benchmark configs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from agent.CognitiveModel.networks import MLP, ReplayBuffer
from agent.rl_board_utils import action_invalid as _action_invalid
from agent.rl_board_utils import count_correct as _count_correct
from agent.rl_board_utils import encode_board_obs as _encode_obs
from agent.rl_board_utils import sorted_board as _sorted_board

DIR_NAMES = ("left", "right", "up", "down")
# Deltas on (col, row) = current_coordinate indices [0], [1]
DIR_DELTAS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def _legal_mask(board_data: List[dict], max_objects: int, grid_size: int) -> np.ndarray:
    """
    Boolean mask of shape (max_objects * 4,). True if moving object i in direction d
    is one step into an in-bounds empty cell.
    """
    mask = np.zeros((max_objects * 4,), dtype=np.bool_)
    sorted_bd = _sorted_board(board_data)
    n = len(sorted_bd)
    if n == 0:
        return mask

    occ = {}
    for idx, o in enumerate(sorted_bd):
        c = o.get("current_coordinate") or [0, 0]
        col, row = int(round(float(c[0]))), int(round(float(c[1])))
        occ[(col, row)] = idx

    for i, o in enumerate(sorted_bd):
        if i >= max_objects:
            break
        c = o.get("current_coordinate") or [0, 0]
        col, row = int(round(float(c[0]))), int(round(float(c[1])))
        for d, (dc, dr) in enumerate(DIR_DELTAS):
            ncol, nrow = col + dc, row + dr
            if ncol < 0 or nrow < 0 or ncol >= grid_size or nrow >= grid_size:
                continue
            if (ncol, nrow) in occ:
                continue
            mask[i * 4 + d] = True
    return mask


def _decode_command(action_idx: int, board_data: List[dict], max_objects: int) -> str:
    sorted_bd = _sorted_board(board_data)
    obj_i = action_idx // 4
    d = action_idx % 4
    if obj_i < 0 or obj_i >= len(sorted_bd) or obj_i >= max_objects:
        return "error"
    o = sorted_bd[obj_i]
    return f"move {o['color']} {o['body']} {DIR_NAMES[d]}"


class OnlineRLBrain:
    """Shared Q-network + replay; one instance per Python process (benchmark run)."""

    _instance: Optional["OnlineRLBrain"] = None

    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp
        self.max_objects = int(hp.get("rl_max_objects", 16))
        self.grid_size = int(hp.get("rl_grid_size", 4))
        self.obs_dim = self.max_objects * 5
        self.n_actions = self.max_objects * 4
        hidden = tuple(hp.get("rl_hidden_dims", [256, 256]))
        dev = hp.get("rl_device", None)
        self.device = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q = MLP(self.obs_dim, self.n_actions, hidden_dims=hidden).to(self.device)
        self.target = MLP(self.obs_dim, self.n_actions, hidden_dims=hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=float(hp.get("rl_lr", 1e-3)))

        cap = int(hp.get("rl_buffer_size", 50_000))
        self.buffer = ReplayBuffer(cap, self.obs_dim, self.n_actions)
        self.global_step = 0
        self.grad_steps = 0

        ckpt = hp.get("rl_checkpoint_path")
        if ckpt:
            ckpt = os.path.abspath(ckpt)
            if os.path.isfile(ckpt):
                self._load_checkpoint(ckpt)

    @classmethod
    def get(cls, hp: Dict[str, Any]) -> "OnlineRLBrain":
        if cls._instance is None:
            cls._instance = cls(hp)
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def _load_checkpoint(self, path: str) -> None:
        try:
            blob = torch.load(path, map_location=self.device)
            self.q.load_state_dict(blob["q"])
            self.target.load_state_dict(blob["target"])
            self.optim.load_state_dict(blob["optim"])
            self.global_step = int(blob.get("global_step", 0))
            self.grad_steps = int(blob.get("grad_steps", 0))
            logging.info("OnlineRLBrain loaded checkpoint %s", path)
        except Exception as e:
            logging.warning("OnlineRLBrain failed to load %s: %s", path, e)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "global_step": self.global_step,
            "grad_steps": self.grad_steps,
        }
        torch.save(blob, path)
        logging.info("OnlineRLBrain saved checkpoint %s", path)

    def epsilon(self) -> float:
        t = min(self.global_step, int(self.hp.get("rl_epsilon_decay_steps", 20_000)))
        lo = float(self.hp.get("rl_epsilon_end", 0.05))
        hi = float(self.hp.get("rl_epsilon_start", 1.0))
        decay = int(self.hp.get("rl_epsilon_decay_steps", 20_000))
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
            qv = qv.squeeze(0).cpu().numpy()
            qv = qv.copy()
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

        starts = int(self.hp.get("rl_learning_starts", 256))
        every = int(self.hp.get("rl_train_every", 4))
        batch = int(self.hp.get("rl_batch_size", 128))
        gamma = float(self.hp.get("rl_gamma", 0.99))

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

        tau = int(self.hp.get("rl_target_update_every", 500))
        if self.grad_steps % tau == 0:
            self.target.load_state_dict(self.q.state_dict())


class OnlineRLAgent:
    act_needs_game_reference = True

    def __init__(
        self,
        episode_path: str,
        episode_logger: Optional[logging.Logger],
        agent_params: Dict[str, Any],
        **_unused: Any,
    ) -> None:
        self.episode_path = episode_path
        self.episode_logger = episode_logger
        self.delay = float(agent_params.get("delay", 0.0))
        self._params = dict(agent_params)
        self.brain = OnlineRLBrain.get(agent_params)

        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_correct: int = 0
        self._finalize_done: bool = False

    def act(self, observation: Any, loop_iteration: int, game: Any = None) -> str:
        if game is None:
            logging.error("OnlineRLAgent requires `game` reference.")
            return "error"

        if getattr(game, "representation_type", None) != "text":
            logging.error("OnlineRLAgent needs representation_type=='text' for board_data.")
            return "error"

        if loop_iteration == 0:
            self._last_obs = None
            self._last_action = None
            self._last_correct = 0
            self._finalize_done = False
            return "start"

        sim = getattr(game, "last_sim_dict", None)
        if not isinstance(sim, dict):
            logging.error("OnlineRLAgent: missing last_sim_dict.")
            return "error"

        bd = sim.get("board_data") or []
        hp = self._params
        max_o = self.brain.max_objects
        gsz = self.brain.grid_size

        obs = _encode_obs(bd, max_o, gsz)
        legal = _legal_mask(bd, max_o, gsz)
        cur_correct = _count_correct(bd)

        if loop_iteration >= 2 and self._last_obs is not None and self._last_action is not None:
            r = float(hp.get("rl_step_penalty", -0.02))
            r += float(hp.get("rl_shaping_scale", 0.5)) * (cur_correct - self._last_correct)
            if _action_invalid(sim):
                r += float(hp.get("rl_illegal_penalty", -0.5))
            if sim.get("game_done"):
                r += float(hp.get("rl_win_bonus", 10.0))

            done = bool(sim.get("game_done"))
            self.brain.push(
                self._last_obs,
                self._last_action,
                r,
                obs,
                done,
                legal,
            )

        action = self.brain.select_action(obs, legal)
        if not legal[action]:
            # Fallback: uniform over legal (e.g. Q ties / bad mask)
            nz = np.flatnonzero(legal)
            if nz.size:
                action = int(np.random.choice(nz))

        cmd = _decode_command(action, bd, max_o)
        self._last_obs = obs
        self._last_action = action
        self._last_correct = cur_correct

        if self.episode_logger:
            self.episode_logger.info(
                "OnlineRL step i=%s eps=%.3f action=%s cmd=%s correct=%s/%s",
                loop_iteration,
                self.brain.epsilon(),
                action,
                cmd,
                cur_correct,
                len(bd),
            )
        return cmd

    def finalize_episode_rl(self, message_data: dict, game: Any) -> None:
        """Append the last transition when the loop exits without another act() call."""
        if self._finalize_done:
            return
        sim = getattr(game, "last_sim_dict", None)
        if not isinstance(sim, dict):
            return
        if self._last_obs is None or self._last_action is None:
            return

        bd = sim.get("board_data") or []
        max_o = self.brain.max_objects
        gsz = self.brain.grid_size
        next_obs = _encode_obs(bd, max_o, gsz)
        next_legal = _legal_mask(bd, max_o, gsz)
        cur_correct = _count_correct(bd)
        hp = self._params

        r = float(hp.get("rl_step_penalty", -0.02))
        r += float(hp.get("rl_shaping_scale", 0.5)) * (cur_correct - self._last_correct)
        if _action_invalid(sim):
            r += float(hp.get("rl_illegal_penalty", -0.5))
        if sim.get("game_done"):
            r += float(hp.get("rl_win_bonus", 10.0))

        mx = getattr(game, "max_game_length", None)
        it = getattr(game, "iteration", 0)
        forced = mx is not None and it >= mx
        done = bool(sim.get("game_done")) or forced
        self.brain.push(
            self._last_obs,
            self._last_action,
            r,
            next_obs,
            done,
            next_legal,
        )
        self._finalize_done = True

        ckpt = hp.get("rl_checkpoint_path")
        if ckpt:
            path = os.path.abspath(str(ckpt))
            try:
                self.brain.save_checkpoint(path)
            except Exception as e:
                logging.error("OnlineRL checkpoint save failed: %s", e)
