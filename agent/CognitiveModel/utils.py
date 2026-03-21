"""Utilities: seeds, grid indexing, observation encoding, torch helpers."""

from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import EMPTY


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_col_to_index(r: int, c: int, size: int) -> int:
    return r * size + c


def index_to_row_col(idx: int, size: int) -> Tuple[int, int]:
    return divmod(idx, size)


def count_correct_objects(state: Sequence[int], goal: Sequence[int], n_objects: int) -> int:
    correct = 0
    for obj_id in range(n_objects):
        if state.index(obj_id) == goal.index(obj_id):
            correct += 1
    return correct


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(values[lo : i + 1]))
    return out


def board_to_onehot(board: Sequence[int], n_objects: int) -> np.ndarray:
    board_arr = np.asarray(board, dtype=np.int64)
    n_cells = board_arr.shape[0]
    channels = n_objects + 1  # channel 0 is EMPTY, channels 1..N are objects 0..N-1
    out = np.zeros((n_cells, channels), dtype=np.float32)
    idx = np.where(board_arr == EMPTY, 0, board_arr + 1)
    out[np.arange(n_cells), idx] = 1.0
    return out.reshape(-1)


def encode_observation(state: Sequence[int], goal: Sequence[int], n_objects: int) -> np.ndarray:
    return np.concatenate([board_to_onehot(state, n_objects), board_to_onehot(goal, n_objects)]).astype(
        np.float32
    )


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = q_values.masked_fill(~action_mask, -1e9)
    return masked.argmax(dim=-1)


def ensure_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
