"""Shared board_data → vector / reward helpers for RL (no torch)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def sorted_board(board_data: List[dict]) -> List[dict]:
    return sorted(board_data, key=lambda o: (str(o.get("color", "")), str(o.get("body", ""))))


def count_correct(board_data: List[dict]) -> int:
    n = 0
    for o in board_data:
        c = o.get("current_coordinate") or [0, 0]
        g = o.get("goal_coordinate") or [0, 0]
        if abs(float(c[0]) - float(g[0])) < 0.05 and abs(float(c[1]) - float(g[1])) < 0.05:
            n += 1
    return n


def action_invalid(sim: Optional[dict]) -> bool:
    if not sim:
        return True
    acts = sim.get("Actions") or []
    if not acts:
        return False
    for v in acts[0].get("valididy") or []:
        if isinstance(v, str) and "not a legal" in v.lower():
            return True
    return False


def encode_board_obs(board_data: List[dict], max_objects: int, grid_size: int) -> np.ndarray:
    """Per slot [present, col, row, gcol, grow] normalized to ~[0,1]."""
    g = float(max(grid_size - 1, 1))
    sorted_bd = sorted_board(board_data)
    vec: List[float] = []
    for i in range(max_objects):
        if i < len(sorted_bd):
            o = sorted_bd[i]
            c = o.get("current_coordinate") or [0, 0]
            gg = o.get("goal_coordinate") or [0, 0]
            vec.extend(
                [
                    1.0,
                    float(c[0]) / g,
                    float(c[1]) / g,
                    float(gg[0]) / g,
                    float(gg[1]) / g,
                ]
            )
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.asarray(vec, dtype=np.float32)
