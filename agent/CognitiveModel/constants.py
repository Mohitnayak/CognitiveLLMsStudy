"""Symbolic grid rearrangement MDP — shared constants."""

from __future__ import annotations

from typing import Tuple

EMPTY = -1
DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES: Tuple[str, ...] = ("up", "down", "left", "right")
