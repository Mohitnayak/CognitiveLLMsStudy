"""Matplotlib helpers for states, training curves, and rollouts."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .constants import EMPTY
from .utils import moving_average


def state_to_matrix(state: Sequence[int], size: int) -> np.ndarray:
    return np.asarray(state, dtype=np.int64).reshape(size, size)


def plot_state_and_goal(
    state: Sequence[int],
    goal: Sequence[int],
    size: int,
    n_objects: int,
    title: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, board, label in zip(axes, [state, goal], ["State", "Goal"]):
        matrix = state_to_matrix(board, size)
        image = matrix.copy().astype(float)
        image[image == EMPTY] = np.nan
        ax.imshow(np.where(np.isnan(image), -1, image), vmin=-1, vmax=n_objects - 1)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_title(label)
        ax.grid(True)
        for r in range(size):
            for c in range(size):
                value = matrix[r, c]
                text = "." if value == EMPTY else str(value)
                ax.text(c, r, text, ha="center", va="center", fontsize=10)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict[str, List[float]], title: str = "Training history") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    returns = history.get("episode_returns", [])
    solved = history.get("episode_solved", [])
    lengths = history.get("episode_lengths", [])

    axes[0].plot(returns, alpha=0.5, label="return")
    if returns:
        axes[0].plot(moving_average(returns, 50), label="50-ep avg")
    axes[0].set_title("Episode return")
    axes[0].legend()

    if solved:
        solved_float = [float(x) for x in solved]
        axes[1].plot(moving_average(solved_float, 50))
    axes[1].set_title("Solve rate (50-ep avg)")

    axes[2].plot(lengths, alpha=0.5)
    if lengths:
        axes[2].plot(moving_average(lengths, 50))
    axes[2].set_title("Episode length")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_rollout_states(
    states: Sequence[Sequence[int]], size: int, max_panels: int = 8, title: str = "Rollout"
) -> None:
    if not states:
        return
    indices = np.linspace(0, len(states) - 1, num=min(max_panels, len(states)), dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(2.2 * len(indices), 2.5))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        matrix = state_to_matrix(states[idx], size)
        image = matrix.copy().astype(float)
        image[image == EMPTY] = np.nan
        ax.imshow(np.where(np.isnan(image), -1, image))
        ax.set_title(f"t={idx}")
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True)
        for r in range(size):
            for c in range(size):
                value = matrix[r, c]
                text = "." if value == EMPTY else str(value)
                ax.text(c, r, text, ha="center", va="center", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
