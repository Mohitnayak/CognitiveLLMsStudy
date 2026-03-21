"""Goal-conditioned symbolic grid rearrangement environment (MDP)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import DIRS, DIR_NAMES, EMPTY
from .utils import (
    count_correct_objects,
    encode_observation,
    index_to_row_col,
    row_col_to_index,
)


@dataclass
class EnvConfig:
    size: int = 4
    n_objects: int = 10
    move_mode: str = "step"
    scramble_steps: int = 12
    max_steps: int = 60
    step_penalty: float = -0.3
    invalid_penalty: float = -0.7
    goal_reward: float = 12.0
    shaping_scale: float = 0.4
    random_goal: bool = False
    seed: int = 0


@dataclass
class StepResult:
    observation: Dict[str, Tuple[int, ...]]
    reward: float
    done: bool
    info: Dict[str, object]


class GridRearrangementEnv:
    """Goal-conditioned symbolic rearrangement environment.

    State: positions of all labeled objects on a size x size grid with empty cells.
    Action: (object_id, direction) encoded into one discrete integer.
    Transition: deterministic under either step or slide dynamics.
    """

    def __init__(self, config: EnvConfig):
        if config.size < 2:
            raise ValueError("size must be at least 2")
        if config.n_objects < 1:
            raise ValueError("n_objects must be at least 1")
        if config.n_objects >= config.size * config.size:
            raise ValueError("n_objects must be smaller than number of cells")
        if config.move_mode not in {"step", "slide"}:
            raise ValueError("move_mode must be 'step' or 'slide'")

        self.config = config
        self.size = config.size
        self.n_cells = self.size * self.size
        self.n_objects = config.n_objects
        self.num_actions = self.n_objects * 4
        self.rng = random.Random(config.seed)

        self.goal: Tuple[int, ...] = self._canonical_goal()
        self.state: Tuple[int, ...] = self.goal
        self.steps_taken = 0

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_cells * (self.n_objects + 1)

    def _canonical_goal(self) -> Tuple[int, ...]:
        cells = [EMPTY] * self.n_cells
        for obj_id in range(self.n_objects):
            cells[obj_id] = obj_id
        return tuple(cells)

    def _validate_layout(self, layout: Sequence[int]) -> Tuple[int, ...]:
        if len(layout) != self.n_cells:
            raise ValueError(f"layout must have length {self.n_cells}")
        arr = list(layout)
        expected = sorted([EMPTY] * (self.n_cells - self.n_objects) + list(range(self.n_objects)))
        if sorted(arr) != expected:
            raise ValueError("layout must contain each object exactly once and fill remaining cells with EMPTY")
        return tuple(int(x) for x in arr)

    def _sample_random_layout(self) -> Tuple[int, ...]:
        cells = [EMPTY] * self.n_cells
        positions = self.rng.sample(range(self.n_cells), self.n_objects)
        for obj_id, pos in enumerate(positions):
            cells[pos] = obj_id
        return tuple(cells)

    def _action_to_parts(self, action: int) -> Tuple[int, int]:
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"invalid action index {action}")
        return action // 4, action % 4

    def action_to_str(self, action: int) -> str:
        obj_id, dir_idx = self._action_to_parts(action)
        return f"obj={obj_id} {DIR_NAMES[dir_idx]}"

    def _find_object(self, state: Sequence[int], obj_id: int) -> Optional[int]:
        try:
            return tuple(state).index(obj_id)
        except ValueError:
            return None

    def _apply_action_to_state(self, state: Sequence[int], action: int) -> Tuple[Tuple[int, ...], bool]:
        obj_id, dir_idx = self._action_to_parts(action)
        pos = self._find_object(state, obj_id)
        if pos is None:
            return tuple(state), False

        r, c = index_to_row_col(pos, self.size)
        dr, dc = DIRS[dir_idx]
        cells = list(state)

        if self.config.move_mode == "step":
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                return tuple(state), False
            npos = row_col_to_index(nr, nc, self.size)
            if cells[npos] != EMPTY:
                return tuple(state), False
            cells[pos], cells[npos] = cells[npos], cells[pos]
            return tuple(cells), True

        nr, nc = r + dr, c + dc
        last_empty_pos: Optional[int] = None
        while 0 <= nr < self.size and 0 <= nc < self.size:
            npos = row_col_to_index(nr, nc, self.size)
            if cells[npos] != EMPTY:
                break
            last_empty_pos = npos
            nr += dr
            nc += dc
        if last_empty_pos is None:
            return tuple(state), False
        cells[pos], cells[last_empty_pos] = cells[last_empty_pos], cells[pos]
        return tuple(cells), True

    def legal_actions(self, state: Optional[Sequence[int]] = None) -> List[int]:
        src = self.state if state is None else tuple(state)
        return [a for a in range(self.num_actions) if self._apply_action_to_state(src, a)[1]]

    def action_mask(self, state: Optional[Sequence[int]] = None) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        for action in self.legal_actions(state):
            mask[action] = True
        return mask

    def is_goal(self, state: Optional[Sequence[int]] = None) -> bool:
        src = self.state if state is None else tuple(state)
        return src == self.goal

    def observation(self) -> Dict[str, Tuple[int, ...]]:
        return {"state": self.state, "goal": self.goal}

    def observation_features(self) -> np.ndarray:
        return encode_observation(self.state, self.goal, self.n_objects)

    def reset(
        self,
        *,
        scramble_steps: Optional[int] = None,
        goal: Optional[Sequence[int]] = None,
        start_state: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        if seed is not None:
            self.rng.seed(seed)

        if goal is not None:
            self.goal = self._validate_layout(goal)
        elif self.config.random_goal:
            self.goal = self._sample_random_layout()
        else:
            self.goal = self._canonical_goal()

        self.steps_taken = 0

        if start_state is not None:
            self.state = self._validate_layout(start_state)
            return self.observation()

        self.state = self.goal
        steps = self.config.scramble_steps if scramble_steps is None else max(0, int(scramble_steps))
        for _ in range(steps):
            legal = self.legal_actions(self.state)
            if not legal:
                break
            action = self.rng.choice(legal)
            next_state, moved = self._apply_action_to_state(self.state, action)
            if moved:
                self.state = next_state
        return self.observation()

    def step(self, action: int) -> StepResult:
        prev_state = self.state
        prev_correct = count_correct_objects(prev_state, self.goal, self.n_objects)
        next_state, moved = self._apply_action_to_state(prev_state, action)

        reward = self.config.step_penalty
        invalid_action = not moved
        if invalid_action:
            reward += self.config.invalid_penalty
        else:
            self.state = next_state
            next_correct = count_correct_objects(self.state, self.goal, self.n_objects)
            reward += self.config.shaping_scale * (next_correct - prev_correct)

        self.steps_taken += 1
        solved = self.is_goal(self.state)
        timeout = self.steps_taken >= self.config.max_steps
        dead_end = (not solved) and (len(self.legal_actions(self.state)) == 0)
        done = solved or timeout or dead_end
        if solved:
            reward += self.config.goal_reward

        return StepResult(
            observation=self.observation(),
            reward=float(reward),
            done=done,
            info={
                "invalid_action": invalid_action,
                "solved": solved,
                "timeout": timeout,
                "dead_end": dead_end,
                "steps_taken": self.steps_taken,
                "correct_objects": count_correct_objects(self.state, self.goal, self.n_objects),
            },
        )

    def render_ascii(self, state: Optional[Sequence[int]] = None, goal: Optional[Sequence[int]] = None) -> str:
        src_state = self.state if state is None else tuple(state)
        src_goal = self.goal if goal is None else tuple(goal)

        def fmt(x: int) -> str:
            return "." if x == EMPTY else str(x)

        lines = ["State:"]
        for r in range(self.size):
            row = src_state[r * self.size : (r + 1) * self.size]
            lines.append("  " + " ".join(f"{fmt(x):>2}" for x in row))
        lines.append("Goal:")
        for r in range(self.size):
            row = src_goal[r * self.size : (r + 1) * self.size]
            lines.append("  " + " ".join(f"{fmt(x):>2}" for x in row))
        return "\n".join(lines)
