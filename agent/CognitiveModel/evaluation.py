"""Rollouts, checkpoint I/O, and multi-algorithm training / comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .agents import AgentType, DeepQAgent, PPOAgent
from .configs import DQNConfig, PPOConfig
from .environment import EnvConfig, GridRearrangementEnv
from .training import train_dqn_agent, train_ppo_agent
from .utils import count_correct_objects, ensure_device


@dataclass
class RolloutSummary:
    algo: str
    total_reward: float
    steps: int
    solved: bool
    invalid_actions: int
    correct_objects: int
    final_state: Tuple[int, ...]
    actions: List[int] = field(default_factory=list)
    states: List[Tuple[int, ...]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)


def rollout_agent(
    agent: AgentType,
    env_config: EnvConfig,
    *,
    start_state: Sequence[int],
    goal_state: Sequence[int],
    deterministic: bool = True,
) -> RolloutSummary:
    env = GridRearrangementEnv(env_config)
    env.reset(start_state=start_state, goal=goal_state)

    total_reward = 0.0
    invalid_actions = 0
    states = [tuple(env.state)]
    actions: List[int] = []
    rewards: List[float] = []
    solved = False

    for _ in range(env_config.max_steps):
        obs = env.observation_features()
        mask = env.action_mask()
        if not mask.any():
            break

        if isinstance(agent, PPOAgent):
            action, _, _ = agent.act(obs, mask, deterministic=deterministic)
        else:
            action = agent.predict(obs, mask, deterministic=deterministic, epsilon=0.0)
        result = env.step(action)
        actions.append(action)
        rewards.append(result.reward)
        states.append(tuple(env.state))
        total_reward += result.reward
        invalid_actions += int(result.info["invalid_action"])
        solved = bool(result.info["solved"])
        if result.done:
            break

    return RolloutSummary(
        algo=agent.algo_name,
        total_reward=float(total_reward),
        steps=len(actions),
        solved=solved,
        invalid_actions=invalid_actions,
        correct_objects=count_correct_objects(env.state, env.goal, env.n_objects),
        final_state=tuple(env.state),
        actions=actions,
        states=states,
        rewards=rewards,
    )


def save_json_summary(summary_rows: Sequence[Dict[str, object]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(summary_rows), indent=2))


def comparison_rows(results: Dict[str, RolloutSummary]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for algo, result in results.items():
        rows.append(
            {
                "algorithm": algo,
                "solved": result.solved,
                "steps": result.steps,
                "total_reward": round(result.total_reward, 3),
                "invalid_actions": result.invalid_actions,
                "correct_objects": result.correct_objects,
            }
        )
    return rows


def print_comparison_table(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print("No results.")
        return
    headers = ["algorithm", "solved", "steps", "total_reward", "invalid_actions", "correct_objects"]
    widths = {h: max(len(h), max(len(str(row[h])) for row in rows)) for h in headers}
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def load_agent(path: str | Path, device: Optional[str] = None) -> AgentType:
    payload = torch.load(Path(path), map_location=ensure_device(device), weights_only=False)
    algo = payload["algo"]
    if algo in {"dqn", "ddqn"}:
        return DeepQAgent.load(path, device=device)
    if algo == "ppo":
        return PPOAgent.load(path, device=device)
    raise ValueError(f"Unsupported algo in checkpoint: {algo}")


def train_selected_algorithms(
    algorithms: Sequence[str],
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    ppo_config: PPOConfig,
    model_dir: str | Path,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, AgentType], Dict[str, Dict[str, List[float]]], Dict[str, Path]]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    agents: Dict[str, AgentType] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}
    paths: Dict[str, Path] = {}

    for algo in algorithms:
        algo_key = algo.lower()
        save_path = model_dir / f"{algo_key}_grid{env_config.size}_obj{env_config.n_objects}.pt"
        paths[algo_key] = save_path
        if algo_key == "dqn":
            agent, history = train_dqn_agent(
                env_config, dqn_config, algo_name="dqn", device=device, save_path=save_path, verbose=verbose
            )
        elif algo_key == "ddqn":
            agent, history = train_dqn_agent(
                env_config, dqn_config, algo_name="ddqn", device=device, save_path=save_path, verbose=verbose
            )
        elif algo_key == "ppo":
            agent, history = train_ppo_agent(env_config, ppo_config, device=device, save_path=save_path, verbose=verbose)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        agents[algo_key] = agent
        histories[algo_key] = history
    return agents, histories, paths


def evaluate_saved_models(
    model_paths: Dict[str, str | Path],
    env_config: EnvConfig,
    *,
    start_state: Sequence[int],
    goal_state: Sequence[int],
    device: Optional[str] = None,
) -> Dict[str, RolloutSummary]:
    results: Dict[str, RolloutSummary] = {}
    for algo, path in model_paths.items():
        agent = load_agent(path, device=device)
        results[algo] = rollout_agent(agent, env_config, start_state=start_state, goal_state=goal_state)
    return results


def generate_fixed_sample(env_config: EnvConfig, seed: int = 123) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    env = GridRearrangementEnv(env_config)
    env.reset(seed=seed)
    return tuple(env.state), tuple(env.goal)
