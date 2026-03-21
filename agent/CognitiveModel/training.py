"""Training loops for DQN, Double DQN, and PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agents import DeepQAgent, PPOAgent
from .configs import DQNConfig, PPOConfig
from .environment import EnvConfig, GridRearrangementEnv
from .networks import ReplayBuffer
from .utils import masked_argmax, seed_everything

torch.set_num_threads(1)


def _epsilon_by_step(config: DQNConfig, current_step: int, total_steps: int) -> float:
    decay_steps = max(1, int(config.epsilon_decay_fraction * total_steps))
    frac = min(1.0, current_step / decay_steps)
    return config.start_epsilon + frac * (config.end_epsilon - config.start_epsilon)


def _compute_dqn_targets(
    batch: Dict[str, torch.Tensor],
    online_net: nn.Module,
    target_net: nn.Module,
    gamma: float,
    algo_name: str,
) -> torch.Tensor:
    rewards = batch["rewards"]
    dones = batch["dones"]
    next_obs = batch["next_obs"]
    next_masks = batch["next_masks"]

    with torch.no_grad():
        target_q = target_net(next_obs)
        target_q_masked = target_q.masked_fill(~next_masks, -1e9)
        any_legal = next_masks.any(dim=1)

        if algo_name == "ddqn":
            online_q = online_net(next_obs)
            next_actions = masked_argmax(online_q, next_masks)
            next_values = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_values = target_q_masked.max(dim=1).values

        next_values = torch.where(any_legal, next_values, torch.zeros_like(next_values))
        targets = rewards + gamma * (1.0 - dones) * next_values
    return targets


def train_dqn_agent(
    env_config: EnvConfig,
    train_config: DQNConfig,
    *,
    algo_name: str = "dqn",
    device: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[DeepQAgent, Dict[str, List[float]]]:
    if algo_name not in {"dqn", "ddqn"}:
        raise ValueError("algo_name must be 'dqn' or 'ddqn'")

    seed_everything(train_config.seed)
    env = GridRearrangementEnv(env_config)
    agent = DeepQAgent(env_config, train_config, algo_name=algo_name, device=device)
    buffer = ReplayBuffer(train_config.buffer_size, env.obs_dim, env.num_actions)
    history: Dict[str, List[float]] = {
        "episode_returns": [],
        "episode_lengths": [],
        "episode_solved": [],
        "losses": [],
    }

    total_steps_budget = train_config.episodes * env_config.max_steps
    total_steps = 0

    for episode in range(train_config.episodes):
        env.reset(seed=train_config.seed + episode)
        obs = env.observation_features()
        done = False
        episode_return = 0.0
        episode_length = 0
        solved = False

        while not done:
            epsilon = _epsilon_by_step(train_config, total_steps, total_steps_budget)
            mask = env.action_mask()
            action = agent.predict(obs, mask, deterministic=False, epsilon=epsilon)
            result = env.step(action)
            next_obs = env.observation_features()
            next_mask = env.action_mask()
            done = result.done

            buffer.add(obs, action, result.reward, next_obs, done, next_mask)
            obs = next_obs
            episode_return += result.reward
            episode_length += 1
            solved = bool(result.info["solved"])
            total_steps += 1

            if len(buffer) >= train_config.learning_starts and total_steps % train_config.train_freq == 0:
                for _ in range(train_config.gradient_steps):
                    batch = buffer.sample(train_config.batch_size, agent.device)
                    q_values = agent.online_net(batch["obs"])
                    q_selected = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                    targets = _compute_dqn_targets(
                        batch,
                        agent.online_net,
                        agent.target_net,
                        train_config.gamma,
                        algo_name,
                    )
                    loss = F.smooth_l1_loss(q_selected, targets)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.online_net.parameters(), train_config.max_grad_norm)
                    agent.optimizer.step()
                    history["losses"].append(float(loss.item()))

            if total_steps % train_config.target_update_interval == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())

        history["episode_returns"].append(float(episode_return))
        history["episode_lengths"].append(float(episode_length))
        history["episode_solved"].append(float(solved))

        if verbose and ((episode + 1) % max(1, train_config.episodes // 10) == 0 or episode == 0):
            recent_returns = history["episode_returns"][-50:]
            recent_solved = history["episode_solved"][-50:]
            print(
                f"[{algo_name.upper()}] episode {episode + 1}/{train_config.episodes} | "
                f"avg_return={np.mean(recent_returns):.3f} | solve_rate={np.mean(recent_solved):.2%} | "
                f"epsilon={epsilon:.3f}"
            )

        if save_path is not None and train_config.save_every > 0 and (episode + 1) % train_config.save_every == 0:
            agent.save(save_path)

    if save_path is not None:
        agent.save(save_path)
    return agent, history


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages, returns


def train_ppo_agent(
    env_config: EnvConfig,
    train_config: PPOConfig,
    *,
    device: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[PPOAgent, Dict[str, List[float]]]:
    seed_everything(train_config.seed)
    env = GridRearrangementEnv(env_config)
    agent = PPOAgent(env_config, train_config, device=device)

    history: Dict[str, List[float]] = {
        "episode_returns": [],
        "episode_lengths": [],
        "episode_solved": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
    }

    env.reset(seed=train_config.seed)
    obs = env.observation_features()
    episode_return = 0.0
    episode_length = 0
    episode_solved = False
    total_steps = 0
    update_idx = 0

    while total_steps < train_config.total_timesteps:
        obs_buf: List[np.ndarray] = []
        actions_buf: List[int] = []
        log_probs_buf: List[float] = []
        rewards_buf: List[float] = []
        dones_buf: List[float] = []
        values_buf: List[float] = []
        masks_buf: List[np.ndarray] = []

        for _ in range(train_config.rollout_steps):
            action_mask = env.action_mask()
            action, log_prob, value = agent.act(obs, action_mask, deterministic=False)
            result = env.step(action)
            next_obs = env.observation_features()

            obs_buf.append(obs.copy())
            actions_buf.append(action)
            log_probs_buf.append(log_prob)
            rewards_buf.append(result.reward)
            dones_buf.append(float(result.done))
            values_buf.append(value)
            masks_buf.append(action_mask.copy())

            obs = next_obs
            episode_return += result.reward
            episode_length += 1
            episode_solved = episode_solved or bool(result.info["solved"])
            total_steps += 1

            if result.done:
                history["episode_returns"].append(float(episode_return))
                history["episode_lengths"].append(float(episode_length))
                history["episode_solved"].append(float(episode_solved))
                env.reset(seed=train_config.seed + total_steps)
                obs = env.observation_features()
                episode_return = 0.0
                episode_length = 0
                episode_solved = False

            if total_steps >= train_config.total_timesteps:
                break

        last_mask = env.action_mask()
        with torch.no_grad():
            _, last_value = agent._distribution(
                torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0),
                torch.as_tensor(last_mask, dtype=torch.bool, device=agent.device).unsqueeze(0),
            )
        advantages, returns = _compute_gae(
            np.asarray(rewards_buf, dtype=np.float32),
            np.asarray(values_buf, dtype=np.float32),
            np.asarray(dones_buf, dtype=np.float32),
            float(last_value.item()),
            train_config.gamma,
            train_config.gae_lambda,
        )
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=agent.device)
        actions_tensor = torch.as_tensor(np.asarray(actions_buf), dtype=torch.int64, device=agent.device)
        old_log_probs_tensor = torch.as_tensor(np.asarray(log_probs_buf), dtype=torch.float32, device=agent.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)
        masks_tensor = torch.as_tensor(np.asarray(masks_buf), dtype=torch.bool, device=agent.device)

        rollout_size = obs_tensor.shape[0]
        minibatch_size = min(train_config.minibatch_size, rollout_size)

        for _ in range(train_config.epochs):
            indices = np.random.permutation(rollout_size)
            for start in range(0, rollout_size, minibatch_size):
                batch_idx = indices[start : start + minibatch_size]
                dist, value = agent._distribution(obs_tensor[batch_idx], masks_tensor[batch_idx])
                new_log_prob = dist.log_prob(actions_tensor[batch_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_prob - old_log_probs_tensor[batch_idx])

                unclipped = ratio * advantages_tensor[batch_idx]
                clipped = (
                    torch.clamp(ratio, 1.0 - train_config.clip_range, 1.0 + train_config.clip_range)
                    * advantages_tensor[batch_idx]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns_tensor[batch_idx])
                loss = policy_loss + train_config.value_coef * value_loss - train_config.entropy_coef * entropy

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.parameters(), train_config.max_grad_norm)
                agent.optimizer.step()

                history["policy_losses"].append(float(policy_loss.item()))
                history["value_losses"].append(float(value_loss.item()))
                history["entropies"].append(float(entropy.item()))

        update_idx += 1
        if verbose and update_idx % max(1, (train_config.total_timesteps // train_config.rollout_steps) // 10) == 0:
            recent_returns = history["episode_returns"][-50:] or [0.0]
            recent_solved = history["episode_solved"][-50:] or [0.0]
            print(
                f"[PPO] update {update_idx} | total_steps={total_steps}/{train_config.total_timesteps} | "
                f"avg_return={np.mean(recent_returns):.3f} | solve_rate={np.mean(recent_solved):.2%}"
            )

        if save_path is not None and train_config.save_every_updates > 0 and update_idx % train_config.save_every_updates == 0:
            agent.save(save_path)

    if save_path is not None:
        agent.save(save_path)
    return agent, history
