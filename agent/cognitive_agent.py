"""
Cognitive agent — LLM proposes candidate moves; a cognitive reranking step selects one.

Flow (vision or text):
  1) First Ollama call: ask for N distinct candidate moves (same format as baseline).
  2) Rerank:
     - ``critique`` (default): second Ollama call with the same observation; model picks 1..N.
     - ``first_valid``: use the first candidate that parses (fallback: baseline single parse).
     - ``random``: random choice among parsed candidates (ablation).
     - ``dqn``: Bellman DQN (replay + target net) chooses candidate index from ``board_data``
       encoding; requires ``game`` in ``act()`` and ``representation_type`` with JSON ``board_data``
       (text mode recommended). Falls back to critique if ``board_data`` is missing.

Compare to ``OllamaAgent`` with matched params except ``agent_type`` and cognitive settings.
"""

from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from agent.cognitive_parse_utils import (
    extract_candidate_moves,
    is_valid_move_command,
    sanitize_chosen_move,
)
from agent.rl_board_utils import action_invalid, count_correct, encode_board_obs

from .ollama_agent import OllamaAgent


class CognitiveAgent(OllamaAgent):
    """
    Propose K moves with Ollama, then rerank: critique, first_valid, random, or ``dqn``
    (Bellman DQN over candidate indices using ``board_data`` + replay/target net).
    Inherits vision + text behavior from ``OllamaAgent``.
    """

    def __init__(
        self,
        episode_path,
        episode_logger,
        api_key_file_path,
        instruction_prompt_file_path,
        visual_state_embedding,
        single_images=True,
        COT=False,
        delay=0,
        max_history=0,
        model_type=None,
        ollama_base_url="http://localhost:11434",
        temperature=0.5,
        top_p=0.9,
        num_predict=500,
        keep_alive=None,
        timeout=600,
        *,
        n_candidates: int = 3,
        rerank_mode: str = "critique",
        critique_temperature: float = 0.2,
        critique_num_predict: int = 64,
        proposal_temperature: Optional[float] = None,
        proposal_num_predict: Optional[int] = None,
        rl_hparams: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            episode_path=episode_path,
            episode_logger=episode_logger,
            api_key_file_path=api_key_file_path,
            instruction_prompt_file_path=instruction_prompt_file_path,
            visual_state_embedding=visual_state_embedding,
            single_images=single_images,
            COT=COT,
            delay=delay,
            max_history=max_history,
            model_type=model_type,
            ollama_base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            keep_alive=keep_alive,
            timeout=timeout,
        )
        self.n_candidates = max(1, int(n_candidates))
        self.rerank_mode = (
            rerank_mode
            if rerank_mode in ("critique", "first_valid", "random", "dqn")
            else "critique"
        )
        self.critique_temperature = float(critique_temperature)
        self.critique_num_predict = int(critique_num_predict)
        self.proposal_temperature = proposal_temperature
        # Shorter cap reduces prompt re-echo in proposal completions (None = min(num_predict, 400)).
        self.proposal_num_predict = proposal_num_predict
        self._rl_hp: Dict[str, Any] = dict(rl_hparams or {})
        self._selector_brain = None  # lazy: CognitiveSelectorBrain
        self._dqn_last_obs: Optional[np.ndarray] = None
        self._dqn_last_action: Optional[int] = None
        self._dqn_last_correct: int = 0
        self._dqn_finalize_done: bool = False

    @property
    def act_needs_game_reference(self) -> bool:
        return self.rerank_mode == "dqn"

    def _get_selector_brain(self):
        if self._selector_brain is None:
            try:
                from agent.cognitive_selector_dqn import CognitiveSelectorBrain
            except ModuleNotFoundError as e:
                if getattr(e, "name", "") == "torch" or "torch" in str(e):
                    raise ImportError(
                        "rerank_mode=dqn requires PyTorch. Install with: pip install torch"
                    ) from e
                raise
            self._selector_brain = CognitiveSelectorBrain.get(self.n_candidates, self._rl_hp)
        return self._selector_brain

    def _proposal_max_tokens(self) -> int:
        if self.proposal_num_predict is not None:
            return max(64, int(self.proposal_num_predict))
        return max(128, min(int(self.num_predict), 400))

    def _finalize_chosen_move(self, chosen: str) -> str:
        """Single-line validated command; avoids dumping multi-line blobs into parse_action."""
        s = sanitize_chosen_move(chosen)
        if not is_valid_move_command(s):
            self.episode_logger.warning("Rejected invalid chosen move: %r", chosen[:300])
            bad = "No valid action found."
            self.episode_logger.info(f"\nCognitive agent final action: {bad}")
            self.chat_history.append({"response": bad})
            return bad
        action, thoughts = self.parse_action(f"action: {s}")
        thoughts = self.parse_action_rmv_special_chars(thoughts)
        action = self.parse_action_rmv_special_chars(action)
        return self._finalize_action(action, thoughts)

    def _fallback_single_move_from_raw(self, content: str) -> str:
        """If structured extraction fails, scan for one valid line or trim-then-parse."""
        found = extract_candidate_moves(content, 1)
        if found:
            return self._finalize_chosen_move(found[0])
        trimmed = "\n".join(content.replace("\r\n", "\n").split("\n")[:40])
        action, thoughts = self.parse_action(trimmed)
        thoughts = self.parse_action_rmv_special_chars(thoughts)
        action = self.parse_action_rmv_special_chars(action)
        self.episode_logger.info(f"\nCognitive agent fallback (single parse): {action}")
        return self._finalize_action(action, thoughts)

    def _dqn_reset_episode(self) -> None:
        self._dqn_last_obs = None
        self._dqn_last_action = None
        self._dqn_last_correct = 0
        self._dqn_finalize_done = False

    def _dqn_step_reward(self, sim: dict, cur_correct: int) -> float:
        hp = self._rl_hp
        r = float(hp.get("rl_step_penalty", -0.02))
        r += float(hp.get("rl_shaping_scale", 0.5)) * (cur_correct - self._dqn_last_correct)
        if action_invalid(sim):
            r += float(hp.get("rl_illegal_penalty", -0.5))
        if sim.get("game_done"):
            r += float(hp.get("rl_win_bonus", 10.0))
        return r

    def _critique_pick_text(self, candidates: List[str], goal_state: str, current_state: str) -> str:
        crit = f"""
{self._critique_prompt(candidates)}

Context reminder:
Goal:\n{goal_state}\n
Current:\n{current_state}
"""
        crit_resp = self._ollama_generate(
            prompt=crit.strip(),
            images=None,
            temperature=self.critique_temperature,
            num_predict=self.critique_num_predict,
        )
        self.episode_logger.info(f"\nCognitive agent critique raw: {crit_resp}")
        idx = self._parse_critique_choice(crit_resp, len(candidates))
        return candidates[idx]

    def _critique_pick_vision(
        self,
        candidates: List[str],
        current_base64_goal: str,
        current_base64_current: str,
    ) -> str:
        crit = self._critique_prompt(candidates)
        crit_resp = self._ollama_generate(
            prompt=crit,
            images=[current_base64_goal, current_base64_current],
            temperature=self.critique_temperature,
            num_predict=self.critique_num_predict,
        )
        self.episode_logger.info(f"\nCognitive agent critique raw: {crit_resp}")
        idx = self._parse_critique_choice(crit_resp, len(candidates))
        return candidates[idx]

    def _dqn_select_candidate_text(
        self,
        candidates: List[str],
        game: Any,
        loop_iteration: int,
        goal_state: str,
        current_state: str,
    ) -> str:
        brain = self._get_selector_brain()
        sim = getattr(game, "last_sim_dict", None)
        if not isinstance(sim, dict):
            logging.warning("DQN selector: missing last_sim_dict; using critique.")
            return self._critique_pick_text(candidates, goal_state, current_state)

        bd = sim.get("board_data") or []
        if not bd:
            logging.warning("DQN selector: empty board_data; using critique.")
            return self._critique_pick_text(candidates, goal_state, current_state)

        obs = encode_board_obs(bd, brain.max_objects, brain.grid_size)
        m = len(candidates)
        K = brain.K
        legal = np.zeros(K, dtype=np.bool_)
        legal[: min(m, K)] = True

        cur_correct = count_correct(bd)

        if (
            loop_iteration >= 2
            and self._dqn_last_obs is not None
            and self._dqn_last_action is not None
        ):
            r = self._dqn_step_reward(sim, cur_correct)
            done = bool(sim.get("game_done"))
            brain.push(self._dqn_last_obs, self._dqn_last_action, r, obs, done, legal)

        idx = brain.select_action(obs, legal)
        if not legal[idx] and np.any(legal):
            idx = int(np.flatnonzero(legal)[0])

        self._dqn_last_obs = obs
        self._dqn_last_action = idx
        self._dqn_last_correct = cur_correct

        idx = max(0, min(idx, m - 1))
        choice = candidates[idx]
        self.episode_logger.info(
            "\nCognitive DQN selector: eps=%.3f idx=%s/%s choice=%s",
            brain.epsilon(),
            idx,
            m,
            choice,
        )
        return choice

    def _dqn_select_candidate_vision(
        self,
        candidates: List[str],
        game: Any,
        loop_iteration: int,
        current_base64_goal: str,
        current_base64_current: str,
    ) -> str:
        brain = self._get_selector_brain()
        sim = getattr(game, "last_sim_dict", None)
        if not isinstance(sim, dict):
            logging.warning("DQN selector: missing last_sim_dict; using critique.")
            return self._critique_pick_vision(
                candidates, current_base64_goal, current_base64_current
            )

        bd = sim.get("board_data") or []
        if not bd:
            logging.warning("DQN selector: empty board_data; using critique.")
            return self._critique_pick_vision(
                candidates, current_base64_goal, current_base64_current
            )

        obs = encode_board_obs(bd, brain.max_objects, brain.grid_size)
        m = len(candidates)
        K = brain.K
        legal = np.zeros(K, dtype=np.bool_)
        legal[: min(m, K)] = True
        cur_correct = count_correct(bd)

        if (
            loop_iteration >= 2
            and self._dqn_last_obs is not None
            and self._dqn_last_action is not None
        ):
            r = self._dqn_step_reward(sim, cur_correct)
            done = bool(sim.get("game_done"))
            brain.push(self._dqn_last_obs, self._dqn_last_action, r, obs, done, legal)

        idx = brain.select_action(obs, legal)
        if not legal[idx] and np.any(legal):
            idx = int(np.flatnonzero(legal)[0])

        self._dqn_last_obs = obs
        self._dqn_last_action = idx
        self._dqn_last_correct = cur_correct

        idx = max(0, min(idx, m - 1))
        choice = candidates[idx]
        self.episode_logger.info(
            "\nCognitive DQN selector (vision): eps=%.3f idx=%s/%s choice=%s",
            brain.epsilon(),
            idx,
            m,
            choice,
        )
        return choice

    def finalize_episode_rl(self, message_data: dict, game: Any) -> None:
        """Bellman backup for last transition when the loop exits without another act()."""
        if self.rerank_mode != "dqn" or self._dqn_finalize_done:
            return
        sim = getattr(game, "last_sim_dict", None)
        if not isinstance(sim, dict) or self._dqn_last_obs is None or self._dqn_last_action is None:
            return

        brain = self._get_selector_brain()
        bd = sim.get("board_data") or []
        if not bd:
            return

        next_obs = encode_board_obs(bd, brain.max_objects, brain.grid_size)
        next_legal = np.ones(brain.K, dtype=np.bool_)
        cur_correct = count_correct(bd)

        r = self._dqn_step_reward(sim, cur_correct)
        mx = getattr(game, "max_game_length", None)
        it = getattr(game, "iteration", 0)
        forced = mx is not None and it >= mx
        done = bool(sim.get("game_done")) or forced

        brain.push(
            self._dqn_last_obs,
            self._dqn_last_action,
            r,
            next_obs,
            done,
            next_legal,
        )
        self._dqn_finalize_done = True

        ckpt = self._rl_hp.get("rl_selector_checkpoint_path")
        if ckpt:
            try:
                brain.save_checkpoint(os.path.abspath(str(ckpt)))
            except Exception as e:
                logging.error("Cognitive selector checkpoint save failed: %s", e)

    def _ollama_generate(
        self,
        *,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        t = temperature if temperature is not None else self.temperature
        npred = num_predict if num_predict is not None else self.num_predict
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": t,
                "top_p": self.top_p,
                "num_predict": npred,
            },
        }
        if images is not None:
            payload["images"] = images
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _proposal_prompt_vision(self, text_snippet_active: str, text_snippet_goal: str, history_context: str) -> str:
        n = self.n_candidates
        return f"""
## Analyze the Images
You are given exactly two images:
1) The goal state image {text_snippet_goal}.
2) The current active state image {text_snippet_active}.

Propose **{n} different** next moves that could help reach the goal (diverse if possible).
Each move must be on its own line (no extra text on those lines), exactly like:
action: move red cube left
(but replace red/cube/left with a real color, shape, and direction from the images.)

Allowed colors: green, red, blue, yellow.
Allowed shapes: cube, sphere, pyramid, cylinder.
Allowed directions: up, down, left, right.

**Do not** use angle brackets or placeholder words like "object color".
**Do not** paste this instruction block again in your answer.

## Invalid Actions:
- No Overlap: You are not allowed to position two objects in the same tile.
- If a move does not move any objects, it is invalid.

{history_context}

After listing the {n} lines, you may add a short note, but the lines starting with action: must appear first.
""".strip()

    def _proposal_prompt_text(self, goal_state: str, current_state: str, history_context: str) -> str:
        n = self.n_candidates
        return f"""
## Analyze the States
Goal state:
{goal_state}

Current state:
{current_state}

Propose **{n} different** next moves (diverse if possible).
Each move on its own line, exactly like:
action: move yellow sphere down
(but replace with real color, shape, direction from the states above.)

Allowed colors: green, red, blue, yellow.
Allowed shapes: cube, sphere, pyramid, cylinder.
Allowed directions: up, down, left, right.

**Do not** use angle brackets or placeholder words like "object color".
**Do not** paste this instruction block or the goal/current state again in your answer.

## Invalid Actions:
- No Overlap: You are not allowed to position two objects in the same tile.

{history_context}
""".strip()

    def _critique_prompt(self, candidates: List[str]) -> str:
        lines = "\n".join(f"{i + 1}) {c}" for i, c in enumerate(candidates))
        return f"""
You are selecting the single best next move for the sliding puzzle.

Candidate moves:
{lines}

Reply with **only one digit**: the number (1-{len(candidates)}) of the best move. No other text.
""".strip()

    def _parse_critique_choice(self, text: str, n: int) -> int:
        text = text.strip()
        m = re.search(r"[1-9]", text)
        if m:
            idx = int(m.group(0)) - 1
            if 0 <= idx < n:
                return idx
        return 0

    def _finalize_action(self, move: str, thoughts: str) -> str:
        move = self.parse_action_rmv_special_chars(move)
        self.episode_logger.info(f"\nCognitive agent final action: {move}")
        self.chat_history.append({"response": move})
        return move

    def act(self, observation, loop_iteration, game=None):
        if self.rerank_mode == "dqn" and game is None:
            logging.error("CognitiveAgent rerank_mode=dqn requires `game` (loop passes it automatically).")
            return "error"
        if self.rerank_mode == "dqn" and loop_iteration == 0:
            self._dqn_reset_episode()

        if isinstance(observation, Image.Image):
            return self._act_vision(observation, loop_iteration, game)
        if isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            return self._act_text(observation, loop_iteration, game)
        raise ValueError(f"CognitiveAgent unsupported observation type: {type(observation)}")

    def _act_vision(self, observation: Image.Image, loop_iteration: int, game: Any = None):
        try:
            if self.goal_state is None:
                return self.process_goal_state(observation)

            current_base64_goal, current_base64_current = self._embed_images(observation)
            text_snippet_active, text_snippet_goal = self._prompt_snippets()
            history_context = self._history_context()

            prop_temp = self.proposal_temperature if self.proposal_temperature is not None else self.temperature
            prompt = self._proposal_prompt_vision(text_snippet_active, text_snippet_goal, history_context)
            content = self._ollama_generate(
                prompt=prompt,
                images=[current_base64_goal, current_base64_current],
                temperature=prop_temp,
                num_predict=self._proposal_max_tokens(),
            )

            self.episode_logger.info(f"\nCognitive agent proposal raw response:\n{content}")
            candidates = extract_candidate_moves(content, self.n_candidates)

            if not candidates:
                return self._fallback_single_move_from_raw(content)

            chosen = self._rerank_vision(
                candidates,
                current_base64_goal,
                current_base64_current,
                game=game,
                loop_iteration=loop_iteration,
            )
            return self._finalize_chosen_move(chosen)
        except Exception as e:
            logging.error(f"\nError in CognitiveAgent (vision): {e}")
            return "error"

    def _rerank_vision(
        self,
        candidates: List[str],
        current_base64_goal: str,
        current_base64_current: str,
        *,
        game: Any = None,
        loop_iteration: int = 0,
    ) -> str:
        if self.rerank_mode == "dqn":
            if game is None:
                logging.error("dqn rerank needs game reference")
                for c in candidates:
                    s = sanitize_chosen_move(c)
                    if is_valid_move_command(s):
                        return s
                return candidates[0]
            return self._dqn_select_candidate_vision(
                candidates,
                game,
                loop_iteration,
                current_base64_goal,
                current_base64_current,
            )
        if self.rerank_mode == "random":
            return random.choice(candidates)
        if self.rerank_mode == "first_valid":
            for c in candidates:
                s = sanitize_chosen_move(c)
                if is_valid_move_command(s):
                    return s
            return candidates[0]

        return self._critique_pick_vision(
            candidates, current_base64_goal, current_base64_current
        )

    def _act_text(self, observation: List[str], loop_iteration: int, game: Any = None):
        try:
            if self.goal_state is None:
                return self.process_goal_state(observation)

            current_state = "\n".join(observation)
            goal_state = self.goal_state
            if isinstance(goal_state, list):
                goal_state = "\n".join(goal_state)

            history_context = self._history_context()
            prop_temp = self.proposal_temperature if self.proposal_temperature is not None else self.temperature
            prompt = self._proposal_prompt_text(goal_state, current_state, history_context)
            content = self._ollama_generate(
                prompt=prompt,
                images=None,
                temperature=prop_temp,
                num_predict=self._proposal_max_tokens(),
            )

            self.episode_logger.info(f"\nCognitive agent proposal raw response:\n{content}")
            candidates = extract_candidate_moves(content, self.n_candidates)

            if not candidates:
                return self._fallback_single_move_from_raw(content)

            chosen = self._rerank_text(
                candidates,
                goal_state,
                current_state,
                game=game,
                loop_iteration=loop_iteration,
            )
            return self._finalize_chosen_move(chosen)
        except Exception as e:
            logging.error(f"\nError in CognitiveAgent (text): {e}")
            return "error"

    def _rerank_text(
        self,
        candidates: List[str],
        goal_state: str,
        current_state: str,
        *,
        game: Any = None,
        loop_iteration: int = 0,
    ) -> str:
        if self.rerank_mode == "dqn":
            if game is None:
                logging.error("dqn rerank needs game reference")
                for c in candidates:
                    s = sanitize_chosen_move(c)
                    if is_valid_move_command(s):
                        return s
                return candidates[0]
            return self._dqn_select_candidate_text(
                candidates,
                game,
                loop_iteration,
                goal_state,
                current_state,
            )
        if self.rerank_mode == "random":
            return random.choice(candidates)
        if self.rerank_mode == "first_valid":
            for c in candidates:
                s = sanitize_chosen_move(c)
                if is_valid_move_command(s):
                    return s
            return candidates[0]

        return self._critique_pick_text(candidates, goal_state, current_state)
