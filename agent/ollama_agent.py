"""
Ollama-backed agent for iVISPAR.

Uses the Ollama HTTP `/api/generate` endpoint and supports vision models by
sending base64-encoded images in the `images` field.
"""

from __future__ import annotations

import logging

import requests
from PIL import Image

from agent_systems import LLMAgent


class OllamaAgent(LLMAgent):
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
        )
        self.model = model_type or "llama3.2-vision"
        self.ollama_base_url = (ollama_base_url or "http://localhost:11434").rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.chat_history = []

    def _prompt_snippets(self):
        if self.visual_state_embedding == "both":
            text_snippet_active = f"marked with the label {self.color_codes['active_1']['type']} in the image and a {self.color_codes['active_1']['color']} background"
            text_snippet_goal = f"marked with the label {self.color_codes['goal_1']['type']} in the image and a {self.color_codes['goal_1']['color']} background"
            return text_snippet_active, text_snippet_goal
        if self.visual_state_embedding == "color":
            text_snippet_active = f" with a {self.color_codes['active_1']['color']} background"
            text_snippet_goal = f" with a {self.color_codes['goal_1']['color']} background"
            return text_snippet_active, text_snippet_goal
        if self.visual_state_embedding == "label":
            text_snippet_active = f"showing the label in the image {self.color_codes['active_1']['type']}"
            text_snippet_goal = f"showing the label in the image {self.color_codes['goal_1']['type']}"
            return text_snippet_active, text_snippet_goal
        if self.visual_state_embedding == "none":
            return "", ""
        raise ValueError("No viable image state embedding set for agent.")

    def _embed_images(self, current_observation):
        if self.visual_state_embedding == "color":
            current = self.color_code_observation(
                current_observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
            goal = self.color_code_observation(
                self.goal_state.copy(), tuple(self.color_codes["goal_1"]["rgb"])
            )
        elif self.visual_state_embedding == "label":
            # Keep behavior consistent with GPT4Agent/ClaudeAgent in this repo:
            # both current and goal get the same base color, then labels are added.
            current = self.color_code_observation(
                current_observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
            goal = self.color_code_observation(
                self.goal_state.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
            current = self.add_action_text(current.copy(), "active")
            goal = self.add_action_text(goal.copy(), "goal")
        elif self.visual_state_embedding == "both":
            current = self.color_code_observation(
                current_observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
            goal = self.color_code_observation(
                self.goal_state.copy(), tuple(self.color_codes["goal_1"]["rgb"])
            )
            current = self.add_action_text(current.copy(), "active")
            goal = self.add_action_text(goal.copy(), "goal")
        elif self.visual_state_embedding == "none":
            current = self.color_code_observation(
                current_observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
            goal = self.color_code_observation(
                self.goal_state.copy(), tuple(self.color_codes["active_1"]["rgb"])
            )
        else:
            raise ValueError("No viable image state embedding set for agent.")

        goal_base64 = self.encode_image_from_pil(goal)
        current_base64 = self.encode_image_from_pil(current)
        return goal_base64, current_base64

    def _history_context(self):
        if self.max_history <= 0 or not self.chat_history:
            return ""
        recent = self.chat_history[-self.max_history :]
        if not recent:
            return ""
        lines = ["Previous actions (oldest -> newest):"]
        for i, item in enumerate(recent, 1):
            action = item.get("response", "")
            if action:
                lines.append(f"{i}) {action}")
        return "\n".join(lines)

    def act(self, observation, loop_iteration):
        if isinstance(observation, Image.Image):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                current_base64_goal, current_base64_current = self._embed_images(observation)
                text_snippet_active, text_snippet_goal = self._prompt_snippets()
                history_context = self._history_context()

                prompt = f"""
## Analyze the Images
You are given exactly two images:
1) The goal state image {text_snippet_goal}.
2) The current active state image {text_snippet_active}.

Study both images and determine how to move objects in the current state to match the goal state.

## Invalid Actions:
- No Overlap: You are not allowed to position two objects in the same tile.
- If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
- Use the current state image to infer why the last action failed (you may use the history below).

{history_context}

It is of most importance you always end your response with this exact format:
action: move <object color> <object shape> <direction>
where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
""".strip()

                payload = {
                    "model": self.model,
                    "system": self.system_prompt,
                    "prompt": prompt,
                    "images": [current_base64_goal, current_base64_current],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.num_predict,
                    },
                }
                if self.keep_alive is not None:
                    payload["keep_alive"] = self.keep_alive

                resp = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("response", "")

                action, thoughts = self.parse_action(content)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)

                self.episode_logger.info(f"\nOllama suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nOllama suggested action: {action}")

                self.chat_history.append({"response": action})
                return action
            except Exception as e:
                logging.error(f"\nError calling Ollama API: {e}")
                return "error"

        # Text-only representation (rare, but keep it consistent with other agents)
        if isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                current_state = "\n".join(observation)
                goal_state = self.goal_state
                if isinstance(goal_state, list):
                    goal_state = "\n".join(goal_state)

                history_context = self._history_context()
                prompt = f"""
## Analyze the States
The goal state is provided as the first set of coordinates:
{goal_state}
The current state is provided as the second set of coordinates:
{current_state}

Determine how to move objects in the current state to match the goal state.

## Invalid Actions:
- No Overlap: You are not allowed to position two objects in the same tile.
- If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).

{history_context}

It is of most importance you always end your response with this exact format:
action: move <object color> <object shape> <direction>
""".strip()

                payload = {
                    "model": self.model,
                    "system": self.system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.num_predict,
                    },
                }
                if self.keep_alive is not None:
                    payload["keep_alive"] = self.keep_alive

                resp = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("response", "")

                action, thoughts = self.parse_action(content)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)

                self.episode_logger.info(f"\nOllama suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nOllama suggested action: {action}")
                self.chat_history.append({"response": action})
                return action
            except Exception as e:
                logging.error(f"\nError calling Ollama API: {e}")
                return "error"

        raise ValueError(f"OllamaAgent unsupported observation type: {type(observation)}")
