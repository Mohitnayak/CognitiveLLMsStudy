# ---------------------------------------------------------------------------
# OllamaAgent for iVISPAR
# Add this class to iVISPAR's Source/Experiment/agent_systems.py (at the end).
# Ensure "ollama" is installed in the iVISPAR conda env: pip install ollama
# ---------------------------------------------------------------------------

import ollama


class OllamaAgent(LLMAgent):
    """
    LLM agent that uses a local Ollama server (e.g. llava, qwen3-vl).
    Same interface as GPT4Agent but calls Ollama API; no API key required.
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
        ollama_model="llava",
    ):
        super().__init__(
            episode_path,
            episode_logger,
            api_key_file_path,
            instruction_prompt_file_path,
            visual_state_embedding,
            single_images,
            COT,
            delay,
            max_history,
        )
        self.ollama_model = ollama_model
        self.chat_history = []

    def act(self, observation, loop_iteration):
        if isinstance(observation, Image.Image):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")

                if self.visual_state_embedding == "color":
                    current = self.color_code_observation(
                        observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )
                    goal = self.color_code_observation(
                        self.goal_state.copy(), tuple(self.color_codes["goal_1"]["rgb"])
                    )
                elif self.visual_state_embedding == "label":
                    current = self.color_code_observation(
                        observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )
                    goal = self.color_code_observation(
                        self.goal_state.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )
                    current = self.add_action_text(current.copy(), "active")
                    goal = self.add_action_text(goal.copy(), "goal")
                elif self.visual_state_embedding == "both":
                    current = self.color_code_observation(
                        observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )
                    goal = self.color_code_observation(
                        self.goal_state.copy(), tuple(self.color_codes["goal_1"]["rgb"])
                    )
                    current = self.add_action_text(current.copy(), "active")
                    goal = self.add_action_text(goal.copy(), "goal")
                else:
                    current = self.color_code_observation(
                        observation.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )
                    goal = self.color_code_observation(
                        self.goal_state.copy(), tuple(self.color_codes["active_1"]["rgb"])
                    )

                current_base64 = self.encode_image_from_pil(current)
                goal_base64 = self.encode_image_from_pil(goal)

                instruction = (
                    "You see two images: the first is the GOAL state, the second is the CURRENT state. "
                    "Output exactly one move to bring the current state closer to the goal. "
                    "End your response with: action: move <color> <shape> <direction> "
                    "(e.g. action: move blue sphere right). Use only: sphere, cube, pyramid, cylinder and up, down, left, right."
                )

                messages = [{"role": "system", "content": self.system_prompt}]
                if self.max_history > 0:
                    for prev_exchange in self.chat_history[-self.max_history :]:
                        prev_img = self.color_code_observation(
                            prev_exchange["observation"].copy(),
                            tuple(self.color_codes["past"]["rgb"])
                            if self.visual_state_embedding in ("color", "both")
                            else tuple(self.color_codes["active_1"]["rgb"]),
                        )
                        if self.visual_state_embedding in ("label", "both"):
                            prev_img = self.add_action_text(prev_img.copy(), "past")
                        prev_b64 = self.encode_image_from_pil(prev_img)
                        messages.append(
                            {
                                "role": "user",
                                "content": "Previous state:",
                                "images": [prev_b64],
                            }
                        )
                        messages.append(
                            {"role": "assistant", "content": prev_exchange["response"]}
                        )

                messages.append(
                    {
                        "role": "user",
                        "content": instruction,
                        "images": [goal_base64, current_base64],
                    }
                )

                response = ollama.chat(model=self.ollama_model, messages=messages)
                content = (
                    response.get("message", {}).get("content", "")
                    if isinstance(response, dict)
                    else getattr(
                        getattr(response, "message", None), "content", ""
                    )
                )
                action, thoughts = self.parse_action(content)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nOllama thoughts: {thoughts}")
                self.episode_logger.info(f"\nOllama action: {action}")

                self.chat_history.append({"observation": observation, "response": action})
                return action

            except Exception as e:
                logging.error(f"\nError calling Ollama: {e}")
                return "error"

        elif isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            observation = "\n".join(observation)
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State (text) ===")
                instruction = (
                    f"Goal state:\n{self.goal_state}\n\nCurrent state:\n{observation}\n\n"
                    "Output one move. End with: action: move <color> <shape> <direction>"
                )
                messages = [{"role": "system", "content": self.system_prompt}]
                if self.max_history > 0:
                    for prev_exchange in self.chat_history[-self.max_history :]:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Previous observation: {prev_exchange['observation']}",
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": prev_exchange["response"],
                            }
                        )
                messages.append({"role": "user", "content": instruction})

                response = ollama.chat(model=self.ollama_model, messages=messages)
                content = (
                    response.get("message", {}).get("content", "")
                    if isinstance(response, dict)
                    else getattr(
                        getattr(response, "message", None), "content", ""
                    )
                )
                action, thoughts = self.parse_action(content)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nOllama action: {action}")
                self.chat_history.append({"observation": observation, "response": action})
                return action

            except Exception as e:
                logging.error(f"\nError calling Ollama: {e}")
                return "error"

        return "error"
