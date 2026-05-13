from collections import deque
import copy
import hashlib
import json
import logging
from pathlib import Path
import re

import cv2
import numpy as np
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
import torch

logger = logging.getLogger("policy")
logger.setLevel(20)  # info

TASK_MAPPING_PATH = Path(__file__).resolve().parents[3] / "scripts" / "task_mapping.json"

RESIZE_SIZE = 224
DESPTH_RESIZE_SIZE = 720

SKILL_PROMPT = """
You are a robot that is trying to complete the global task: {task_prompt}

The skills are:
{skill_prompts}

What's the next skill to perform? Only respond with a single skill name.
"""


class B1KPolicyWrapper:
    def __init__(
        self,
        policy: BasePolicy,
        task_name: str = "turning_on_radio",
        prompt_override: str | None = None,
        control_mode: str = "temporal_ensemble",
        max_len: int = 32,  # receeding horizon | receeding temporal mode
        action_horizon: int = 5,  # temporal ensemble mode | receeding temporal mode
        temporal_ensemble_max: int = 3,  # receeding temporal mode
        fine_grained_level: int = 0,
    ) -> None:
        self.policy = policy
        self.task_name = task_name
        self.prompt_override = prompt_override
        # Session id is used to isolate runtime state (e.g., streaming memory) when the
        # same model instance is shared across multiple websocket connections.
        self._session_id: int = id(self)

        # load the task name from the metadata
        with TASK_MAPPING_PATH.open() as f:
            metadata = json.load(f)
        self.task_prompt = prompt_override if prompt_override is not None else metadata[task_name].get("task")
        self.subtask_prompts = metadata[task_name].get("subtask")
        self.skill_prompts = metadata[task_name].get("skill")

        self.control_mode = control_mode
        self.action_queue = deque(maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        self.action_horizon = action_horizon

        self.replan_interval = action_horizon  # K: replan every 10 steps
        self.max_len = max_len  # how long the policy sequences are
        self.temporal_ensemble_max = temporal_ensemble_max  # max number of sequences to ensemble
        self.step_counter = 0
        self.last_policy_inferred = False

        self.fine_grained_level = fine_grained_level
        self.last_generated_subtask = None
        self.last_prompt_debug: dict[str, object] | None = None
        if self.fine_grained_level > 0:
            from openpi.shared.client import Client

            self.reasoner = Client(model="/workspace/model")
        else:
            self.reasoner = None

        self.log_config()

    def _maybe_set_active_session(self) -> None:
        model = getattr(self.policy, "_model", None)
        if model is None:
            return
        setter = getattr(model, "set_active_session", None)
        if callable(setter):
            try:
                setter(self._session_id)
            except Exception:
                # Best-effort: do not break rollout if the underlying model doesn't support it.
                return

    def _maybe_reset_streaming_state(self) -> None:
        model = getattr(self.policy, "_model", None)
        if model is None:
            return
        resetter = getattr(model, "reset_streaming_state", None)
        if callable(resetter):
            try:
                resetter(self._session_id)
            except Exception:
                return

    def log_config(self):
        logger.info(f"{self.task_name=}")
        logger.info(f"{self.control_mode=}")
        logger.info(f"{self.max_len=}")
        logger.info(f"{self.action_horizon=}")
        logger.info(f"{self.temporal_ensemble_max=}")
        logger.info(f"{self.replan_interval=}")
        logger.info(f"{self.fine_grained_level=}")
        logger.info(f"{self.step_counter=}")
        logger.info(f"{self.action_queue=}")
        logger.info(f"{self.task_prompt=}")
        logger.info(f"{self.subtask_prompts=}")
        logger.info(f"{self.skill_prompts=}")

    def server_identity_metadata(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "task_prompt_sha256": hashlib.sha256(self.task_prompt.encode("utf-8")).hexdigest(),
            "prompt_override_used": self.prompt_override is not None,
            "control_mode": self.control_mode,
            "fine_grained_level": self.fine_grained_level,
        }

    def reset(self):
        self.action_queue = deque(maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0
        self.last_policy_inferred = False
        self._maybe_set_active_session()
        self._maybe_reset_streaming_state()
        self.last_generated_subtask = None
        self.last_prompt_debug = None
        if self.reasoner:
            self.reasoner.reset()

    def spawn_session(self) -> "B1KPolicyWrapper":
        """Create a per-connection session wrapper.

        The underlying model (`self.policy`) is shared, but rollout state is isolated.
        This enables multiple evaluators to share one websocket server without their
        resets / action queues colliding.
        """

        # Shallow copy is enough to share large model weights while isolating mutable rollout state.
        session = copy.copy(self)
        session._session_id = id(session)
        session.action_queue = deque(maxlen=self.action_horizon)
        session.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        session.step_counter = 0
        session.last_policy_inferred = False
        session.last_generated_subtask = None
        session.last_prompt_debug = None

        # Ensure any optional reasoner state is not shared.
        if self.reasoner is not None:
            try:
                from openpi.shared.client import Client

                session.reasoner = Client(model="/workspace/model")
            except Exception:
                # Best-effort: disable reasoner rather than sharing mutable state.
                session.reasoner = None
        return session

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE,
                ),
            ],
            axis=1,
        )

        if "robot_r1::robot_r1:right_realsense_link:Camera:0::instance_seg" in obs:
            pass  # TODO: add instance segmentation

        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
        }

        if "robot_r1::robot_r1:zed_link:Camera:0::depth_linear" in obs:
            depth_obs = obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]
            depth_obs = cv2.resize(depth_obs, (DESPTH_RESIZE_SIZE, DESPTH_RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
            processed_obs["observation/egocentric_depth"] = depth_obs[None]

        # if "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear" in obs:
        #     depth_obs = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"][None]
        #     processed_obs["observation/wrist_depth_left"] = depth_obs

        # if "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear" in obs:
        #     depth_obs = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"][None]
        #     processed_obs["observation/wrist_depth_right"] = depth_obs

        return processed_obs

    @staticmethod
    def _normalize_skill_text(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _match_allowed_skill(self, raw_reasoner_output: str | None) -> tuple[str | None, str | None]:
        normalized_output = self._normalize_skill_text(raw_reasoner_output)
        if not normalized_output or not self.skill_prompts:
            return None, None

        normalized_skills = [(skill, self._normalize_skill_text(skill)) for skill in self.skill_prompts if skill]
        for skill, normalized_skill in normalized_skills:
            if normalized_output == normalized_skill:
                return skill, "exact"

        contains_matches = [
            (skill, normalized_skill)
            for skill, normalized_skill in normalized_skills
            if normalized_skill and normalized_skill in normalized_output
        ]
        if contains_matches:
            best_skill, _ = max(contains_matches, key=lambda item: len(item[1]))
            return best_skill, "contains"

        reverse_matches = [
            (skill, normalized_skill)
            for skill, normalized_skill in normalized_skills
            if normalized_output in normalized_skill
        ]
        if reverse_matches:
            best_skill, _ = min(reverse_matches, key=lambda item: len(item[1]))
            return best_skill, "contained_by"

        return None, None

    def _resolve_policy_prompt(self, *, egocentric_camera: np.ndarray | None = None) -> tuple[str, dict[str, object]]:
        prompt_debug: dict[str, object] = {
            "task_name": self.task_name,
            "task_prompt": self.task_prompt,
            "fine_grained_level": self.fine_grained_level,
            "skill_candidates": list(self.skill_prompts or []),
            "reasoner_enabled": self.fine_grained_level > 0 and self.reasoner is not None,
            "reasoner_output": None,
            "reasoner_error": None,
            "selected_skill": None,
            "match_type": None,
            "fallback_to_task_prompt": False,
            "fallback_reason": None,
            "final_prompt": self.task_prompt,
        }
        if not prompt_debug["reasoner_enabled"]:
            prompt_debug["fallback_reason"] = "fine_grained_disabled"
            return self.task_prompt, prompt_debug

        if not self.skill_prompts:
            prompt_debug["fallback_to_task_prompt"] = True
            prompt_debug["fallback_reason"] = "missing_skill_prompts"
            return self.task_prompt, prompt_debug

        try:
            reasoner_response = self.reasoner.generate_subtask(
                high_level_task=self.task_prompt,
                multi_modals=[egocentric_camera] if egocentric_camera is not None else [],
            )
        except Exception as exc:
            logger.warning("Reasoner failed; falling back to task prompt: %s", exc)
            prompt_debug["fallback_to_task_prompt"] = True
            prompt_debug["fallback_reason"] = "reasoner_error"
            prompt_debug["reasoner_error"] = f"{type(exc).__name__}: {exc}"
            return self.task_prompt, prompt_debug
        logger.info(f"* {reasoner_response}")
        prompt_debug["reasoner_output"] = reasoner_response
        selected_skill, match_type = self._match_allowed_skill(reasoner_response)
        if selected_skill is None:
            prompt_debug["fallback_to_task_prompt"] = True
            prompt_debug["fallback_reason"] = "no_skill_match"
            return self.task_prompt, prompt_debug

        prompt_debug["selected_skill"] = selected_skill
        prompt_debug["match_type"] = match_type
        prompt_debug["final_prompt"] = selected_skill
        return selected_skill, prompt_debug

    def act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        self.last_policy_inferred = False
        if self.step_counter % self.replan_interval == 0:
            nbatch = copy.deepcopy(input_obs)
            if nbatch["observation"].shape[-1] != 3:
                # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
                # permute if pytorch
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            # nbatch["proprio"] is B, 16, where B=1
            joint_positions = nbatch["proprio"][0]
            batch = {
                "observation/egocentric_camera": nbatch["observation"][0, 0],
                "observation/wrist_image_left": nbatch["observation"][0, 1],
                "observation/wrist_image_right": nbatch["observation"][0, 2],
                "observation/state": joint_positions,
                "prompt": self.task_prompt,
            }

            batch["prompt"], self.last_prompt_debug = self._resolve_policy_prompt(
                egocentric_camera=batch["observation/egocentric_camera"]
            )

            if "observation/egocentric_depth" in nbatch:
                batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

            try:
                self._maybe_set_active_session()
                action = self.policy.infer(batch)
                self.last_policy_inferred = True
                if "generated_subtask" in action and action["generated_subtask"] is not None:
                    self.last_generated_subtask = action["generated_subtask"]
                self.last_action = action
            except Exception as e:
                action = self.last_action
                logger.info(
                    f"Error in action prediction at step {self.step_counter}, {joint_positions.shape=}, using last action: {e}"
                )

            target_joint_positions = action["actions"].copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[: self.max_len]])
            self.action_queue.append(new_seq)

            # Optional: limit memory
            while len(self.action_queue) > self.temporal_ensemble_max:
                self.action_queue.popleft()

        # Step 2: Smooth across current step from all stored sequences
        if len(self.action_queue) == 0:
            raise ValueError("Action queue empty in receeding_temporal mode.")

        actions_current_timestep = np.empty((len(self.action_queue), self.action_queue[0][0].shape[0]))

        for i in range(len(self.action_queue)):
            actions_current_timestep[i] = self.action_queue[i].popleft()

        # Drop exhausted sequences
        self.action_queue = deque([q for q in self.action_queue if len(q) > 0])

        # Apply temporal ensemble
        k = 0.005
        exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()

        final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)

        # Preserve grippers from most recent rollout
        final_action[-9] = actions_current_timestep[0, -9]
        final_action[-1] = actions_current_timestep[0, -1]
        final_action = final_action[None]

        self.step_counter += 1

        return torch.as_tensor(final_action, dtype=torch.float32)

    def act(self, input_obs):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images
        """
        Model input expected:
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something

        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        input_obs = self.process_obs(input_obs)
        self.last_policy_inferred = False
        if self.control_mode == "receeding_temporal":
            return self.act_receeding_temporal(input_obs)

        if self.control_mode == "receeding_horizon":
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return torch.as_tensor(final_action, dtype=torch.float32)

        nbatch = copy.deepcopy(input_obs)
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, 16, where B=1
        joint_positions = nbatch["proprio"][0]
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": self.task_prompt,
        }

        if "observation/egocentric_depth" in nbatch:
            batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

        batch["prompt"], self.last_prompt_debug = self._resolve_policy_prompt(
            egocentric_camera=batch["observation/egocentric_camera"]
        )

        try:
            self._maybe_set_active_session()
            action = self.policy.infer(batch)
            self.last_policy_inferred = True
            if "generated_subtask" in action and action["generated_subtask"] is not None:
                self.last_generated_subtask = action["generated_subtask"]
            self.last_action = action
        except Exception as e:
            action = self.last_action
            raise e
        # convert to absolute action and append gripper command
        # action shape: (10, 23), joint_positions shape: (23,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy()
        if self.control_mode == "receeding_horizon":
            self.action_queue = deque([a for a in target_joint_positions[: self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == "temporal_ensemble":
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))

            # k = 0.01
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-9] = target_joint_positions[0, -9]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions
        return torch.as_tensor(final_action, dtype=torch.float32)
