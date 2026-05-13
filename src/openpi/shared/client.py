import base64
from collections import deque
from collections.abc import Callable
import inspect
from io import BytesIO
import json
import logging
import re
from typing import Any
import uuid

import imageio.v2 as iio
from json_repair import repair_json
import numpy as np
import openai
from openai import OpenAIError
from openai._types import NOT_GIVEN
from PIL import Image
from pydantic import validate_call
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("client.log"), logging.StreamHandler()],
)


def _retry_error_callback(retry_state):
    last_exception = retry_state.outcome.exception()
    args = last_exception.args
    return args


def pack_images_to_video(image_items: list[np.ndarray], uuid: uuid.UUID) -> str:
    video_path = f"~/.cache/openpi/{uuid}.mp4"
    iio.mimwrite(video_path, image_items, fps=20, codec="libx264")
    return video_path


def encode(image, format="png"):
    if type(image) is str:
        suffix = image.split(".")[-1]
        if suffix == "mp4":
            return f"data:video/mp4;base64,{base64.b64encode(open(image, 'rb').read()).decode('utf-8')}"
        prefix = "data:image/jpeg;base64" if suffix in ("jpg", "jpeg") else "data:image/png;base64"
        return f"{prefix},{base64.b64encode(open(image, 'rb').read()).decode('utf-8')}"
    if type(image) is bytes:
        return f"data:image/{format};base64,{base64.b64encode(image).decode('utf-8')}"
    # image shape is (h, w, 3)
    if type(image) is torch.Tensor:
        image = np.array(image)
    if type(image) is np.ndarray:
        image = Image.fromarray(image)
    if type(image) is Image.Image:
        buffered = BytesIO()
        image.save(buffered, format=format)
        return f"data:image/{format};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def is_video(multi_modal_item: str):
    try:
        return multi_modal_item.rsplit(".", maxsplit=1)[-1] == "mp4"
    except:
        return False


def sample_images(image_list, max_len=64):
    sampled_images = []
    for i in range(len(image_list) - 1, -1, -5):
        sampled_images.append(image_list[i])
        if len(sampled_images) >= max_len:
            break
    sampled_images.reverse()
    return sampled_images


def strip_think_tags(text: str) -> str:
    """
    Remove content within <think></think> tags from the text.
    Handles multiple scenarios:
    1. Proper <think>...</think> pairs - removes content between tags
    2. Only </think> tag - removes everything up to and including </think>
    3. No think tags - returns text as-is

    Args:
        text: Input text that may contain <think></think> tags

    Returns:
        Text with <think></think> content removed and cleaned up
    """
    # Check if there's a proper <think>...</think> pair
    if "<think>" in text.lower() and "</think>" in text.lower():
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    elif "</think>" in text.lower():
        # If there's only </think> without <think>, remove everything up to and including </think>
        cleaned_text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        # No think tags at all, use the text as is
        cleaned_text = text

    # Clean up extra whitespace
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())
    return cleaned_text


def load_json_from_str(text: str) -> list | dict:
    repaired_text = strip_think_tags(text)
    repaired_text = repair_json(repaired_text)
    return json.loads(repaired_text)


def generate_stylized_plan(list_of_plans: list[str], current_plan: str, current_plan_completed: bool = False) -> str:
    """
    Generate a stylized high-level plan list with status markers.

    Rules:
    - All plans before the current plan are marked as [o].
    - If current_plan_completed is False:
        - Current plan is marked as [-].
        - All plans after the current plan are marked as [x].
    - If current_plan_completed is True:
        - Current plan is marked as [o].
        - The plan immediately after the current plan is marked as [-] (if it exists).
        - All remaining plans after that are marked as [x].

    Args:
        list_of_plans: Ordered list of plan step strings.
        current_plan: The plan string that is the current focus.
        current_plan_completed: Whether the current plan has been completed.

    Returns:
        A single string with one formatted plan per line.

    Raises:
        ValueError: If current_plan is not found in list_of_plans.
    """
    normalized_plans = [p.strip() for p in list_of_plans]
    target = current_plan.strip()
    try:
        current_index = next(i for i, p in enumerate(normalized_plans) if p == target)
    except StopIteration:
        raise ValueError(f"current_plan not found in list_of_plans: '{current_plan}'")

    lines: list[str] = []
    for i, plan in enumerate(list_of_plans):
        if i < current_index:
            prefix = "[o]"
        elif i == current_index:
            prefix = "[o]" if current_plan_completed else "[-]"
        elif current_plan_completed and i == current_index + 1:
            prefix = "[-]"
        else:
            prefix = "[x]"
        lines.append(f"{prefix} {plan}")
    return "\n".join(lines)


class Client:
    def __init__(
        self,
        api_key="bXAp7xSTTlevMkOL5TxiDhK7V2fMhUcM",
        base_url="https://b5k2m9x7-pre-exp011-043-32000.xenon.lepton.run/v1",
        model="Qwen3-VL-30B-A3B-Instruct",
    ):
        self.model = model
        self.preserved_kwargs = inspect.signature(
            openai.OpenAI(api_key=api_key).beta.chat.completions.parse
        ).parameters.keys()
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=600)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.uuid = uuid.uuid4()
        self.history_multi_modals = deque(maxlen=64 * 10)
        self.plan_status = ""
        self.subtask_history: list[str] = []

    def reset(self):
        self.history_multi_modals.clear()
        self.uuid = uuid.uuid4()
        self.plan_status = ""
        self.subtask_history = []

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_random_exponential(min=2, max=15),
        retry=retry_if_exception_type(OpenAIError),
        retry_error_callback=_retry_error_callback,
    )
    @validate_call
    def create_request(
        self,
        *,
        user_prompt: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        multi_modals: list[Any] = [],
        encode_format: str = "png",
        conversations: list[dict[str, Any]] = [],
        max_images_before_packed: int = 20,
        id_key: str | None = None,
        filter_fn: Callable[[str], bool] = lambda x: True,
        **kwargs,
    ):
        self.history_multi_modals.extend(multi_modals)
        multi_modals = sample_images(self.history_multi_modals)
        if len(multi_modals) > max_images_before_packed:
            video_path = pack_images_to_video(multi_modals, self.uuid)
            multi_modals = [video_path]

        if user_prompt is None and len(conversations) == 0:
            raise ValueError("user_prompt and conversations cannot be both None")
        if user_prompt is not None and len(conversations) > 0:
            raise ValueError("user_prompt and conversations cannot be both not None")
        if user_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        *[
                            {
                                "type": "image_url" if not is_video(multi_modal_item) else "video_url",
                                "image_url" if not is_video(multi_modal_item) else "video_url": {
                                    "url": encode(multi_modal_item, encode_format)
                                },
                            }
                            for multi_modal_item in multi_modals
                        ],
                    ],
                },
            ]
        else:
            messages = conversations

        if id_key is None:
            current_sample_id = ""
        else:
            current_sample_id = kwargs.get(id_key)
        try:
            self.logger.info(f"Creating request {current_sample_id}")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", NOT_GIVEN),
                top_p=kwargs.get("top_p", NOT_GIVEN),
                response_format=kwargs.get("response_format", NOT_GIVEN),
                extra_body=kwargs.get("extra_body"),
            )
            reasoning_content = (
                f"<raw_think>{response.choices[0].message.reasoning_content}</raw_think>\n"
                if hasattr(response.choices[0].message, "reasoning_content")
                and response.choices[0].message.reasoning_content is not None
                else ""
            )
            result = reasoning_content + response.choices[0].message.content
            if filter_fn(result):
                return True, {
                    "result": result,
                    **{k: v for k, v in kwargs.items() if k not in self.preserved_kwargs},
                }
            self.logger.warning(f"Result {current_sample_id} does not pass filter_fn")
            raise Exception(False, {"result": result})

        except TimeoutError as e:
            self.logger.error(f"Timeout {current_sample_id}")
            raise Exception(False, {f"{id_key}": current_sample_id, "error": "timeout"}) from e

        except Exception as e:
            self.logger.error(f"Error {current_sample_id}: {e!s}")
            raise Exception(False, {f"{id_key}": current_sample_id, "error": f"{e!s}"}) from e

    def generate_plan(self, high_level_task, initial_image):
        user_prompt = f"""Given the task '{high_level_task}', break it down into several concrete high-level steps."""

        response = self.create_request(user_prompt=user_prompt, multi_modals=[initial_image])
        response = response[1]["result"]
        plans = load_json_from_str(response)
        current_plan = plans[0]
        plan_status = generate_stylized_plan(plans, current_plan, current_plan_completed=False)
        return plan_status

    def plan_critique(self, high_level_task, multi_modals) -> str:
        user_prompt = f"""You are given the task of '{high_level_task}'. The status of the plans are:
    {self.plan_status}
    Note that [-] indicates in progress. [o] indicates completed. [x] indicates not started.
    The last high-level objective given to the robot was '{self.subtask_history[-1]}'. Based on your analysis, what are the updated status of the plans?"""

        response = self.create_request(user_prompt=user_prompt, multi_modals=multi_modals)
        response = response[1]["result"]
        updated_plan_status = strip_think_tags(response)
        return updated_plan_status

    def generate_subtask(self, high_level_task, multi_modals):
        if len(self.history_multi_modals) == 0:
            self.plan_status = self.generate_plan(high_level_task, multi_modals[0])
            self.subtask_history = []

        if len(self.subtask_history) != 0:
            updated_plan_status = self.plan_critique(high_level_task, multi_modals)
            if updated_plan_status != self.plan_status:
                self.subtask_history = []
            self.plan_status = updated_plan_status

        user_prompt = f"""You are given the task of '{high_level_task}'. The status of the plans are:
    {self.plan_status}
    Note that [-] indicates in progress. [o] indicates completed. [x] indicates not started.
    The last high-level objective given to the robot was '{self.subtask_history[-1] if len(self.subtask_history) != 0 else "None"}'.Based on your analysis, what should be the next high-level objective the robot should achieve?"""

        response = self.create_request(user_prompt=user_prompt, multi_modals=multi_modals)
        response = response[1]["result"]
        subtask = strip_think_tags(response)
        self.subtask_history.append(subtask)
        return subtask
