from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import memory_cache as _memory_cache
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        held_memory_enabled: bool = False,
        memory_post_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._held_memory_enabled = bool(held_memory_enabled)
        self._memory_post_transform = _transforms.compose(memory_post_transforms)
        # MoMA-VLA held state (design section 2.2 / 4.4).
        #
        # These three names previously existed in this constructor and were never
        # read or written anywhere in the tree.  They are now backed by a real
        # cache with explicit invalidate semantics.  The cache holds memory
        # token IDs only -- MemoryTokenCache refuses a past_key_values payload,
        # because caching the prefix KV would carry a stale image and (for pi05) a
        # stale discretised state into later action chunks (design section 4.5).
        self._memory_cache = _memory_cache.MemoryTokenCache()
        self._cached_subtask_prompt: str | None = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @property
    def memory_cache(self) -> "_memory_cache.MemoryTokenCache":
        """The held memory for this policy instance."""
        return self._memory_cache

    @property
    def _cached_subtask_tokens(self):
        """Back-compat alias for the held memory token ids (None when cold)."""
        return self._memory_cache.tokens

    @property
    def _cached_subtask_text(self) -> str | None:
        """Back-compat alias for the held memory text (None when cold)."""
        return self._memory_cache.text

    def reset(self, reason: str = "episode boundary") -> None:
        """Drop held memory state.

        Call this between episodes.  A held memory that survives an episode
        boundary is stale by construction, and nothing else in the stack will
        notice: it is the caller's responsibility, so it is made explicit here
        rather than inferred.
        """
        self._memory_cache.invalidate(reason)
        self._cached_subtask_prompt = None


    def infer_memory_chunk(
        self,
        obs: dict,
        *,
        planner_tick: bool,
        held_memory_tokens: Any | None,
        previous_memory_text: str | None,
        chunk_index: int,
        noise: np.ndarray | None = None,
    ) -> dict:
        """Run one stateless MoMA action chunk on the caller-owned held memory.

        The wrapper owns time and memory.  This method owns only model execution:
        every call starts again from the supplied raw observation, so image/state
        and prefix KV are rebuilt for this chunk.  Raw generated token ids (without
        BOS) are returned to the wrapper; no KV/cache object crosses the boundary.
        """
        if not self._is_pytorch_model or not hasattr(self._model, "predict_subtask_tokens"):
            raise RuntimeError("HeldMemory inference requires the PyTorch PI05_SUBTASK model")
        if "subtask_text" in obs:
            raise ValueError(
                "MoMA HeldMemory and explicit subtask_text share the action conditioning slot; "
                "this fixed-K P1 path refuses the ambiguous combination."
            )
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise ValueError(f"chunk_index must be a non-negative int, got {chunk_index!r}")

        # Build the latest-observation base once.  Runtime memory fields are
        # injected after B1kInputs/Normalize and before tokenization; the strict
        # offline Memory transform is intentionally bypassed here because there
        # is no teacher-forced current-memory target online.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._memory_post_transform(self._input_transform(inputs))
        if "state" not in inputs or "prompt" not in inputs:
            raise ValueError("runtime Memory inference requires transformed state and prompt")

        tokenizer = getattr(self, "_held_memory_tokenizer", None)
        if tokenizer is None:
            tokenizer = _tokenizer.SubtaskTokenizer(
                prompt_max_len=int(self._model.config.max_token_len),
                subtask_max_len=int(self._model.config.subtask_max_len),
            )
            self._held_memory_tokenizer = tokenizer
        prompt = inputs.pop("prompt")

        def make_observation(previous_text: str | None):
            prompt_tokens, prompt_mask = tokenizer.tokenize_prompt(
                str(prompt), np.asarray(inputs["state"]), previous_memory=previous_text
            )
            payload = {**inputs, "tokenized_prompt": prompt_tokens, "tokenized_prompt_mask": prompt_mask}
            device_payload = jax.tree.map(
                lambda x: torch.from_numpy(np.asarray(x)).to(self._pytorch_device)[None, ...], payload
            )
            return _model.Observation.from_dict(device_payload)

        generated_tokens = None
        generated_text = None
        if planner_tick:
            planner_observation = make_observation(previous_memory_text)
            max_tokens = max(1, int(self._model.config.subtask_max_len) - 1)  # reserve BOS slot
            generated_tokens = self._model.predict_subtask_tokens(
                planner_observation, max_tokens=max_tokens
            )
            texts = self._model.decode_subtask_tokens(generated_tokens)
            generated_text = texts[0] if texts else None
            action_tokens = generated_tokens
        else:
            if held_memory_tokens is None:
                raise ValueError(f"chunk {chunk_index} is a fast tick but held_memory_tokens is empty")
            action_tokens = torch.as_tensor(held_memory_tokens, dtype=torch.int32, device=self._pytorch_device)
            if action_tokens.ndim == 1:
                action_tokens = action_tokens[None, ...]

        # Current-only action path: rebuild latest image/state/task without the
        # previous-memory text. The current raw IDs are the sole action plan.
        action_observation = make_observation(None)
        conditioned = self._model.build_hierarchical_observation(action_observation, action_tokens)
        mask = conditioned.subtask_mask
        if mask is None or not bool(torch.any(mask).item()):
            raise RuntimeError("held Memory tokens did not enter the action sequence (subtask mask is all false)")

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_t = torch.as_tensor(noise, device=self._pytorch_device)
            if noise_t.ndim == 2:
                noise_t = noise_t[None, ...]
            sample_kwargs["noise"] = noise_t
        actions = self._sample_actions(self._pytorch_device, conditioned, **sample_kwargs)
        raw_tokens = None if generated_tokens is None else np.asarray(generated_tokens[0].detach().cpu())
        outputs = self._output_transform(
            {"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0].detach().cpu())}
        )
        # Output transforms intentionally own only robot actions and may rebuild
        # the dict (B1kOutputs does). Runtime Memory is protocol metadata, so
        # attach it afterwards or it silently disappears before the wrapper can
        # commit a Planner result.
        outputs["held_memory_tokens"] = raw_tokens
        outputs["held_memory_text"] = generated_text
        return outputs

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        raw_prompt = obs.get("prompt")
        if raw_prompt is not None and not isinstance(raw_prompt, str):
            raw_prompt = str(raw_prompt)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        observation = _model.Observation.from_dict(inputs)

        generated_subtask = None
        if self._is_pytorch_model and hasattr(self._model, "predict_subtask_tokens") and hasattr(
            self._model, "build_hierarchical_observation"
        ):
            subtask_mask = getattr(observation, "subtask_mask", None)
            should_predict_subtask = subtask_mask is None or (not bool(torch.any(subtask_mask).item()))
        else:
            should_predict_subtask = False

        if should_predict_subtask:
            # The high-level subtask should depend on the current observation.
            # Caching by raw task prompt makes it effectively constant for the
            # whole episode because the prompt usually never changes.
            subtask_tokens = self._model.predict_subtask_tokens(observation)
            generated_texts = self._model.decode_subtask_tokens(subtask_tokens)
            generated_subtask = generated_texts[0] if generated_texts else None
            observation = self._model.build_hierarchical_observation(observation, subtask_tokens)

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if generated_subtask is None and hasattr(self._model, "_last_predicted_subtasks"):
            predicted = getattr(self._model, "_last_predicted_subtasks", None)
            if isinstance(predicted, list) and predicted:
                generated_subtask = predicted[0]
        if generated_subtask is not None:
            outputs["generated_subtask"] = generated_subtask
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def reset(self) -> None:
        """Forward the reset to the wrapped policy.

        Without this, wrapping a Policy in a PolicyRecorder silently disarms
        memory invalidation: PolicyRecorder would inherit BasePolicy's no-op
        reset, the inner Policy would never be told the episode ended, and the
        held memory would leak into the next episode with nothing reporting
        it. A guard has to be reachable from every consumer, not just the
        unwrapped one.
        """
        self._policy.reset()

    def infer_memory_chunk(self, obs: dict, **kwargs) -> dict:
        """Forward the stateless Memory chunk API without owning rollout state."""
        return self._policy.infer_memory_chunk(obs, **kwargs)

    @property
    def held_memory_enabled(self) -> bool:
        return bool(getattr(self._policy, "_held_memory_enabled", False))

    @property
    def model_config(self):
        return getattr(getattr(self._policy, "_model", None), "config", None)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
