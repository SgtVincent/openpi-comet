"""Tests for the hierarchy token cache and the prefix-KV staleness guard
(Hierarchical-MoMA-VLA P1, item 2).

Each test states what would make it fail.

The headline test here is
``test_reusing_prefix_kv_from_a_stale_observation_raises``: design section 4.5
warns that reusing a full prefix KV makes the Action Expert act on an old image
(and, for pi05, an old discretised state, because the state is text inside the
prefix). That risk is currently absent only because ``encode_prefix`` hardcodes
``past_key_values=None``; introducing a hierarchy cache is what brings it back.
These tests make CI catch it rather than leaving it as a paragraph in a doc.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from openpi.models.hierarchy_cache import HierarchyCacheError
from openpi.models.hierarchy_cache import HierarchyTokenCache
from openpi.models.hierarchy_cache import StalePrefixKVError
from openpi.models.hierarchy_cache import assert_prefix_fresh
from openpi.models.hierarchy_cache import observation_fingerprint
from openpi.models_pytorch.action_experts.subtask_expert import SubtaskActionExpert

# ---------------------------------------------------------------------------
# Fingerprint sensitivity -- without this, every guard test below is vacuous
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_the_image_changes():
    """FAILS IF: the fingerprint is insensitive to image content, which would
    make the staleness guard a no-op that still passes its own tests."""
    torch.manual_seed(0)
    img_a = [torch.randn(1, 3, 8, 8)]
    img_b = [torch.randn(1, 3, 8, 8)]
    lang = torch.arange(6).reshape(1, 6)
    assert observation_fingerprint(img_a, lang) != observation_fingerprint(img_b, lang)


def test_fingerprint_changes_when_the_state_changes():
    """FAILS IF: the fingerprint ignores language tokens.

    For pi05 the robot state is discretised into 256 bins and interpolated into
    the prompt STRING (tokenizer.py:504-507), so it reaches the prefix as
    language tokens. A fingerprint over images alone would let a stale state
    through undetected -- the part of the risk the design does not spell out.
    """
    img = [torch.zeros(1, 3, 8, 8)]
    lang_a = torch.tensor([[1, 2, 3, 4]])
    lang_b = torch.tensor([[1, 2, 3, 9]])
    assert observation_fingerprint(img, lang_a) != observation_fingerprint(img, lang_b)


def test_fingerprint_is_stable_for_the_same_observation():
    """FAILS IF: the fingerprint is nondeterministic, which would make the guard
    fire on correct code (a false alarm is as bad as a miss here)."""
    img = [torch.randn(1, 3, 8, 8)]
    lang = torch.arange(6).reshape(1, 6)
    assert observation_fingerprint(img, lang) == observation_fingerprint(img, lang)


# ---------------------------------------------------------------------------
# The required stale-prefix-KV test, exercised through the production method
# ---------------------------------------------------------------------------


class _StubModel(torch.nn.Module):
    """Minimal stand-in exposing only what encode_prefix / compute_velocity_infer touch.

    Deliberately a stub rather than a real PaliGemma: the thing under test is
    the fingerprint stamping and the guard, and a stub keeps this runnable on
    CPU in CI. The production methods themselves are the real ones.
    """

    def __init__(self, emb_dim: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.denoise_calls: list = []
        lm = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager"))
        self.paligemma_with_expert = types.SimpleNamespace(
            paligemma=types.SimpleNamespace(language_model=lm),
            embed_language_tokens=lambda toks: torch.zeros(*toks.shape, self.emb_dim),
            forward=self._fake_forward,
        )

    def _fake_forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
        # Stand-in for the real KV cache object.
        return None, {"fake_kv": inputs_embeds[0].shape}

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        b, n_lang = lang_tokens.shape
        n_img = 2 * len(images)
        n = n_img + n_lang
        embs = torch.zeros(b, n, self.emb_dim)
        pad = torch.ones(b, n, dtype=torch.bool)
        att = torch.zeros(b, n, dtype=torch.bool)
        return embs, pad, att

    def make_att_2d_masks(self, pad_masks, att_masks):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        return make_att_2d_masks(pad_masks, att_masks)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        return att_2d_masks[:, None, :, :]

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, time):
        self.denoise_calls.append(past_key_values)
        return torch.zeros_like(x_t)


def _obs(seed: int):
    torch.manual_seed(seed)
    images = [torch.randn(1, 3, 8, 8)]
    img_masks = [torch.ones(1, dtype=torch.bool)]
    lang_tokens = torch.randint(10, 100, (1, 5))
    lang_masks = torch.ones(1, 5, dtype=torch.bool)
    return images, img_masks, lang_tokens, lang_masks


def test_encode_prefix_stamps_the_observation_fingerprint():
    """FAILS IF: encode_prefix stops stamping obs_fingerprint, which would
    silently disarm the staleness guard everywhere (a guard with no input)."""
    expert = SubtaskActionExpert()
    model = _StubModel()
    images, img_masks, lang_tokens, lang_masks = _obs(0)
    ctx = expert.encode_prefix(
        model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
    )
    assert "obs_fingerprint" in ctx
    assert ctx["obs_fingerprint"] == observation_fingerprint(images, lang_tokens)


def test_reusing_prefix_kv_from_a_stale_observation_raises():
    """REQUIRED TEST -- design section 4.5 as an executable invariant.

    Encodes a prefix from observation A, then tries to run the action expert on
    observation B with that same prefix -- exactly what 'cache the full KV
    across chunks' looks like in code.

    FAILS IF: the guard is removed, the fingerprint stops being stamped or
    compared, or someone reintroduces cross-chunk prefix KV reuse. Then no
    exception is raised and the model silently acts on a stale image/state.
    """
    expert = SubtaskActionExpert()
    model = _StubModel()

    images_a, img_masks_a, lang_a, lang_masks_a = _obs(1)
    prefix_ctx = expert.encode_prefix(
        model=model, images=images_a, img_masks=img_masks_a, lang_tokens=lang_a, lang_masks=lang_masks_a
    )

    # A new chunk arrives with a fresh observation.
    images_b, _, lang_b, _ = _obs(2)
    fingerprint_b = observation_fingerprint(images_b, lang_b)

    with pytest.raises(StalePrefixKVError, match="different observation"):
        expert.compute_velocity_infer(
            model=model,
            prefix_ctx=prefix_ctx,  # <-- stale: built from observation A
            state=torch.zeros(1, 4),
            x_t=torch.zeros(1, 2, 4),
            time=torch.ones(1),
            expected_obs_fingerprint=fingerprint_b,
        )
    assert model.denoise_calls == [], "denoise_step ran despite the stale prefix"


def test_fresh_prefix_kv_is_accepted():
    """Negative control for the test above.

    FAILS IF: the guard is over-eager and rejects the correct path (recompute
    the prefix from the latest observation each chunk). Without this, a guard
    that raises unconditionally would pass the stale-KV test.
    """
    expert = SubtaskActionExpert()
    model = _StubModel()
    images, img_masks, lang_tokens, lang_masks = _obs(3)
    prefix_ctx = expert.encode_prefix(
        model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
    )
    out = expert.compute_velocity_infer(
        model=model,
        prefix_ctx=prefix_ctx,
        state=torch.zeros(1, 4),
        x_t=torch.zeros(1, 2, 4),
        time=torch.ones(1),
        expected_obs_fingerprint=observation_fingerprint(images, lang_tokens),
    )
    assert out.shape == (1, 2, 4)
    assert len(model.denoise_calls) == 1


def test_guard_is_a_noop_for_callers_that_do_not_opt_in():
    """FAILS IF: adding the guard breaks the pre-existing call signature.

    Every current caller omits expected_obs_fingerprint; they must keep working.
    """
    expert = SubtaskActionExpert()
    model = _StubModel()
    images, img_masks, lang_tokens, lang_masks = _obs(4)
    prefix_ctx = expert.encode_prefix(
        model=model, images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
    )
    expert.compute_velocity_infer(
        model=model,
        prefix_ctx=prefix_ctx,
        state=torch.zeros(1, 4),
        x_t=torch.zeros(1, 2, 4),
        time=torch.ones(1),
    )
    assert len(model.denoise_calls) == 1


def test_assert_prefix_fresh_message_names_the_remedy():
    """FAILS IF: the error stops telling the reader what to do instead.

    A guard that fires without naming the fix gets worked around rather than
    obeyed.
    """
    with pytest.raises(StalePrefixKVError, match="Cache the hierarchy token ids instead"):
        assert_prefix_fresh({"obs_fingerprint": ("a",)}, ("b",))


# ---------------------------------------------------------------------------
# HierarchyTokenCache -- what it accepts, and its invalidate semantics
# ---------------------------------------------------------------------------


def test_cache_refuses_a_kv_payload():
    """FAILS IF: the cache accepts a past_key_values object.

    This is the design section 4.5 mistake at its source: caching a KV rather
    than token ids is what carries a stale observation forward.
    """

    class DynamicCache:  # name deliberately matches the HF class
        pass

    cache = HierarchyTokenCache()
    with pytest.raises(HierarchyCacheError, match="token ids only"):
        cache.store(DynamicCache())


def test_cache_refuses_float_tensors():
    """FAILS IF: embeddings or KV tensors can be stored as if they were tokens."""
    cache = HierarchyTokenCache()
    with pytest.raises(HierarchyCacheError, match="non-integer"):
        cache.store(torch.zeros(4, 8, dtype=torch.float32))
    with pytest.raises(HierarchyCacheError, match="non-integer"):
        cache.store(np.zeros(4, dtype=np.float32))


def test_cache_accepts_integer_tokens_from_torch_and_numpy():
    """FAILS IF: the type guard is so strict it rejects legitimate token arrays."""
    cache = HierarchyTokenCache()
    cache.store(torch.tensor([1, 2, 3]))
    assert cache.is_populated
    cache.store(np.asarray([1, 2, 3], dtype=np.int32))
    cache.store([1, 2, 3])
    assert cache.generation == 3


def test_cache_refuses_none_and_empty():
    """FAILS IF: an empty Planner generation silently wipes the held hierarchy.

    The design's degradation rule is 'keep the previous tokens when generation
    is empty'. That must be the caller's explicit choice, not something the
    cache does silently in either direction.
    """
    cache = HierarchyTokenCache()
    with pytest.raises(HierarchyCacheError, match="refusing to store None"):
        cache.store(None)
    with pytest.raises(HierarchyCacheError, match="empty token array"):
        cache.store([])


def test_cold_cache_read_raises_and_names_the_reason():
    """FAILS IF: reading an unpopulated cache returns None and lets a caller act
    on 'no hierarchy' without noticing."""
    cache = HierarchyTokenCache()
    with pytest.raises(HierarchyCacheError, match="never populated"):
        cache.get()
    assert cache.get_or_none() is None  # opt-in non-raising read still available


def test_invalidate_requires_a_reason_and_is_observable():
    """FAILS IF: invalidation becomes silent. An unexplained invalidation in a
    rollout log cannot be told apart from a bug."""
    cache = HierarchyTokenCache()
    cache.store([1, 2, 3], text="<MEM>m</MEM>")
    with pytest.raises(HierarchyCacheError, match="non-empty reason"):
        cache.invalidate("")
    cache.invalidate("planner timeout")
    assert not cache.is_populated
    assert cache.invalidated_reason == "planner timeout"
    with pytest.raises(HierarchyCacheError, match="planner timeout"):
        cache.get()


def test_store_overwrites_and_bumps_generation():
    """FAILS IF: the 'commit is an overwrite' semantics of design section 2.3
    changes, or generation stops tracking Planner ticks."""
    cache = HierarchyTokenCache()
    cache.store([1, 2], text="first")
    cache.store([3, 4, 5], text="second")
    tokens, _ = cache.get()
    assert list(tokens) == [3, 4, 5]
    assert cache.text == "second"
    assert cache.generation == 2


def test_policy_reset_clears_held_hierarchy():
    """FAILS IF: held hierarchy survives an episode boundary.

    A hierarchy carried into a new episode is stale by construction and nothing
    downstream would notice.
    """
    from openpi.policies.policy import Policy

    policy = Policy.__new__(Policy)  # bypass model construction; only cache wiring is under test
    policy._hierarchy_cache = HierarchyTokenCache()
    policy._cached_subtask_prompt = "stale"
    policy._hierarchy_cache.store([1, 2, 3], text="<MEM>m</MEM>")
    assert policy._cached_subtask_tokens is not None

    policy.reset()
    assert policy._cached_subtask_tokens is None
    assert policy._cached_subtask_text is None
    assert policy._cached_subtask_prompt is None
    assert policy._hierarchy_cache.invalidated_reason == "episode boundary"
