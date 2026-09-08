#!/usr/bin/env python3
"""Pre-launch weight-loading gate (CPU-only, read-only, re-runnable).

WHY THIS EXISTS
---------------
``scripts/train_accelerate.py`` warm-starts a fine-tune like this (~line 4941)::

    if not os.path.exists(model_path):
        logging.warning("Model checkpoint not found at %s. Skipping weight loading.", model_path)
    else:
        load_strict = config.pytorch_model_name not in (
            "pi05_ki_joint_fast", "vlm2", "vlm2_subtask", "subtask",
            "pi0_hamlet", "pi0_memoryvla", "pi05_ki_joint_query",
        )
        safetensors.torch.load_model(model, model_path, strict=load_strict)
        logging.info("Loaded PyTorch weights from %s", config.pytorch_weight_path)

Three independent ways that hides a broken run:

1. Every ``pi05_ki_joint_*`` config -- i.e. both historical 4x8 H20 runs -- is in
   the exclusion list, so ``strict`` is **False**.
2. The ``(missing, unexpected)`` tuple that ``load_model`` returns is discarded,
   and the "Loaded PyTorch weights from ..." line prints unconditionally. A load
   that matched nothing and a load that matched everything emit byte-identical
   logs.
3. If ``model.safetensors`` is absent the code only warns and continues, so a
   fully randomly-initialised run trains and produces normal-looking curves.

This module is the gate that refuses to let such a launch happen. It is a
separate, additive artifact: it does not modify ``train_accelerate.py``.

WHAT IT ENFORCES (all must hold for a PASS)
-------------------------------------------
* the config name is actually **registered** -- ``train_config.get_config()``
  silently falls back to ``pi05_b1k-base`` for an unknown name, which would
  otherwise gate the wrong config (see ``train_config.py:325-330``);
* ``pytorch_weight_path`` is configured, and ``<path>/model.safetensors``
  exists as a regular file; its absolute path, byte size and md5 are printed;
* the load is performed with ``strict=True`` **explicitly** -- never inheriting
  the exclusion-list default;
* ``missing_keys`` and ``unexpected_keys`` are both empty, and no tensor has a
  mismatched shape;
* every float parameter/persistent buffer was actually overwritten. Before
  loading, each is poisoned with NaN; anything still all-NaN afterwards was left
  at initialisation regardless of what the key bookkeeping claimed;
* a random sample of tensors in the loaded model is byte-compared against the
  tensors re-read from the very file that was opened, so "the log said loaded"
  is backed by "the values came from this inode".

If ``strict=True`` legitimately fails on a real checkpoint, this module does
**not** relax anything: it refuses, prints the exact key names, their counts and
their module prefixes, and leaves the allow/deny decision to a human. There is
deliberately no ``--relax``/``--strict=False`` escape hatch, and no command-line
allowlist either -- a flag any launcher could pass would just recreate the
implicit exclusion list somewhere new. The only way to permit an absent key is
``REGISTERED_ALLOWLIST`` below: per-config, per-key, in source, reviewable, and
required to carry a justification. It is currently **empty**, because no such
decision has been made.

VERDICT CONTRACT (how a launcher consumes this)
-----------------------------------------------
Exactly one line is printed to stdout whose first field is the token
``WEIGHT_LOAD_GATE_VERDICT``::

    WEIGHT_LOAD_GATE_VERDICT <STATUS> reason=<TOKEN> key=value ...

STATUS/exit code pairs:

===============  ====  ==========================================================
STATUS           exit  meaning
===============  ====  ==========================================================
PASS                0  a real strict load succeeded and every check above held
REFUSE              2  a check failed -- do not launch
INCONCLUSIVE        4  NOT MEASURED (e.g. ``--load-mode header``: no tensor data
                       was read, so nothing is proven about the payload)
ERROR               3  the gate itself could not run to a decision
===============  ====  ==========================================================

A caller must require **both** ``exit == 0`` **and** the presence of a stdout
line starting with ``WEIGHT_LOAD_GATE_VERDICT PASS``. Requiring the token as
well as the exit code is what distinguishes "measured zero problems" from
"killed before it measured anything" -- an empty output is INCONCLUSIVE, not a
pass. Example::

    out=$(python scripts/hier/preflight_weight_load.py --config "$CFG") ; rc=$?
    echo "$out"
    if [ "$rc" -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^WEIGHT_LOAD_GATE_VERDICT PASS'; then
        echo "REFUSING TO LAUNCH" >&2 ; exit 1
    fi

(Note the ``rc=$?`` is read immediately after the assignment, not after a pipe.)

USAGE
-----
::

    # gate a registered config (resolves the weight source from the config)
    python scripts/hier/preflight_weight_load.py \
        --config pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16

    # gate a config against an explicitly overridden weight directory
    python scripts/hier/preflight_weight_load.py --config <name> --weight-dir <dir>

    # cheap key-only triage (never returns PASS; exit 4 == NOT MEASURED)
    python scripts/hier/preflight_weight_load.py --config <name> --load-mode header

Always run with ``PYTHONPATH=<this worktree>/src``; ``openpi`` is an editable
install pinned to a different tree and without the override the gate would
inspect the wrong source. ``--require-openpi-under`` asserts this for you.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

VERDICT_TOKEN = "WEIGHT_LOAD_GATE_VERDICT"

EXIT_PASS = 0
EXIT_REFUSE = 2
EXIT_ERROR = 3
EXIT_INCONCLUSIVE = 4

_STATUS_EXIT = {
    "PASS": EXIT_PASS,
    "REFUSE": EXIT_REFUSE,
    "ERROR": EXIT_ERROR,
    "INCONCLUSIVE": EXIT_INCONCLUSIVE,
}

# The two weight packages this project actually warm-starts from. Sizes are
# measured facts, used only to *label* what was opened -- never to decide the
# verdict, so an unknown source is reported as ``unknown`` rather than refused.
KNOWN_SOURCES: dict[str, dict[str, Any]] = {
    "pi05_base_pytorch": {
        "size_bytes": 7233650408,
        "note": "pi05 pretrain, safetensors only, BF16, ~7.23 GB",
    },
    "pi05-b1kpt50-cs32": {
        "size_bytes": 14467165872,
        "note": "Comet B1K fine-tune, safetensors F32 ~14.47 GB + unused orbax params/ ~12.44 GB",
    },
}

_PARTIAL_HASH_WINDOW = 64 * 1024 * 1024  # 64 MiB head + 64 MiB tail


# --------------------------------------------------------------------------- #
# THE registered allowlist
# --------------------------------------------------------------------------- #
# Maps a *registered config name* to the exact state-dict keys that a human has
# decided may legitimately be absent from that config's checkpoint, each with a
# justification. This is the only permitted way to tolerate a missing key.
#
# It is deliberately NOT a command-line flag and NOT prefix-based: an implicit,
# broad exclusion list is precisely the defect this gate exists to stop, and
# reproducing it as `--allow-missing some.prefix.` would move the problem rather
# than fix it. Adding an entry is a reviewed source change.
#
# Empty on purpose. As measured on both candidate checkpoints, nothing needs to
# be here: tied parameters (``paligemma.lm_head.weight`` <-> embedding) are
# resolved by ``load_model`` via the file's ``__metadata__`` and are reported as
# TIED, not MISSING, so they never required an allowlist entry.
#
# Format: {config_name: {state_dict_key: "why this may be absent"}}
# The formal MoMA checkpoint was measured against the actual training model:
# 813 logical state_dict keys = 812 physical tensors + one tied alias, with
# strict=True returning missing=[] and unexpected=[]. Therefore its exact
# allowlist is intentionally empty. Keeping the config entry explicit makes that
# measurement a reviewable contract: a future missing key cannot be mistaken for
# "no policy was registered" and silently broadened.
REGISTERED_ALLOWLIST: dict[str, dict[str, str]] = {
    "pi05_moma_memory_b1k-k5": {},
    "pi05_moma_memory_b1k-k5_smoke": {},
    "pi05_moma_memory_b1k-k1-short": {},
    "pi05_moma_memory_b1k-mix-c-short": {},
    "pi05_moma_memory_b1k-mix-c": {},
}


def allowlist_for(config_name: str) -> tuple[str, ...]:
    """Exact keys a human has registered as permitted-absent for this config."""
    return tuple(sorted(REGISTERED_ALLOWLIST.get(config_name, {})))


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KeyReport:
    """Classification of model keys against checkpoint keys."""

    matched: list[str] = dataclasses.field(default_factory=list)
    tied: list[str] = dataclasses.field(default_factory=list)
    missing: list[str] = dataclasses.field(default_factory=list)
    unexpected: list[str] = dataclasses.field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = dataclasses.field(
        default_factory=list
    )
    allowlisted_missing: list[str] = dataclasses.field(default_factory=list)
    #: every absent key, allowlisted or not -- what an un-allowlisted strict load sees
    raw_missing: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GateResult:
    status: str
    reason: str
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)
    keys: KeyReport = dataclasses.field(default_factory=KeyReport)
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return _STATUS_EXIT[self.status]

    def verdict_line(self) -> str:
        parts = [VERDICT_TOKEN, self.status, f"reason={self.reason}"]
        for key, value in self.fields.items():
            parts.append(f"{key}={_scalar(value)}")
        return " ".join(parts)


def _scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value)
    # The verdict line is whitespace-delimited key=value; never emit a space.
    return text.replace(" ", "_") if " " in text else text


# --------------------------------------------------------------------------- #
# weight source resolution
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class WeightSource:
    configured: str | None
    cwd: str
    directory: str | None = None
    file: str | None = None
    realpath: str | None = None
    exists: bool = False
    is_file: bool = False
    size_bytes: int | None = None
    st_dev: int | None = None
    st_ino: int | None = None
    is_relative: bool = False
    source_id: str = "unknown"
    source_note: str | None = None
    size_matches_known: bool | None = None
    sibling_orbax_params: bool = False


def resolve_weight_source(configured: str | None, *, override_dir: str | None = None) -> WeightSource:
    """Resolve ``<pytorch_weight_path>/model.safetensors`` the way training does.

    ``train_accelerate.py`` uses a bare ``os.path.join`` + ``os.path.exists``, so a
    *relative* ``pytorch_weight_path`` (most registered configs use one) resolves
    against the launch directory. That is recorded here explicitly: ``cwd`` and
    ``is_relative`` are part of the report because the same config can resolve to
    a different file depending on where it was launched from.
    """
    cwd = os.getcwd()
    chosen = override_dir if override_dir is not None else configured
    src = WeightSource(configured=configured, cwd=cwd)
    if chosen is None:
        return src
    src.is_relative = not os.path.isabs(chosen)
    src.directory = os.path.abspath(chosen)
    # Exactly the join train_accelerate.py performs, then made absolute.
    src.file = os.path.abspath(os.path.join(chosen, "model.safetensors"))
    src.exists = os.path.exists(src.file)
    src.is_file = os.path.isfile(src.file)
    if src.is_file:
        stat = os.stat(src.file)
        src.size_bytes = int(stat.st_size)
        src.st_dev = int(stat.st_dev)
        src.st_ino = int(stat.st_ino)
        src.realpath = os.path.realpath(src.file)
    basename = os.path.basename(src.directory.rstrip("/"))
    if basename in KNOWN_SOURCES:
        src.source_id = basename
        src.source_note = KNOWN_SOURCES[basename]["note"]
        if src.size_bytes is not None:
            src.size_matches_known = src.size_bytes == KNOWN_SOURCES[basename]["size_bytes"]
    src.sibling_orbax_params = os.path.isdir(os.path.join(src.directory, "params"))
    return src


def hash_file(path: str, mode: str) -> tuple[str, str, int]:
    """Return ``(algo_label, hexdigest, bytes_covered)``.

    ``mode="full"``  -> real md5 over the whole file; label ``md5`` (legacy).
    ``mode="sha256"`` -> real SHA-256 over the whole file; label ``sha256``.
    ``mode="partial"`` -> md5 over the first 64 MiB, the last 64 MiB and the
        decimal file size. This is **not** an md5 of the file; the label spells out
        exactly which byte ranges went in, e.g.
        ``md5_partial:head=0..67108864,tail=7166541544..7233650408,plus_size_string``.
    ``mode="none"`` -> no hashing; label ``none``, digest ``-``.

    ``full`` is the default because it is affordable here: measured NAS read is
    ~185 MB/s cold, i.e. ~50 s for the 7.23 GB file and ~72 s for the 14.47 GB one.
    """
    if mode == "none":
        return "none", "-", 0
    size = os.path.getsize(path)
    if mode == "sha256":
        digest = hashlib.sha256()
        covered = 0
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                covered += len(chunk)
        return "sha256", digest.hexdigest(), covered
    digest = hashlib.md5()
    covered = 0
    if mode == "full":
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                covered += len(chunk)
        return "md5", digest.hexdigest(), covered
    if mode == "partial":
        window = min(_PARTIAL_HASH_WINDOW, size)
        tail_start = max(window, size - window)  # never re-hash the head range
        with open(path, "rb") as handle:
            head = handle.read(window)
            digest.update(head)
            covered += len(head)
            tail_len = 0
            if tail_start < size:
                handle.seek(tail_start)
                tail = handle.read(size - tail_start)
                digest.update(tail)
                tail_len = len(tail)
                covered += tail_len
        digest.update(str(size).encode())
        label = (
            f"md5_partial:head=0..{len(head)},"
            f"tail={tail_start}..{tail_start + tail_len},plus_size_string"
        )
        return label, digest.hexdigest(), covered
    raise ValueError(f"unknown hash mode {mode!r}")


def read_safetensors_header(path: str) -> tuple[dict[str, tuple[int, ...]], dict[str, str], dict[str, str]]:
    """Read ``({key: shape}, {key: dtype}, metadata)`` from the header only.

    No tensor payload is touched. ``metadata`` is the file's ``__metadata__``
    dict; ``safetensors.torch.save_model`` records tied/shared tensors there
    (absent name -> kept name), and ``load_model`` uses it to repopulate them.
    """
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = handle.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(header_bytes)
    metadata = header.get("__metadata__") or {}
    shapes = {}
    dtypes = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        shapes[key] = tuple(value["shape"])
        dtypes[key] = str(value.get("dtype"))
    return shapes, dtypes, metadata


# --------------------------------------------------------------------------- #
# key classification -- mirrors safetensors.torch.load_model exactly
# --------------------------------------------------------------------------- #
def classify_keys(
    model: Any,
    ckpt_shapes: dict[str, tuple[int, ...]],
    *,
    allow_missing: tuple[str, ...] = (),
) -> KeyReport:
    """Classify model vs checkpoint keys with ``load_model``'s own tie handling.

    ``safetensors.torch.load_model`` (safetensors/torch.py, ``load_model``) does::

        to_removes = _remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        ... move tied names out of `missing` / into `unexpected` ...
        if strict and (missing or unexpected): raise

    so the *same* private helper is used here rather than re-deriving tie
    resolution from ``__metadata__``. Shape mismatches are collected separately
    because ``load_state_dict`` raises on those unconditionally, outside the
    ``if strict:`` block (torch/nn/modules/module.py), i.e. ``strict=False``
    never suppressed them.
    """
    from safetensors.torch import _remove_duplicate_names

    model_sd = model.state_dict()
    model_shapes = {name: tuple(tensor.shape) for name, tensor in model_sd.items()}
    file_keys = set(ckpt_shapes)
    model_keys = set(model_shapes)

    to_removes = _remove_duplicate_names(model_sd, preferred_names=list(file_keys))

    missing = model_keys - file_keys
    unexpected = list(file_keys - model_keys)
    tied: list[str] = []
    for group in to_removes.values():
        for name in group:
            if name not in missing:
                unexpected.append(name)
            else:
                missing.remove(name)
                tied.append(name)

    shared = sorted(model_keys & file_keys)
    matched = [k for k in shared if model_shapes[k] == ckpt_shapes[k]]
    mismatch = [(k, model_shapes[k], ckpt_shapes[k]) for k in shared if model_shapes[k] != ckpt_shapes[k]]

    allowlisted = sorted(k for k in missing if k in set(allow_missing))
    remaining_missing = sorted(k for k in missing if k not in set(allowlisted))

    return KeyReport(
        matched=matched,
        tied=sorted(tied),
        missing=remaining_missing,
        unexpected=sorted(unexpected),
        shape_mismatch=sorted(mismatch),
        allowlisted_missing=allowlisted,
        raw_missing=sorted(missing),
    )


def prefix_histogram(keys: list[str], depth: int = 2, limit: int = 15) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for key in keys:
        counts.setdefault(".".join(key.split(".")[:depth]), 0)
        counts[".".join(key.split(".")[:depth])] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


# --------------------------------------------------------------------------- #
# poison / population detection
# --------------------------------------------------------------------------- #
def poison_state(model: Any) -> tuple[int, int]:
    """Fill every float tensor in ``state_dict()`` with NaN.

    Returns ``(poisoned, skipped)``. Non-float tensors cannot hold NaN and are
    skipped (they are still covered by the key checks). This is what turns "the
    log said loaded" into a falsifiable claim: anything still all-NaN after the
    load was never written, whatever the key bookkeeping reported.
    """
    import torch

    poisoned = skipped = 0
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.detach().fill_(float("nan"))
                poisoned += 1
            else:
                skipped += 1
    return poisoned, skipped


def unpopulated_after_load(model: Any) -> list[str]:
    """Names of float tensors that are still entirely NaN, i.e. never written."""
    import torch

    stale: list[str] = []
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if tensor.is_floating_point() and tensor.numel() and bool(torch.isnan(tensor).all()):
                stale.append(name)
    return sorted(stale)


def sample_verify(model: Any, path: str, keys: list[str], count: int, seed: int = 0) -> tuple[int, list[str]]:
    """Re-read tensors from ``path`` and compare against the model.

    Proves the values in memory came from *this* file rather than from
    initialisation. ``count < 0`` verifies every key (a second full read of the
    file); otherwise a deterministic ``seed``-fixed sample of ``count`` keys is
    used so repeat runs are comparable.

    Comparison is against ``file_tensor.to(param.dtype)`` because
    ``load_state_dict`` casts on copy (an F32 checkpoint into a BF16 model is a
    legitimate downcast, not a mismatch).
    """
    from safetensors import safe_open
    import torch

    if count == 0 or not keys:
        return 0, []
    picked = list(keys) if count < 0 else random.Random(seed).sample(keys, min(count, len(keys)))
    model_sd = model.state_dict()
    bad: list[str] = []
    ok = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in picked:
            want = handle.get_tensor(key)
            got = model_sd[key]
            if tuple(want.shape) != tuple(got.shape) or not torch.equal(got, want.to(got.dtype)):
                bad.append(key)
            else:
                ok += 1
    return ok, bad


# --------------------------------------------------------------------------- #
# model construction -- mirrors scripts/train_accelerate.py
# --------------------------------------------------------------------------- #
def build_model_for_config(config: Any) -> Any:
    """Build the same nn.Module ``train_accelerate.py`` would build.

    Mirrors ``scripts/train_accelerate.py`` "Build model" block (~4781-4901),
    including the ``object.__setattr__(model_cfg, "dtype", ...)`` so the gate
    exercises the dtype the run will actually use. Model modules are imported
    lazily and *outside* any device context, because importing HuggingFace
    submodules under ``torch.device("meta")`` breaks their module-level
    initialisation and the failure is then cached in ``sys.modules``.
    """
    import openpi.models.pi0_config as pi0_config
    import openpi.models.pi05_subtask_config as pi05_subtask_config
    import openpi.models.vlm2_vla_config as vlm2_vla_config

    if isinstance(config.model, vlm2_vla_config.VLM2VLAConfig | pi05_subtask_config.Pi05SubtaskConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    elif not isinstance(config.model, pi0_config.Pi0Config):
        model_cfg = pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    name = config.pytorch_model_name
    if name in ("vlm2", "vlm2_subtask"):
        import openpi.models_pytorch.vlm2.vlm2_model as vlm2_model

        vlm2_config = vlm2_model.VLM2Config(
            visual_dim=2048,
            geometry_dim=config.vlm2_geometry_dim,
            view_dim=config.vlm2_view_dim,
            working_memory_size=config.vlm2_working_memory_size,
            episodic_memory_capacity=config.vlm2_episodic_memory_capacity,
            episodic_similarity_threshold=config.vlm2_episodic_similarity_threshold,
            episodic_fusion_alpha=config.vlm2_episodic_fusion_alpha,
            sem_geo_fusion_tanh_gate_enable=config.vlm2_sem_geo_fusion_tanh_gate_enable,
            sem_geo_fusion_tanh_gate_init_alpha=config.vlm2_sem_geo_fusion_tanh_gate_init_alpha,
            num_heads=8,
            hidden_dim=1024,
            dropout=0.0,
            pi05=True,
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            dtype=config.pytorch_training_precision,
            paligemma_variant=model_cfg.paligemma_variant,
            action_expert_variant=model_cfg.action_expert_variant,
            num_frames=config.vlm2_num_frames,
            frame_height=224,
            frame_width=224,
            patch_size=16,
            vggt_pretrained=getattr(model_cfg, "vggt_pretrained", None),
            vggt_load_strict=getattr(model_cfg, "vggt_load_strict", False),
            vggt_enable_track=getattr(model_cfg, "vggt_enable_track", False),
            freeze_vggt_backbone=getattr(model_cfg, "freeze_vggt_backbone", False),
            freeze_image_encoder=getattr(model_cfg, "freeze_image_encoder", False),
        )
        if name == "vlm2_subtask":
            return vlm2_model.VLM2SubtaskWithPi05(vlm2_config, alpha=getattr(model_cfg, "alpha", 10.0))
        return vlm2_model.VLM2WithPi05(vlm2_config)
    if name == "pi0_hamlet":
        import openpi.models_pytorch.pi0_hamlet as pi0_hamlet

        return pi0_hamlet.Pi05WithHamlet(model_cfg)
    if name == "pi0_memoryvla":
        import openpi.models_pytorch.pi0_memoryvla as pi0_memoryvla

        return pi0_memoryvla.Pi05WithMemoryVLA(model_cfg)
    if name == "subtask":
        import openpi.models_pytorch.pi05_subtask as pi05_subtask

        return pi05_subtask.PI05SubtaskPytorch(
            model_cfg,
            alpha=getattr(model_cfg, "alpha", 10.0),
            ce_weight=getattr(model_cfg, "ce_weight", 1.0),
            action_expert_name="subtask",
        )
    if name == "pi05_ki_joint_fast":
        import openpi.models_pytorch.pi05_ki_joint_fast as pi05_ki_joint_fast

        return pi05_ki_joint_fast.PI05KIJointFastPytorch(model_cfg)
    if name == "pi05_ki_joint_query":
        import openpi.models_pytorch.pi05_ki_joint_query as pi05_ki_joint_query

        return pi05_ki_joint_query.PI05KIJointQueryPytorch(model_cfg)
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

    return pi0_pytorch.PI0Pytorch(model_cfg)


def load_registered_config(config_name: str) -> Any:
    """Fetch a config, refusing ``get_config``'s silent fallback.

    ``train_config.get_config`` logs a warning and returns ``pi05_b1k-base`` for
    an unknown name (``train_config.py:325-330``). Gating the wrong config is
    exactly the class of silent substitution this module exists to stop, so an
    unregistered name is an error here.
    """
    from openpi.training.train_config import _CONFIGS_DICT

    if config_name not in _CONFIGS_DICT:
        raise KeyError(config_name)
    return _CONFIGS_DICT[config_name]


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def run_gate(
    *,
    model_factory: Callable[[], Any],
    configured_weight_path: str | None,
    override_dir: str | None = None,
    config_name: str = "-",
    model_name: str = "-",
    load_mode: str = "real",
    hash_mode: str = "full",
    verify_sample: int = 8,
    poison_init: bool = True,
    allow_missing: tuple[str, ...] = (),
    expect_size: int | None = None,
    expect_hash: str | None = None,
    max_key_examples: int = 12,
    out: Any = None,
) -> GateResult:
    """Run the gate. Returns a :class:`GateResult`; never raises for a policy failure.

    ``load_mode``:
      * ``real``   -- ``safetensors.torch.load_model(model, path, strict=True)``.
      * ``stream`` -- same enforcement, tensors copied one at a time via
        ``safe_open`` so peak RSS is model + largest tensor instead of
        model + whole state dict. Equivalent verdicts (see the equivalence test).
      * ``header`` -- key/shape triage only, no payload read. Cannot PASS;
        returns INCONCLUSIVE because nothing about the payload was measured.
    """
    out = out if out is not None else sys.stdout
    t0 = time.time()

    def emit(text: str = "") -> None:
        print(text, file=out, flush=True)

    fields: dict[str, Any] = {
        "config": config_name,
        "model_name": model_name,
        "load_mode": load_mode,
        "strict": True,
    }

    emit("=" * 78)
    emit("PRE-LAUNCH WEIGHT LOAD GATE")
    emit(f"  config              : {config_name}")
    emit(f"  pytorch_model_name  : {model_name}")
    emit(f"  load_mode           : {load_mode}   (strict=True, explicit -- no exclusion list)")
    emit(f"  cwd                 : {os.getcwd()}")

    # ---- 1. resolve + describe the weight file ----------------------------- #
    src = resolve_weight_source(configured_weight_path, override_dir=override_dir)
    fields["source_id"] = src.source_id
    emit(f"  configured path     : {src.configured!r}")
    if override_dir is not None:
        emit(f"  OVERRIDE --weight-dir: {override_dir!r}  (config value ignored)")
    if src.directory is None:
        emit("  RESULT              : pytorch_weight_path is None -> nothing would be loaded")
        result = GateResult(
            status="REFUSE",
            reason="WEIGHT_PATH_NOT_CONFIGURED",
            fields={**fields, "weight_file": None, "elapsed_s": round(time.time() - t0, 2)},
        )
        emit(result.verdict_line())
        return result

    emit(f"  resolved dir        : {src.directory}")
    emit(f"  resolved file       : {src.file}")
    emit(f"  path was relative   : {src.is_relative}  (relative paths resolve against cwd above)")
    emit(f"  exists / is_file    : {src.exists} / {src.is_file}")
    fields["weight_file"] = src.file

    if not src.is_file:
        emit("")
        emit("  REFUSED: the resolved weight file does not exist as a regular file.")
        emit("  train_accelerate.py would only log 'Skipping weight loading' here and")
        emit("  train from random initialisation.")
        result = GateResult(
            status="REFUSE",
            reason="WEIGHT_FILE_MISSING" if not src.exists else "WEIGHT_FILE_NOT_A_FILE",
            fields={**fields, "size_bytes": None, "elapsed_s": round(time.time() - t0, 2)},
            detail={"weight_source": dataclasses.asdict(src)},
        )
        emit(result.verdict_line())
        return result

    emit(f"  realpath            : {src.realpath}")
    emit(f"  size_bytes          : {src.size_bytes}")
    emit(f"  st_dev / st_ino     : {src.st_dev} / {src.st_ino}")
    emit(f"  known source        : {src.source_id}  ({src.source_note or 'not a registered source'})")
    if src.size_matches_known is not None:
        emit(f"  size matches source : {src.size_matches_known}")
    emit(f"  sibling orbax params/: {src.sibling_orbax_params}  (unused by the safetensors path)")

    t_hash = time.time()
    algo, digest, covered = hash_file(src.file, hash_mode)
    emit(f"  hash_algo           : {algo}")
    emit(f"  hash                : {digest}")
    emit(f"  hash bytes covered  : {covered} of {src.size_bytes}  ({round(time.time() - t_hash, 2)}s)")
    fields.update({"size_bytes": src.size_bytes, "hash_algo": algo, "hash": digest})

    if expect_size is not None and src.size_bytes != expect_size:
        emit(f"\n  REFUSED: size {src.size_bytes} != --expect-size {expect_size}")
        result = GateResult(
            status="REFUSE",
            reason="SIZE_MISMATCH",
            fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
        )
        emit(result.verdict_line())
        return result
    if expect_hash is not None and digest != expect_hash:
        emit(f"\n  REFUSED: hash {digest} != --expect-hash {expect_hash}")
        result = GateResult(
            status="REFUSE",
            reason="HASH_MISMATCH",
            fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
        )
        emit(result.verdict_line())
        return result

    # ---- 2. build the model, classify keys -------------------------------- #
    emit("-" * 78)
    t_build = time.time()
    model = model_factory()
    emit(f"  model built         : {type(model).__name__} in {round(time.time() - t_build, 2)}s")

    try:
        ckpt_shapes, ckpt_dtypes, ckpt_meta = read_safetensors_header(src.file)
    except Exception as exc:
        emit(f"\n  REFUSED: cannot parse safetensors header: {exc!r}")
        result = GateResult(
            status="REFUSE",
            reason="HEADER_UNREADABLE",
            fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
            detail={"exception": repr(exc)},
        )
        emit(result.verdict_line())
        return result

    keys = classify_keys(model, ckpt_shapes, allow_missing=allow_missing)
    model_sd_names = list(model.state_dict().keys())
    emit(f"  model tensors       : {len(model_sd_names)}")
    emit(f"  ckpt  tensors       : {len(ckpt_shapes)}")
    emit(f"  ckpt tie entries    : {len(ckpt_meta)}  (__metadata__ aliases from save_model dedup)")
    emit(f"  MATCHED             : {len(keys.matched)}")
    emit(f"  TIED (resolvable)   : {len(keys.tied)}")
    emit(f"  MISSING             : {len(keys.missing)}")
    emit(f"  UNEXPECTED          : {len(keys.unexpected)}")
    emit(f"  SHAPE_MISMATCH      : {len(keys.shape_mismatch)}")
    emit(f"  ALLOWLISTED MISSING : {len(keys.allowlisted_missing)}  (REGISTERED_ALLOWLIST, reviewed in source)")

    fields.update(
        {
            "missing": len(keys.missing),
            "unexpected": len(keys.unexpected),
            "shape_mismatch": len(keys.shape_mismatch),
            "tied": len(keys.tied),
            "allowlisted": len(keys.allowlisted_missing),
        }
    )

    if keys.allowlisted_missing:
        emit("\n  --- ALLOWLISTED MISSING (a registered, per-key human decision) ---")
        for key in keys.allowlisted_missing[:max_key_examples]:
            emit(f"      {key}")

    def dump(label: str, items: list[Any]) -> None:
        if not items:
            return
        names = [i if isinstance(i, str) else i[0] for i in items]
        emit(f"\n  --- {label} ({len(items)}) by module prefix ---")
        for prefix, count in prefix_histogram(names):
            emit(f"      {count:>6}  {prefix}.*")
        emit(f"  first {min(max_key_examples, len(items))} of {len(items)}:")
        for item in items[:max_key_examples]:
            if isinstance(item, str):
                emit(f"      {item}")
            else:
                emit(f"      {item[0]}  model={item[1]} ckpt={item[2]}")

    dump("MISSING KEYS (model expects, checkpoint lacks)", keys.missing)
    dump("UNEXPECTED KEYS (checkpoint has, model lacks)", keys.unexpected)
    dump("SHAPE MISMATCH", keys.shape_mismatch)

    if keys.missing or keys.unexpected or keys.shape_mismatch:
        reason = (
            "SHAPE_MISMATCH"
            if keys.shape_mismatch
            else "MISSING_AND_UNEXPECTED_KEYS"
            if (keys.missing and keys.unexpected)
            else "MISSING_KEYS"
            if keys.missing
            else "UNEXPECTED_KEYS"
        )
        emit("")
        emit("  REFUSED: strict=True cannot be satisfied. NOT relaxing the check and NOT")
        emit("  adding a blanket allowlist -- whether any of these keys may legitimately")
        emit("  be absent is a human decision that must be recorded per key in")
        emit("  REGISTERED_ALLOWLIST (reviewed source), never as an implicit exclusion")
        emit("  list or a command-line flag. Escalate the key names above.")
        result = GateResult(
            status="REFUSE",
            reason=reason,
            fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
            keys=keys,
            detail={"weight_source": dataclasses.asdict(src), "ckpt_dtypes_sample": _dtype_hist(ckpt_dtypes)},
        )
        emit(result.verdict_line())
        return result

    if load_mode == "header":
        emit("")
        emit("  NOT MEASURED: --load-mode header read no tensor payload, so nothing is")
        emit("  proven about the values. Key/shape triage passed; that is not a PASS.")
        result = GateResult(
            status="INCONCLUSIVE",
            reason="HEADER_ONLY_NOT_MEASURED",
            fields={**fields, "populated": "-", "sampled": "-", "elapsed_s": round(time.time() - t0, 2)},
            keys=keys,
            detail={"weight_source": dataclasses.asdict(src)},
        )
        emit(result.verdict_line())
        return result

    # ---- 3. the real strict load ------------------------------------------ #
    poisoned = skipped = 0
    if poison_init:
        poisoned, skipped = poison_state(model)
        emit(f"  poisoned with NaN   : {poisoned} float tensors ({skipped} non-float skipped)")

    t_load = time.time()
    allowlisted = set(keys.allowlisted_missing)
    # When a registered allowlist is in force, `load_model(strict=True)` will raise
    # *because of those very keys*. It raises only AFTER
    # `model.load_state_dict(state_dict, strict=False)` has already copied every
    # present tensor (safetensors/torch.py, load_model), so the load is real and
    # complete for everything the file does contain. The raise is therefore
    # tolerated only if it is attributable to exactly the registered keys and
    # nothing else -- verified below against the header triage, which the
    # no-allowlist path proves agrees with the real load.
    strict_raised_as_registered = False
    try:
        if load_mode == "real":
            import safetensors.torch

            # strict=True is passed literally. This is the whole point: the
            # reference path computes `strict` from an exclusion list and throws
            # the returned tuple away.
            missing_ret, unexpected_ret = safetensors.torch.load_model(model, src.file, strict=True)
            missing_ret, unexpected_ret = sorted(missing_ret), sorted(unexpected_ret)
        elif load_mode == "stream":
            missing_ret, unexpected_ret = _stream_load_strict(model, src.file, ckpt_shapes)
        else:
            raise ValueError(f"unknown load mode {load_mode!r}")
    except Exception as exc:
        attributable = (
            isinstance(exc, RuntimeError)
            and bool(allowlisted)
            and sorted(keys.raw_missing) == sorted(allowlisted)
            and not keys.unexpected
            and not keys.shape_mismatch
        )
        if not attributable:
            emit(f"\n  REFUSED: the strict=True load raised {type(exc).__name__}: {exc}")
            result = GateResult(
                status="REFUSE",
                reason="LOAD_ERROR",
                fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
                keys=keys,
                detail={"exception": f"{type(exc).__name__}: {exc}"},
            )
            emit(result.verdict_line())
            return result
        strict_raised_as_registered = True
        missing_ret, unexpected_ret = [], []
        emit(f"  strict load         : raised as expected for the {len(allowlisted)} registered key(s)")
        emit(f"                        {type(exc).__name__}: {exc}")
        emit("                        (tensors present in the file were copied before the raise)")

    if not strict_raised_as_registered:
        emit(f"  strict load         : OK in {round(time.time() - t_load, 2)}s")
    # `missing_ret` from stream mode is not filtered by the allowlist; do it here so
    # both modes are compared on the same basis as the header triage.
    missing_ret = [k for k in missing_ret if k not in allowlisted]
    emit(f"  returned missing    : {len(missing_ret)} {missing_ret[:max_key_examples]}")
    emit(f"  returned unexpected : {len(unexpected_ret)} {unexpected_ret[:max_key_examples]}")

    # The returned tuple is the value the reference path discards. Assert it is
    # empty. Given the header triage above already refused on any non-empty
    # classification, this is a defensive invariant rather than the primary check:
    # it fires only if safetensors' own bookkeeping ever disagrees with the header.
    if missing_ret or unexpected_ret:
        emit("\n  REFUSED: load_model returned a non-empty (missing, unexpected) tuple,")
        emit("  which disagrees with the header triage. The gate's own bookkeeping is")
        emit("  not trustworthy here, so it must not authorise a launch.")
        result = GateResult(
            status="REFUSE",
            reason="NONEMPTY_LOAD_RETURN",
            fields={
                **fields,
                "missing": len(missing_ret),
                "unexpected": len(unexpected_ret),
                "elapsed_s": round(time.time() - t0, 2),
            },
            keys=keys,
            detail={"returned_missing": missing_ret, "returned_unexpected": unexpected_ret},
        )
        emit(result.verdict_line())
        return result

    # ---- 4. did the bytes actually land? ---------------------------------- #
    stale: list[str] = []
    if poison_init:
        all_stale = unpopulated_after_load(model)
        stale = [k for k in all_stale if k not in allowlisted]
        emit(
            f"  still-NaN tensors   : {len(all_stale)} "
            f"({len(all_stale) - len(stale)} registered-allowlisted, {len(stale)} unaccounted) "
            f"of {poisoned} poisoned"
        )
        fields["populated"] = f"{poisoned - len(all_stale)}/{poisoned}"
        if stale:
            dump("UNPOPULATED (still at NaN poison after 'successful' load)", stale)
            result = GateResult(
                status="REFUSE",
                reason="UNPOPULATED_TENSORS",
                fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
                keys=keys,
                detail={"unpopulated": stale, "unpopulated_allowlisted": sorted(set(all_stale) & allowlisted)},
            )
            emit(result.verdict_line())
            return result
    else:
        fields["populated"] = "-"

    ok, bad = sample_verify(model, src.file, sorted(set(ckpt_shapes) & set(model_sd_names)), verify_sample)
    if verify_sample == 0:
        emit("  sampled re-read     : NOT MEASURED (--verify-sample 0). The NaN-poison check")
        emit("                        above still proves every tensor was written; the value")
        emit("                        provenance check was skipped. Consumers that require it")
        emit("                        should reject a verdict with sampled=0/0.")
    else:
        emit(
            f"  sampled re-read     : {ok} verified, {len(bad)} mismatched "
            f"({'ALL keys' if verify_sample < 0 else f'n={verify_sample}'}) from {src.file}"
        )
    fields["sampled"] = f"{ok}/{ok + len(bad)}"
    if bad:
        dump("SAMPLE MISMATCH (in-memory value != this file's tensor)", bad)
        result = GateResult(
            status="REFUSE",
            reason="SAMPLE_DATA_MISMATCH",
            fields={**fields, "elapsed_s": round(time.time() - t0, 2)},
            keys=keys,
            detail={"sample_mismatch": bad},
        )
        emit(result.verdict_line())
        return result

    fields["elapsed_s"] = round(time.time() - t0, 2)
    result = GateResult(
        status="PASS",
        reason="OK",
        fields=fields,
        keys=keys,
        detail={"weight_source": dataclasses.asdict(src), "ckpt_dtypes": _dtype_hist(ckpt_dtypes)},
    )
    emit("-" * 78)
    emit("  ALL CHECKS PASSED: file present, md5/size recorded, strict=True load with")
    emit("  0 missing / 0 unexpected / 0 shape mismatch, every float tensor overwritten,")
    emit("  and sampled values byte-match the file that was opened.")
    emit(result.verdict_line())
    return result


def _dtype_hist(dtypes: dict[str, str]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in dtypes.values():
        hist[value] = hist.get(value, 0) + 1
    return hist


def _stream_load_strict(model: Any, path: str, ckpt_shapes: dict[str, tuple[int, ...]]) -> tuple[list[str], list[str]]:
    """Memory-bounded equivalent of ``load_model(..., strict=True)``.

    Same missing/unexpected computation (including ``_remove_duplicate_names``
    tie resolution), same "shape mismatch raises" behaviour, but tensors are
    copied one at a time so peak RSS is ``model + largest tensor`` rather than
    ``model + entire state dict``.
    """
    from safetensors import safe_open
    from safetensors.torch import _remove_duplicate_names
    import torch

    model_sd = model.state_dict()
    file_keys = set(ckpt_shapes)
    to_removes = _remove_duplicate_names(model_sd, preferred_names=list(file_keys))
    missing = set(model_sd) - file_keys
    unexpected = list(file_keys - set(model_sd))
    for group in to_removes.values():
        for name in group:
            if name not in missing:
                unexpected.append(name)
            else:
                missing.remove(name)

    errors: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as handle, torch.no_grad():
        for key in sorted(file_keys & set(model_sd)):
            tensor = handle.get_tensor(key)
            target = model_sd[key]
            if tuple(tensor.shape) != tuple(target.shape):
                errors.append(
                    f"size mismatch for {key}: copying a param with shape {tuple(tensor.shape)} "
                    f"from checkpoint, the shape in current model is {tuple(target.shape)}."
                )
                continue
            target.detach().copy_(tensor)
    if errors:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t" + "\n\t".join(errors)
        )
    return sorted(missing), sorted(unexpected)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_weight_load.py",
        description="Refuse to launch unless the configured weights load completely under strict=True.",
    )
    parser.add_argument("--config", required=True, help="Registered TrainConfig name (must exist).")
    parser.add_argument(
        "--weight-dir",
        default=None,
        help="Override the directory holding model.safetensors (for controls/what-if).",
    )
    parser.add_argument("--load-mode", choices=("real", "stream", "header"), default="real")
    parser.add_argument("--hash-mode", choices=("sha256", "full", "partial", "none"), default="sha256")
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=8,
        help="Tensors to re-read and byte-compare against the file (-1 = all, 0 = none).",
    )
    parser.add_argument("--no-poison-init", action="store_true", help="Skip the NaN-poison population check.")
    parser.add_argument("--expect-size", type=int, default=None)
    parser.add_argument("--expect-hash", default=None)
    parser.add_argument("--json", default=None, help="Write the full report as JSON here.")
    parser.add_argument(
        "--require-openpi-under",
        default=None,
        metavar="DIR",
        help="Abort unless openpi.__file__ resolves under DIR (guards the editable-install trap).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Always print which openpi tree is in play. The editable install is pinned to
    # a DIFFERENT tree, and without PYTHONPATH pointing at this worktree's src the
    # gate would inspect the wrong source silently -- so record it unconditionally
    # and warn even when the assertion flag was not passed.
    import openpi

    resolved = os.path.realpath(openpi.__file__)
    expected_under = os.path.realpath(str(_REPO_ROOT / "src"))
    print(f"OPENPI_RESOLVED={resolved}")
    print(f"OPENPI_EXPECTED_UNDER={expected_under}")
    if not resolved.startswith(expected_under + os.sep):
        print("WARNING: openpi did NOT resolve under this worktree's src/ -- the gate is")
        print("         inspecting a different source tree than the one you edited.")

    if args.require_openpi_under:
        wanted = os.path.realpath(args.require_openpi_under)
        if not resolved.startswith(wanted + os.sep):
            result = GateResult(
                status="ERROR",
                reason="WRONG_OPENPI_TREE",
                fields={"config": args.config, "openpi": resolved, "expected_under": wanted},
            )
            print(result.verdict_line())
            return result.exit_code

    try:
        config = load_registered_config(args.config)
    except KeyError:
        print(f"ERROR: config {args.config!r} is not registered.")
        print("       train_config.get_config() would SILENTLY fall back to 'pi05_b1k-base';")
        print("       this gate refuses to evaluate a config the launcher did not ask for.")
        result = GateResult(status="REFUSE", reason="CONFIG_NOT_REGISTERED", fields={"config": args.config})
        print(result.verdict_line())
        return result.exit_code

    result = run_gate(
        model_factory=lambda: build_model_for_config(config),
        configured_weight_path=config.pytorch_weight_path,
        override_dir=args.weight_dir,
        config_name=config.name,
        model_name=config.pytorch_model_name,
        load_mode=args.load_mode,
        hash_mode=args.hash_mode,
        verify_sample=args.verify_sample,
        poison_init=not args.no_poison_init,
        allow_missing=allowlist_for(config.name),
        expect_size=args.expect_size,
        expect_hash=args.expect_hash,
    )

    if args.json:
        payload = {
            "status": result.status,
            "reason": result.reason,
            "exit_code": result.exit_code,
            "verdict_line": result.verdict_line(),
            "fields": result.fields,
            "keys": {
                "matched": len(result.keys.matched),
                "tied": result.keys.tied,
                "missing": result.keys.missing,
                "unexpected": result.keys.unexpected,
                "shape_mismatch": [[k, list(a), list(b)] for k, a, b in result.keys.shape_mismatch],
                "allowlisted_missing": result.keys.allowlisted_missing,
            },
            "detail": result.detail,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"JSON report written to {args.json}")

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
