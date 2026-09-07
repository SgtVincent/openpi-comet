"""Batch-level smoke test: real BehaviorLeRobotDataset -> real transforms -> tokens.

Why this exists
---------------
Breaks (1) and (2) of the memory conditioning chain -- the `include_memory_text`
flags in data_loader.py and the `require_memory` flag in data_config.py -- were
covered only by assertions that a string appeared in the source.  That is how a
crash shipped: `ModelTransformFactory` read `self.subtask_source`, a field it did
not have, and no test noticed because no test ever *called* the code.  The
callable tests added afterwards fixed the unit-level gap, but they stub the item.

This script is the only check that runs the real dataset, so it is the only one
that can show the dataset actually emits the fields the transforms require.  It
deliberately does the expensive thing (construct BehaviorLeRobotDataset, decode
video frames) and folds every dataset-dependent measurement into one run.

Each assertion is paired with a control:
  - a positive control, so a check that can only pass is distinguishable from a
    check that verified something;
  - a negative control, so a check that would pass on the wrong data is caught.

Run with the numpy overlay first on PYTHONPATH, otherwise `import omnigibson`
(pulled in transitively by dataset.py) fails on numba/numpy 2.4:
    PYTHONPATH=<overlay>:<worktree>/src python scripts/hier/batch_smoke_memory.py
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import json
import pathlib
import re
import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def preflight() -> None:
    """Assert numpy really RESOLVES to the overlay build.

    Checking that the overlay is on PYTHONPATH would only prove it was set.
    `__version__` plus `__file__` prove it took effect and where from -- two
    different questions that must not share one answer.
    """
    section("PREFLIGHT: numpy provenance")
    import numpy

    ver_ok = numpy.__version__.startswith("2.3")
    check(
        "numpy resolves to a numba-compatible build (2.3.x)",
        ver_ok,
        f"__version__={numpy.__version__}",
    )
    check(
        "numpy is served from the overlay, not the env",
        "pyoverlay" in numpy.__file__,
        f"__file__={numpy.__file__}",
    )
    try:
        import numba

        check("numba imports", True, f"version={numba.__version__}")
    except Exception as exc:  # noqa: BLE001
        check("numba imports", False, f"{type(exc).__name__}: {exc}")
    if not ver_ok:
        print(
            "\nABORT: numpy is not the overlay build, so `import omnigibson` will fail "
            "later in a way that looks unrelated. Set PYTHONPATH=<overlay>:<worktree>/src.",
            flush=True,
        )
        raise SystemExit(2)


def build_dataset(
    root: str, episodes: list[int], subtask_source: str, action_horizon: int, modalities: list[str] | None = None
):
    """Build the real dataset.

    ``modalities`` defaults to RGB only, and that is not a shortcut: the
    ``seg_instance_id`` loader calls ``.to(device="cuda")`` in its constructor
    (obs_utils.py:358), so requesting it makes the smoke need a GPU for reasons
    that have nothing to do with the memory wiring under test.  RGB decode is
    pure CPU (obs_utils.py:323-327).
    """
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    kwargs = dict(
        repo_id="behavior-1k/2025-challenge-demos",
        root=root,
        tolerance_s=1e-4,
        local_only=True,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps={"action": [t / 30.0 for t in range(action_horizon)]},
        episodes=episodes,
        chunk_streaming_using_keyframe=True,
        shuffle=False,
        subtask_source=subtask_source,
        modalities=list(modalities) if modalities else ["rgb"],
    )
    if subtask_source not in ("orchestrator", "annotations_memory"):
        # These sources genuinely need phrase assets; memory text is pre-generated.
        import os

        base = os.path.join(root, "derived")
        kwargs["subtask_template_path"] = os.path.join(base, "subtask_templates.json")
        kwargs["subtask_object_name_mapping_path"] = os.path.join(base, "object_name_mapping.json")
    return BehaviorLeRobotDataset(**kwargs)


def pull_items(dataset, n: int) -> list[dict]:
    items = []
    it = iter(dataset)
    for _ in range(n):
        try:
            items.append(next(it))
        except StopIteration:
            break
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos")
    ap.add_argument("--episodes", default="0,1,2")
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--modalities", default="rgb", help="comma list; rgb only keeps the smoke CPU-only")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    modalities = [x.strip() for x in args.modalities.split(",") if x.strip()]

    preflight()

    section("CONSTRUCT: real BehaviorLeRobotDataset with the memory source")
    dataset = build_dataset(args.root, episodes, "annotations_memory", args.action_horizon, modalities)
    check("dataset constructed with subtask_source='annotations_memory'", True, f"episodes={episodes}")
    check(
        "memory source is actually enabled on the instance",
        bool(getattr(dataset, "memory_source_enabled", False)),
        f"memory_source_enabled={getattr(dataset, 'memory_source_enabled', None)}",
    )
    n_chunks = len(getattr(dataset, "chunks", []) or [])
    check("dataset retained chunks after Memory-coverage restriction", n_chunks > 0, f"chunks={n_chunks}")

    section("ITEMS: the dataset emits the fields the transforms require")
    items = pull_items(dataset, args.items)
    check("pulled at least one item", len(items) > 0, f"items={len(items)}")
    if not items:
        return summarize(args.json_out)  # incomplete: no items

    have_mem = [i for i in items if i.get("memory_text") is not None]
    have_prev = [i for i in items if i.get("previous_memory_text") is not None]
    check(
        "every item carries memory_text",
        len(have_mem) == len(items),
        f"{len(have_mem)}/{len(items)}",
    )
    check(
        "every item carries previous_memory_text",
        len(have_prev) == len(items),
        f"{len(have_prev)}/{len(items)}",
    )
    check(
        "memory subsumes the subtask channel (no second writer into subtask_text)",
        all(i.get("subtask_text") is None for i in items),
        "subtask_text absent on all memory items",
    )
    sample = items[0]
    print(f"\n  --- memory_text[0] ---\n{sample['memory_text']}\n", flush=True)
    print(f"  --- previous_memory_text[0] ---\n{sample['previous_memory_text']}\n", flush=True)

    section("TRANSFORMS: the real production chain, not a hand-built dict")
    import openpi.models.pi05_subtask_config as _p5
    import openpi.training.config as _config
    import openpi.training.moma_memory_config as _mm
    from openpi.training.data_loader import prompt_transform_for
    from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE
    import openpi.transforms as _transforms

    # Use the REGISTERED config, not one assembled here.  A config built inside
    # the test would only prove "my config agrees with my config"; the point is
    # that the thing a launch would actually load turns the memory path on.
    train_cfg = _mm.memory_configs()[0]
    model_config = train_cfg.model
    check(
        "registered config pins the decided budgets",
        model_config.subtask_max_len == 192 and model_config.max_token_len == 320,
        f"name={train_cfg.name} subtask_max_len={model_config.subtask_max_len} max_token_len={model_config.max_token_len}",
    )
    check(
        "registered config selects the Pi05Subtask PyTorch factory",
        train_cfg.pytorch_model_name == "subtask",
        f"pytorch_model_name={train_cfg.pytorch_model_name!r}; 'pi05_subtask' is ModelType, not the factory key",
    )
    factory = train_cfg.data[0] if isinstance(train_cfg.data, (list, tuple)) else train_cfg.data
    # Resolve the real assets. An earlier version of this script skipped this,
    # on the theory that `maybe_download` was stalling -- measured wrong:
    # `maybe_download` returns in 0.0s on this path (cache-hit branch,
    # download.py:82-85). The >120s was the IMPORT of openpi.shared.download,
    # which is the known slow NAS import chain. Attributing a stall to the
    # function being called rather than to getting it loaded cost a premature kill.
    assets_root = pathlib.Path(_mm._PI05_BASE_CKPT) / "assets"
    data_config = factory.create(assets_root, model_config)
    check(
        "registered config selects the memory source",
        data_config.subtask_source == MEMORY_SUBTASK_SOURCE,
        f"subtask_source={data_config.subtask_source!r}",
    )

    prompt_tf = prompt_transform_for(data_config, model_config)
    check(
        "production helper turns the memory flags ON for this config",
        prompt_tf.include_memory_text and prompt_tf.require_memory_text,
        f"include={prompt_tf.include_memory_text} require={prompt_tf.require_memory_text}",
    )
    other = dataclasses.replace(_config.DataConfig(), subtask_source="annotations_skill")
    other_tf = prompt_transform_for(other, model_config)
    check(
        "NEGATIVE CONTROL: non-memory source leaves the flags OFF",
        (not other_tf.include_memory_text) and (not other_tf.require_memory_text),
        f"include={other_tf.include_memory_text} require={other_tf.require_memory_text}",
    )

    tok_tf = None
    for t in data_config.model_transforms.inputs:
        if isinstance(t, _transforms.TokenizeSubtaskInputs):
            tok_tf = t
    check("production config yields a TokenizeSubtaskInputs", tok_tf is not None)
    if tok_tf is None:
        return summarize(args.json_out)
    check(
        "production config set require_memory=True",
        tok_tf.require_memory,
        f"require_memory={tok_tf.require_memory}",
    )
    tokenizer = tok_tf.tokenizer
    check(
        "tokenizer budget matches the config (not a re-derived guess)",
        tokenizer._subtask_max_len == model_config.subtask_max_len,
        f"tokenizer._subtask_max_len={tokenizer._subtask_max_len}",
    )

    # Production order (data_loader.py:174 then :208-215):
    #   prompt transform  ->  repack  ->  data transforms  ->  Normalize  ->  model transforms
    # The prompt transform runs FIRST because it renames `task` to `prompt`, which
    # the repack allowlist then expects. Getting this order wrong is not a
    # cosmetic difference: repack raises KeyError('prompt') without it.
    prompted = prompt_tf(dict(sample))
    check(
        "memory_text survives the prompt transform",
        prompted.get("memory_text") is not None,
        "present after PromptFromLeRobotItem",
    )
    check(
        "previous_memory_text survives the prompt transform",
        prompted.get("previous_memory_text") is not None,
    )

    # The DATA transforms are what this smoke exists to exercise: the repack
    # allowlist and B1kInputs both REBUILD the item, and both silently dropped the
    # memory fields. The first version of this script skipped them, which is
    # precisely how that pair of drops stayed invisible.
    repack_fn = _transforms.compose(data_config.repack_transforms.inputs)
    data_fn = _transforms.compose(data_config.data_transforms.inputs)
    repacked = repack_fn(dict(prompted))
    check(
        "repack keeps memory_text (break 4)",
        repacked.get("memory_text") is not None,
        f"memory keys={sorted(k for k in repacked if 'memory' in k)}",
    )
    check("repack keeps previous_memory_text (break 4)", repacked.get("previous_memory_text") is not None)
    after_data = data_fn(dict(repacked))
    check(
        "B1kInputs forwards memory_text (break 5)",
        after_data.get("memory_text") is not None,
        f"memory keys={sorted(k for k in after_data if 'memory' in k)}",
    )
    check("B1kInputs forwards previous_memory_text (break 5)", after_data.get("previous_memory_text") is not None)
    state = after_data.get("state")
    check("state survives the data transforms", state is not None)
    check(
        "state reaching the tokenizer is 23 wide, not the padded 32",
        state is not None and np.asarray(state).shape[-1] == 23,
        f"dim={None if state is None else np.asarray(state).shape[-1]}; "
        "PadStatesAndActions runs after tokenize so its 9 slots never reach State:",
    )

    # Normalize is part of the production chain and changes the State: bucket
    # VALUES. Running without it would still count buckets correctly, but the
    # numbers would not be the ones training sees, so it is applied when the
    # stats are available and reported as skipped -- never as passed -- when not.
    norm_stats = getattr(data_config, "norm_stats", None)
    if norm_stats:
        after_data = _transforms.Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        )(dict(after_data))
        check(
            "Normalize applied with real stats",
            True,
            f"keys={sorted(norm_stats.keys())}",
        )
    else:
        check(
            "Normalize applied with real stats",
            False,
            "NOT MEASURED: no norm_stats resolved for this assets dir; State: bucket "
            "VALUES below are pre-normalization and are not what training sees",
        )

    # The tokenizer consumes the item AFTER the data transforms: `state` is
    # produced by B1kInputs, so handing it the pre-repack item raises
    # "State is required for subtask tokenization".
    tokenized = tok_tf(dict(after_data))
    for key in ("tokenized_prompt", "subtask_tokens", "subtask_mask", "subtask_ar_mask", "subtask_loss_mask"):
        check(f"tokenized output has {key}", key in tokenized)

    section("LOSS MASK: BOS unsupervised, EOS inside the valid region")

    st_tokens = np.asarray(tokenized["subtask_tokens"])
    st_mask = np.asarray(tokenized["subtask_mask"])
    st_loss = np.asarray(tokenized["subtask_loss_mask"])
    eos_id = tokenizer._tokenizer.eos_id()
    bos_id = tokenizer._tokenizer.bos_id()
    check("first token is BOS", int(st_tokens[0]) == bos_id, f"tokens[0]={int(st_tokens[0])} bos={bos_id}")
    check("BOS is NOT supervised", not bool(st_loss[0]), f"loss_mask[0]={bool(st_loss[0])}")
    eos_pos = np.where(st_tokens == eos_id)[0]
    check("EOS present in the sequence", eos_pos.size > 0, f"positions={eos_pos.tolist()[:4]}")
    if eos_pos.size:
        p = int(eos_pos[0])
        check("EOS is inside the attention mask", bool(st_mask[p]), f"mask[{p}]={bool(st_mask[p])}")
        check("EOS is inside the loss mask", bool(st_loss[p]), f"loss_mask[{p}]={bool(st_loss[p])}")
        check(
            "padding after EOS is neither attended nor supervised",
            (not st_mask[p + 1 :].any()) and (not st_loss[p + 1 :].any()),
            f"any_mask_after={bool(st_mask[p + 1 :].any())} any_loss_after={bool(st_loss[p + 1 :].any())}",
        )
    check(
        "supervised positions are a strict subset of attended positions",
        bool((st_loss & ~st_mask).sum() == 0),
        f"supervised_but_unattended={int((st_loss & ~st_mask).sum())}",
    )
    check(
        "POSITIVE CONTROL: something is actually supervised",
        int(st_loss.sum()) > 0,
        f"supervised_tokens={int(st_loss.sum())}",
    )

    section("PROMPT: State: must be discretised joints, NOT a frame counter")
    prompt_ids = np.asarray(tokenized["tokenized_prompt"]).reshape(-1).tolist()
    prompt_text = tokenizer._tokenizer.decode([int(t) for t in prompt_ids if int(t) > 0])
    print(f"\n  --- decoded prompt ---\n{prompt_text[:400]}\n", flush=True)
    leak = re.search(r"Frame\s+\d+\s+of\s+\d+", prompt_text)
    check(
        "no 'Frame N of M' oracle leak in State:",
        leak is None,
        "no match" if leak is None else f"LEAK: {leak.group(0)!r}",
    )
    check(
        "POSITIVE CONTROL: the leak regex can fire at all",
        re.search(r"Frame\s+\d+\s+of\s+\d+", "State: Frame 3 of 120") is not None,
        "matches a synthetic leaky string",
    )
    m_state = re.search(r"State:\s*([^;]*)", prompt_text)
    state_body = m_state.group(1) if m_state else ""
    nums = re.findall(r"\d+", state_body)
    check(
        "State: carries multiple numeric buckets (discretised joints)",
        len(nums) >= 8,
        f"numeric_tokens={len(nums)} sample={nums[:8]}",
    )
    check(
        "all State: buckets are within the 256-bucket range",
        all(0 <= int(x) <= 255 for x in nums) if nums else False,
        f"min={min(map(int, nums)) if nums else None} max={max(map(int, nums)) if nums else None}",
    )

    section("TARGET: 'Action Query:' must not be re-encoded as text")
    mem_decoded = tokenizer.decode_memory(st_tokens)
    check(
        "Action Query is absent from the tokenized memory target",
        "Action Query" not in mem_decoded,
        f"decoded_head={mem_decoded[:70]!r}",
    )
    check(
        "Action Query is absent from the dataset's memory_text too",
        "Action Query" not in sample["memory_text"],
        "checked the raw field, not just the tokens",
    )

    section("LENGTH: real token lengths against the configured budget")
    lens = [tokenizer.memory_token_length(i["memory_text"]) for i in have_mem]
    check(
        "no item exceeds subtask_max_len",
        all(x <= model_config.subtask_max_len for x in lens),
        f"max={max(lens)} budget={model_config.subtask_max_len} n={len(lens)}",
    )

    section("NEGATIVE CONTROL: the non-memory production chain remains distinct")
    # Do NOT construct the dataset a second time merely to prove a transform
    # property. The first version did exactly that, cold-read another ~2 GiB, and
    # sat in uninterruptible I/O for 15+ minutes after every substantive check had
    # already passed. Reuse the REAL item above, remove the fields this control arm
    # is defined not to have, and run it through a real non-memory config's
    # production transforms. This preserves behavioural coverage of the
    # repack/B1kInputs/tokenizer boundary; what it deliberately does not claim is a
    # second measurement of the annotations_skill dataset loader.
    control_factory = dataclasses.replace(
        factory,
        base_config=dataclasses.replace(factory.base_config, subtask_source="annotations_skill"),
    )
    control_config = control_factory.create(assets_root, model_config)
    control_item = dict(sample)
    control_item.pop("memory_text", None)
    control_item.pop("previous_memory_text", None)
    control_item["subtask_text"] = "NEGATIVE-CONTROL-SUBTASK"
    control_prompted = prompt_transform_for(control_config, model_config)(control_item)
    control_repacked = _transforms.compose(control_config.repack_transforms.inputs)(control_prompted)
    control_after_data = _transforms.compose(control_config.data_transforms.inputs)(control_repacked)
    check(
        "non-memory chain carries no memory fields",
        all(k not in control_after_data for k in ("memory_text", "previous_memory_text")),
        f"memory keys={sorted(k for k in control_after_data if 'memory' in k)}",
    )
    check(
        "POSITIVE CONTROL: non-memory chain still carries subtask_text",
        control_after_data.get("subtask_text") == "NEGATIVE-CONTROL-SUBTASK",
        "the control arm is live, not simply empty",
    )
    control_tok = next(
        t for t in control_config.model_transforms.inputs if isinstance(t, _transforms.TokenizeSubtaskInputs)
    )
    control_tokenized = control_tok(dict(control_after_data))
    check(
        "non-memory tokenizer supervises its subtask",
        bool(np.asarray(control_tokenized["subtask_loss_mask"]).any()),
        f"supervised_tokens={int(np.asarray(control_tokenized['subtask_loss_mask']).sum())}",
    )
    check(
        "memory tokenizer rejects the same non-memory item",
        _raises(tok_tf, control_after_data),
        "require_memory=True raises rather than silently unconditioning",
    )

    return summarize(args.json_out, complete=True)


def _raises(tok_tf, item: dict) -> bool:
    if not item:
        return False
    probe = {k: v for k, v in item.items() if k != "memory_text"}
    probe["prompt"] = probe.pop("task", "")
    probe.pop("subtask_text", None)
    try:
        tok_tf(probe)
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


#: Number of checks a complete run performs.  Without this, a run that dies
#: partway reports "0 failed" and renders as a pass -- which is exactly what the
#: first execution of this script did: it crashed on a missing CUDA driver before
#: any substantive check ran, and still printed `SMOKE_VERDICT: ok passed=6`.
#: "Did not run" must never be indistinguishable from "ran and passed".
EXPECTED_CHECKS = 51


def summarize(json_out: str, *, complete: bool = False) -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    short = len(RESULTS) < EXPECTED_CHECKS
    incomplete = (not complete) or short
    print(f"\n{'=' * 60}\nSMOKE SUMMARY: {passed} passed, {len(failed)} failed, ran {len(RESULTS)}/{EXPECTED_CHECKS}", flush=True)
    for n, d in failed:
        print(f"  FAILED: {n} -- {d}", flush=True)
    if incomplete:
        print(
            f"  INCOMPLETE: only {len(RESULTS)} of {EXPECTED_CHECKS} checks ran"
            f"{' (run aborted)' if not complete else ''}. Unrun checks are NOT passes.",
            flush=True,
        )
    verdict = "FAILED" if failed else ("INCOMPLETE" if incomplete else "ok")
    print(
        f"SMOKE_VERDICT: {verdict} passed={passed} failed={len(failed)} ran={len(RESULTS)}/{EXPECTED_CHECKS}",
        flush=True,
    )
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(
                {"verdict": verdict, "passed": passed, "failed": len(failed),
                 "ran": len(RESULTS), "expected": EXPECTED_CHECKS, "results": [{"name": n, "ok": o, "detail": d} for n, o, d in RESULTS]},
                f,
                indent=2,
            )
    return 0 if (not failed and not incomplete) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        summarize("")
        sys.exit(3)
