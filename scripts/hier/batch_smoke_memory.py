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
import json
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
    from openpi.training.data_loader import prompt_transform_for
    from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE
    import openpi.transforms as _transforms

    model_config = _p5.Pi05SubtaskConfig()
    data_config = dataclasses.replace(
        _config.DataConfig(),
        subtask_source=MEMORY_SUBTASK_SOURCE,
    )
    prompt_tf = prompt_transform_for(data_config, model_config)
    check(
        "production helper turns the memory flags ON for the memory source",
        prompt_tf.include_memory_text and prompt_tf.require_memory_text,
        f"include={prompt_tf.include_memory_text} require={prompt_tf.require_memory_text}",
    )
    # Negative control: the same helper on a non-memory config must NOT set them,
    # otherwise the assertion above would pass on a hardcoded True.
    other = dataclasses.replace(_config.DataConfig(), subtask_source="annotations_skill")
    other_tf = prompt_transform_for(other, model_config)
    check(
        "NEGATIVE CONTROL: non-memory source leaves the flags OFF",
        (not other_tf.include_memory_text) and (not other_tf.require_memory_text),
        f"include={other_tf.include_memory_text} require={other_tf.require_memory_text}",
    )

    # Build the tokenize transform through the REAL production factory rather than
    # re-deriving its arguments here.  Re-deriving them is how this script first
    # failed: it guessed `prompt_max_len` as a config field, while production maps
    # it from `max_token_len` (data_config.py:274).  A smoke that reconstructs the
    # chain can pass while the chain it imitates is broken -- and the factory is
    # exactly the object that shipped an AttributeError.
    factory = _config.ModelTransformFactory(subtask_source=MEMORY_SUBTASK_SOURCE)
    group = factory(model_config)
    tok_tf = None
    for t in group.inputs:
        if isinstance(t, _transforms.TokenizeSubtaskInputs):
            tok_tf = t
    check(
        "production factory yields a TokenizeSubtaskInputs",
        tok_tf is not None,
        f"transforms={[type(t).__name__ for t in group.inputs]}",
    )
    if tok_tf is None:
        return summarize(args.json_out)
    check(
        "production factory set require_memory=True for the memory source",
        tok_tf.require_memory,
        f"require_memory={tok_tf.require_memory}",
    )
    ctrl_tf = None
    for t in _config.ModelTransformFactory(subtask_source="annotations_skill")(model_config).inputs:
        if isinstance(t, _transforms.TokenizeSubtaskInputs):
            ctrl_tf = t
    check(
        "NEGATIVE CONTROL: factory leaves require_memory=False off the memory source",
        ctrl_tf is not None and not ctrl_tf.require_memory,
        f"require_memory={None if ctrl_tf is None else ctrl_tf.require_memory}",
    )
    tokenizer = tok_tf.tokenizer

    prompted = prompt_tf({**sample, "task": sample.get("task", "")})
    check(
        "memory_text survives the prompt transform",
        prompted.get("memory_text") is not None,
        "present after PromptFromLeRobotItem",
    )
    check(
        "previous_memory_text survives the prompt transform",
        prompted.get("previous_memory_text") is not None,
        "present after PromptFromLeRobotItem",
    )
    tokenized = tok_tf(dict(prompted))
    for key in ("tokenized_prompt", "subtask_tokens", "subtask_mask", "subtask_ar_mask", "subtask_loss_mask"):
        check(f"tokenized output has {key}", key in tokenized)

    section("LOSS MASK: BOS unsupervised, EOS inside the valid region")
    import numpy as np

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

    section("NEGATIVE CONTROL: a non-memory dataset must NOT emit memory_text")
    try:
        skill_ds = build_dataset(args.root, episodes[:1], "annotations_skill", args.action_horizon, modalities)
        skill_items = pull_items(skill_ds, 2)
        check(
            "annotations_skill items carry no memory_text",
            all(i.get("memory_text") is None for i in skill_items),
            f"items={len(skill_items)}",
        )
        check(
            "annotations_skill still produces its own subtask_text (source is alive)",
            any(i.get("subtask_text") is not None for i in skill_items),
            "positive control that the control arm is not simply empty",
        )
        check(
            "require_memory=True REJECTS a non-memory item instead of silently unconditioning",
            _raises(tok_tf, skill_items[0] if skill_items else {}),
            "TokenizeSubtaskInputs(require_memory=True) raised as designed",
        )
    except Exception as exc:  # noqa: BLE001
        check("negative-control dataset built", False, f"{type(exc).__name__}: {str(exc)[:160]}")

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
EXPECTED_CHECKS = 33


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
