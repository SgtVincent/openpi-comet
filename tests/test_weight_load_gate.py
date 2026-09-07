#!/usr/bin/env python3
"""Tests for the pre-launch weight-loading gate (``scripts/hier/preflight_weight_load.py``).

Every assertion the gate makes is shown here to be *capable of failing*:

* ``test_positive_control_*``     -- intact input PASSES, so refusal is specific.
* ``test_negative_control_*``     -- deliberately damaged input is REFUSED, with
  the reason token and the offending key names.
* ``test_defect_reproduction_*``  -- reproduces the original defect (``strict=False``
  + discarded return tuple + unconditional success log) on the *same* damaged
  input the gate refuses, so the gate's value is demonstrated rather than asserted.
* ``test_equivalence_*``          -- ``--load-mode stream`` and ``--load-mode real``
  return identical verdicts, so the memory-bounded path is not a second opinion.
* ``test_no_relax_escape_hatch`` / ``test_registered_allowlist_*`` -- there is no
  CLI way to ask for ``strict=False`` and no CLI way to allowlist a key.

Run with (8-core cgroup, single-threaded BLAS, worktree-pinned openpi)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet-hier-moma/src \
    /mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python \
    -m pytest tests/test_weight_load_gate.py -v -p no:cacheprovider

Nothing here touches a GPU, a real 7-14 GB checkpoint, or the network: the
controls run on a ~1 KB synthetic safetensors file so they are fast and can be
damaged freely. The real-checkpoint runs are separate, recorded commands.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "scripts" / "hier" / "preflight_weight_load.py"


def _load_gate_module():
    """Import the gate by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("_preflight_weight_load", _GATE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
class TinyTiedModel(nn.Module):
    """Small stand-in with the structural feature that matters: a tied weight.

    ``paligemma.lm_head.weight`` is tied to ``embed_tokens.weight`` in the real
    checkpoints, which is why ``save_model`` writes only one of them and records
    the alias in ``__metadata__``. A gate that did not honour that would call a
    perfectly good checkpoint broken, so the synthetic model reproduces it.
    """

    def __init__(self, vocab: int = 8, dim: int = 4, fill: float = 1.0):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied, same storage
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("scale", torch.full((dim,), fill))
        self.register_buffer("step", torch.zeros(2, dtype=torch.long))
        with torch.no_grad():
            self.embed.weight.fill_(fill)
            self.proj.weight.fill_(fill * 2)
            self.proj.bias.fill_(fill * 3)


def _model_factory(**kwargs):
    return lambda: TinyTiedModel(**kwargs)


@pytest.fixture
def intact_ckpt(tmp_path: Path) -> Path:
    """A directory holding a ``model.safetensors`` written by ``save_model``."""
    import safetensors.torch

    directory = tmp_path / "intact"
    directory.mkdir()
    reference = TinyTiedModel(fill=0.5)
    safetensors.torch.save_model(reference, str(directory / "model.safetensors"))
    return directory


def _rewrite(src_dir: Path, dst_dir: Path, mutate) -> Path:
    """Copy ``src_dir/model.safetensors`` into ``dst_dir`` applying ``mutate``.

    ``__metadata__`` (the tie map) is carried across so damage is limited to what
    ``mutate`` does and nothing else.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(str(src_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict; .keys() is its API
            tensors[key] = handle.get_tensor(key)
    mutate(tensors, metadata)
    save_file(tensors, str(dst_dir / "model.safetensors"), metadata=metadata)
    return dst_dir


def _run(directory, *, load_mode="real", **kwargs):
    buffer = io.StringIO()
    result = gate.run_gate(
        model_factory=_model_factory(),
        configured_weight_path=str(directory) if directory is not None else None,
        config_name="synthetic",
        model_name=kwargs.pop("model_name", "pi05_ki_joint_fast"),
        load_mode=load_mode,
        hash_mode=kwargs.pop("hash_mode", "full"),
        out=buffer,
        **kwargs,
    )
    return result, buffer.getvalue()


def _verdict_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(gate.VERDICT_TOKEN)]


def _expect(result, status: str, reason: str, text: str = "") -> None:
    """Assert status and reason separately so a failure names which one differed."""
    assert result.status == status, (result.status, result.reason, text)
    assert result.reason == reason, (result.reason, text)


# --------------------------------------------------------------------------- #
# POSITIVE CONTROLS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("load_mode", ["real", "stream"])
def test_positive_control_intact_checkpoint_passes(intact_ckpt: Path, load_mode: str):
    """An intact file PASSES -- so any refusal below is specific, not unconditional."""
    result, text = _run(intact_ckpt, load_mode=load_mode)
    assert result.status == "PASS", text
    assert result.reason == "OK"
    assert result.exit_code == 0
    assert result.fields["missing"] == 0
    assert result.fields["unexpected"] == 0
    assert result.fields["shape_mismatch"] == 0
    # The tied weight must be recognised as resolvable, not counted as missing.
    assert result.keys.tied == ["head.weight"], result.keys.tied
    lines = _verdict_lines(text)
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"{gate.VERDICT_TOKEN} PASS reason=OK ")
    assert "strict=True" in lines[0]


def test_positive_control_populated_and_sampled(intact_ckpt: Path):
    """PASS also proves every float tensor was overwritten and values came from the file."""
    result, text = _run(intact_ckpt, verify_sample=4)
    assert result.status == "PASS", text
    # 5 float tensors in state_dict: embed.weight, head.weight (tied), proj.weight,
    # proj.bias, and the float buffer 'scale'. The int buffer 'step' cannot hold NaN.
    assert result.fields["populated"] == "5/5", result.fields
    # 4 keys are actually stored in the file (head.weight is deduped as tied).
    assert result.fields["sampled"] == "4/4", result.fields


def test_positive_control_hash_is_a_real_md5(intact_ckpt: Path):
    """The ``md5`` label must be an actual md5 of the whole file (sentinel: wrong digest)."""
    path = intact_ckpt / "model.safetensors"
    algo, digest, covered = gate.hash_file(str(path), "full")
    assert algo == "md5"
    assert covered == path.stat().st_size
    assert digest == hashlib.md5(path.read_bytes()).hexdigest()
    # sentinel: an md5 of different bytes must NOT match
    assert digest != hashlib.md5(path.read_bytes() + b"x").hexdigest()


def test_partial_hash_label_states_its_coverage(intact_ckpt: Path):
    """A partial hash must never be called an md5 and must declare what it covers."""
    path = intact_ckpt / "model.safetensors"
    algo, digest, covered = gate.hash_file(str(path), "partial")
    assert algo.startswith("md5_partial:")
    assert "head=0.." in algo
    assert "tail=" in algo
    assert "plus_size_string" in algo
    assert algo != "md5"
    assert digest
    assert covered > 0
    # The declared ranges must add up to the bytes actually hashed, and (for a file
    # smaller than the window) must not double-count.
    assert covered == path.stat().st_size
    assert digest != gate.hash_file(str(path), "full")[1]  # different algorithm, different digest


def test_verify_sample_modes(intact_ckpt: Path):
    """-1 verifies every stored key; 0 is reported as NOT MEASURED, never as 0 problems."""
    all_keys, text = _run(intact_ckpt, verify_sample=-1)
    assert all_keys.status == "PASS", text
    # Count the keys that are both physically stored and present in the model,
    # which is exactly the population sample_verify draws from. TinyTiedModel has
    # six logical state_dict names, while safetensors stores only one member of
    # the tied head/embed pair, so pinning either 4 or 6 would encode the wrong
    # entity count.
    from safetensors import safe_open

    with safe_open(str(intact_ckpt / "model.safetensors"), framework="pt", device="cpu") as handle:
        expected_keys = len(set(handle.keys()) & set(TinyTiedModel().state_dict()))
    assert all_keys.fields["sampled"] == f"{expected_keys}/{expected_keys}"
    assert "ALL keys" in text
    none, text0 = _run(intact_ckpt, verify_sample=0)
    assert none.status == "PASS", text0
    assert none.fields["sampled"] == "0/0"
    assert "NOT MEASURED (--verify-sample 0)" in text0


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS -- damaged checkpoints
# --------------------------------------------------------------------------- #
def test_negative_control_removed_key_is_refused(intact_ckpt: Path, tmp_path: Path):
    damaged = _rewrite(intact_ckpt, tmp_path / "removed", lambda t, m: t.pop("proj.weight"))
    result, text = _run(damaged)
    assert result.status == "REFUSE", text
    assert result.reason == "MISSING_KEYS"
    assert result.exit_code == 2
    assert result.keys.missing == ["proj.weight"], result.keys.missing
    assert result.keys.unexpected == []
    assert "MISSING KEYS (model expects, checkpoint lacks)" in text
    assert "proj.weight" in text
    line = _verdict_lines(text)[0]
    assert "missing=1" in line
    assert "unexpected=0" in line


def test_negative_control_renamed_key_is_refused(intact_ckpt: Path, tmp_path: Path):
    def rename(tensors, _metadata):
        tensors["proj.weight_RENAMED"] = tensors.pop("proj.weight")

    damaged = _rewrite(intact_ckpt, tmp_path / "renamed", rename)
    result, text = _run(damaged)
    assert result.status == "REFUSE", text
    assert result.reason == "MISSING_AND_UNEXPECTED_KEYS"
    assert result.keys.missing == ["proj.weight"]
    assert result.keys.unexpected == ["proj.weight_RENAMED"]
    assert "UNEXPECTED KEYS (checkpoint has, model lacks)" in text


def test_negative_control_extra_key_is_refused(intact_ckpt: Path, tmp_path: Path):
    def add(tensors, _metadata):
        tensors["bogus.extra"] = torch.zeros(3)

    damaged = _rewrite(intact_ckpt, tmp_path / "extra", add)
    result, text = _run(damaged)
    assert result.status == "REFUSE", text
    assert result.reason == "UNEXPECTED_KEYS"
    assert result.keys.unexpected == ["bogus.extra"]
    assert result.keys.missing == []


def test_negative_control_wrong_shape_is_refused(intact_ckpt: Path, tmp_path: Path):
    def reshape(tensors, _metadata):
        tensors["proj.weight"] = torch.zeros(5, 5)

    damaged = _rewrite(intact_ckpt, tmp_path / "shape", reshape)
    result, text = _run(damaged)
    assert result.status == "REFUSE", text
    assert result.reason == "SHAPE_MISMATCH"
    assert [k for k, _, _ in result.keys.shape_mismatch] == ["proj.weight"]
    assert "SHAPE MISMATCH" in text
    assert "model=(4, 4) ckpt=(5, 5)" in text


def test_negative_control_missing_file_is_refused(tmp_path: Path):
    """The 'Skipping weight loading' branch: no file at all must refuse."""
    empty = tmp_path / "no_such_dir"
    empty.mkdir()
    result, text = _run(empty)
    assert result.status == "REFUSE", text
    assert result.reason == "WEIGHT_FILE_MISSING"
    assert result.exit_code == 2
    assert "does not exist as a regular file" in text
    assert "Skipping weight loading" in text  # names the reference-path behaviour
    assert result.fields["weight_file"].endswith("/no_such_dir/model.safetensors")


def test_negative_control_weight_path_none_is_refused():
    result, text = _run(None)
    assert result.status == "REFUSE", text
    assert result.reason == "WEIGHT_PATH_NOT_CONFIGURED"
    assert "nothing would be loaded" in text


def test_negative_control_truncated_header_is_refused(intact_ckpt: Path, tmp_path: Path):
    directory = tmp_path / "truncated"
    directory.mkdir()
    raw = (intact_ckpt / "model.safetensors").read_bytes()
    (directory / "model.safetensors").write_bytes(raw[:24])
    result, text = _run(directory)
    assert result.status == "REFUSE", text
    assert result.reason == "HEADER_UNREADABLE"


def test_negative_control_size_and_hash_pinning(intact_ckpt: Path):
    result, text = _run(intact_ckpt, expect_size=1)
    _expect(result, "REFUSE", "SIZE_MISMATCH", text)
    result, text = _run(intact_ckpt, expect_hash="0" * 32)
    _expect(result, "REFUSE", "HASH_MISMATCH", text)


# --------------------------------------------------------------------------- #
# the gate does not inherit the exclusion list
# --------------------------------------------------------------------------- #
_EXCLUSION_LIST = (
    "pi05_ki_joint_fast",
    "vlm2",
    "vlm2_subtask",
    "subtask",
    "pi0_hamlet",
    "pi0_memoryvla",
    "pi05_ki_joint_query",
)


@pytest.mark.parametrize("model_name", _EXCLUSION_LIST)
def test_strictness_is_explicit_not_inherited(intact_ckpt: Path, tmp_path: Path, model_name: str):
    """Damaged input is refused for EVERY name that ``train_accelerate.py`` exempts.

    In the reference path each of these names selects ``strict=False``. Behavioural
    proof (not a source grep) that the gate ignores that list.
    """
    damaged = _rewrite(intact_ckpt, tmp_path / f"excl_{model_name}", lambda t, m: t.pop("proj.bias"))
    result, text = _run(damaged, model_name=model_name)
    assert result.status == "REFUSE", text
    assert result.reason == "MISSING_KEYS"
    assert "strict=True" in _verdict_lines(text)[0]


def test_no_relax_escape_hatch():
    """There is no CLI way to ask for a non-strict load (behavioural, via argparse)."""
    parser = gate.build_parser()
    for argv in (
        ["--config", "x", "--strict", "False"],
        ["--config", "x", "--no-strict"],
        ["--config", "x", "--relax"],
        ["--config", "x", "--load-mode", "nonstrict"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    # positive control: the flags that DO exist parse cleanly, so the four
    # SystemExits above are about those flags and not about argparse being broken.
    ok = parser.parse_args(["--config", "x", "--load-mode", "header", "--hash-mode", "partial"])
    assert ok.load_mode == "header"
    assert ok.hash_mode == "partial"


# --------------------------------------------------------------------------- #
# reproduce the defect the gate exists to stop
# --------------------------------------------------------------------------- #
def test_defect_reproduction_strict_false_hides_partial_load(intact_ckpt: Path, tmp_path: Path):
    """The reference path succeeds silently on the input the gate refuses.

    Mirrors ``train_accelerate.py``: ``strict`` comes from the exclusion list
    (False here), the returned tuple is discarded, and the success line prints
    unconditionally. The poisoned tensor proves the parameter was never written.
    """
    import safetensors.torch

    damaged = _rewrite(intact_ckpt, tmp_path / "defect", lambda t, m: t.pop("proj.weight"))

    model = TinyTiedModel()
    gate.poison_state(model)
    load_strict = "pi05_ki_joint_fast" not in _EXCLUSION_LIST  # -> False, as in the reference
    assert load_strict is False
    missing, unexpected = safetensors.torch.load_model(
        model, str(damaged / "model.safetensors"), strict=load_strict
    )  # does NOT raise
    assert sorted(missing) == ["proj.weight"]
    assert unexpected == []
    # ...and the parameter is still exactly what initialisation left there:
    assert bool(torch.isnan(model.proj.weight).all())
    assert gate.unpopulated_after_load(model) == ["proj.weight"]

    # Same input through the gate:
    result, text = _run(damaged)
    _expect(result, "REFUSE", "MISSING_KEYS", text)


def test_unpopulated_detection_catches_silently_unwritten_tensor(intact_ckpt: Path, tmp_path: Path):
    """The NaN-poison check flags a tensor left at init even though a load 'succeeded'."""
    import safetensors.torch

    damaged = _rewrite(intact_ckpt, tmp_path / "unpop", lambda t, m: t.pop("proj.bias"))
    model = TinyTiedModel()
    poisoned, skipped = gate.poison_state(model)
    assert poisoned == 5  # 4 float params + the float buffer 'scale'
    assert skipped == 1  # the int buffer 'step' cannot hold NaN
    safetensors.torch.load_model(model, str(damaged / "model.safetensors"), strict=False)
    assert gate.unpopulated_after_load(model) == ["proj.bias"]
    # sentinel: with the intact file nothing is left unpopulated
    model2 = TinyTiedModel()
    gate.poison_state(model2)
    safetensors.torch.load_model(model2, str(intact_ckpt / "model.safetensors"), strict=True)
    assert gate.unpopulated_after_load(model2) == []


def test_sample_verify_detects_values_that_did_not_come_from_the_file(intact_ckpt: Path):
    """``sample_verify`` must fail when memory disagrees with the file (and pass when it agrees)."""
    import safetensors.torch

    path = str(intact_ckpt / "model.safetensors")
    model = TinyTiedModel()
    safetensors.torch.load_model(model, path, strict=True)
    keys = sorted(set(gate.read_safetensors_header(path)[0]) & set(model.state_dict()))
    ok, bad = gate.sample_verify(model, path, keys, len(keys))
    assert bad == []  # positive control
    assert ok == len(keys)
    with torch.no_grad():
        model.proj.bias.add_(1.0)  # tamper
    ok, bad = gate.sample_verify(model, path, keys, len(keys))
    assert "proj.bias" in bad, (ok, bad)


# --------------------------------------------------------------------------- #
# header mode is NOT MEASURED, not a pass
# --------------------------------------------------------------------------- #
def test_header_mode_never_passes(intact_ckpt: Path):
    result, text = _run(intact_ckpt, load_mode="header")
    assert result.status == "INCONCLUSIVE", text
    assert result.reason == "HEADER_ONLY_NOT_MEASURED"
    assert result.exit_code == 4
    assert "NOT MEASURED" in text
    assert _verdict_lines(text)[0].startswith(f"{gate.VERDICT_TOKEN} INCONCLUSIVE")


def test_header_mode_still_refuses_damage(intact_ckpt: Path, tmp_path: Path):
    damaged = _rewrite(intact_ckpt, tmp_path / "hdr_damaged", lambda t, m: t.pop("proj.weight"))
    result, text = _run(damaged, load_mode="header")
    _expect(result, "REFUSE", "MISSING_KEYS", text)


# --------------------------------------------------------------------------- #
# real vs stream must be the same opinion, not a second one
# --------------------------------------------------------------------------- #
def _all_cases(intact: Path, tmp_path: Path) -> list[tuple[str, Path]]:
    cases = [("intact", intact)]
    cases.append(("removed", _rewrite(intact, tmp_path / "eq_removed", lambda t, m: t.pop("proj.weight"))))
    cases.append(
        (
            "renamed",
            _rewrite(
                intact,
                tmp_path / "eq_renamed",
                lambda t, m: t.__setitem__("proj.weight_X", t.pop("proj.weight")),
            ),
        )
    )
    cases.append(
        ("extra", _rewrite(intact, tmp_path / "eq_extra", lambda t, m: t.__setitem__("z.z", torch.zeros(2))))
    )
    cases.append(
        (
            "shape",
            _rewrite(intact, tmp_path / "eq_shape", lambda t, m: t.__setitem__("proj.weight", torch.zeros(5, 5))),
        )
    )
    return cases


def test_equivalence_real_vs_stream(intact_ckpt: Path, tmp_path: Path):
    """``stream`` mode must return byte-identical verdict fields to ``real`` mode."""
    for label, directory in _all_cases(intact_ckpt, tmp_path):
        real, real_text = _run(directory, load_mode="real")
        stream, stream_text = _run(directory, load_mode="stream")
        assert real.status == stream.status, (label, real_text, stream_text)
        assert real.reason == stream.reason, (label, real.reason, stream.reason)
        assert real.keys.missing == stream.keys.missing, label
        assert real.keys.unexpected == stream.keys.unexpected, label
        assert real.keys.shape_mismatch == stream.keys.shape_mismatch, label
        for field in ("missing", "unexpected", "shape_mismatch", "tied", "populated", "sampled"):
            assert real.fields.get(field) == stream.fields.get(field), (label, field)


# --------------------------------------------------------------------------- #
# the registered allowlist is registered, per-key, and not reachable from a flag
# --------------------------------------------------------------------------- #
def test_registered_allowlist_is_empty_and_not_settable_from_cli():
    """No key is currently allowlisted, and no CLI flag can add one."""
    assert gate.REGISTERED_ALLOWLIST == {}, gate.REGISTERED_ALLOWLIST
    assert gate.allowlist_for("anything") == ()
    parser = gate.build_parser()
    for argv in (
        ["--config", "x", "--allow-missing", "proj.bias"],
        ["--config", "x", "--allow-missing-prefix", "proj."],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_registered_allowlist_mechanism_works_and_stays_narrow(intact_ckpt: Path, tmp_path: Path):
    """A registered key is tolerated; anything else in the same file still refuses.

    Exercises the branch where ``load_model(..., strict=True)`` legitimately raises
    for the registered key: the raise is accepted only because it is attributable
    to exactly that key, and the still-NaN tensor is reported as allowlisted rather
    than silently counted as populated.
    """
    damaged = _rewrite(intact_ckpt, tmp_path / "reg_allow", lambda t, m: t.pop("proj.bias"))

    # No registration -> refuse.
    refused, refused_text = _run(damaged)
    _expect(refused, "REFUSE", "MISSING_KEYS", refused_text)

    # Registered by exact key -> pass, visibly.
    result, text = _run(damaged, allow_missing=("proj.bias",))
    assert result.status == "PASS", text
    assert result.keys.allowlisted_missing == ["proj.bias"]
    assert result.keys.missing == []
    line = _verdict_lines(text)[0]
    assert "allowlisted=1" in line
    assert "missing=0" in line
    assert "raised as expected for the 1 registered key(s)" in text
    assert result.fields["populated"] == "4/5", result.fields  # proj.bias legitimately still NaN

    # Sentinel: registering an unrelated key must not absolve the real one.
    other, other_text = _run(damaged, allow_missing=("something.else",))
    _expect(other, "REFUSE", "MISSING_KEYS", other_text)

    # Sentinel: an allowlist does not excuse a *second*, unregistered problem.
    def drop_two(tensors, _metadata):
        tensors.pop("proj.bias")
        tensors.pop("proj.weight")

    two = _rewrite(intact_ckpt, tmp_path / "reg_allow_two", drop_two)
    both, both_text = _run(two, allow_missing=("proj.bias",))
    _expect(both, "REFUSE", "MISSING_KEYS", both_text)
    assert both.keys.missing == ["proj.weight"]


def test_unpopulated_branch_wiring(intact_ckpt: Path, monkeypatch):
    """The UNPOPULATED_TENSORS refusal path is wired correctly.

    The *detector* is tested for real in
    ``test_unpopulated_detection_catches_silently_unwritten_tensor``. On an intact
    file no tensor can legitimately be left unwritten, so the refusal branch is
    exercised here by stubbing the detector -- this checks the reason token, the
    printed section and the exit code, not the detection itself.
    """
    monkeypatch.setattr(gate, "unpopulated_after_load", lambda _model: ["proj.weight"])
    result, text = _run(intact_ckpt)
    assert result.status == "REFUSE", text
    assert result.reason == "UNPOPULATED_TENSORS"
    assert result.exit_code == 2
    assert "UNPOPULATED (still at NaN poison after 'successful' load)" in text
    assert result.detail["unpopulated"] == ["proj.weight"]
    # positive control: without the stub the same input PASSES (so the refusal is
    # caused by the stub, not by anything else about this file).
    monkeypatch.undo()
    ok, _ = _run(intact_ckpt)
    assert ok.status == "PASS"


def test_sample_mismatch_branch_wiring(intact_ckpt: Path, monkeypatch):
    """The SAMPLE_DATA_MISMATCH refusal path is wired correctly (detector tested elsewhere)."""
    monkeypatch.setattr(gate, "sample_verify", lambda *a, **k: (0, ["proj.weight"]))
    result, text = _run(intact_ckpt)
    _expect(result, "REFUSE", "SAMPLE_DATA_MISMATCH", text)
    assert "SAMPLE MISMATCH (in-memory value != this file's tensor)" in text
    monkeypatch.undo()
    assert _run(intact_ckpt)[0].status == "PASS"


def test_nonempty_load_return_branch_wiring(intact_ckpt: Path, monkeypatch):
    """The defensive 'returned tuple must be empty' refusal is wired correctly.

    Unreachable in normal operation (the header triage refuses first, and
    ``strict=True`` raises rather than returning a non-empty tuple), so the branch
    is exercised by stubbing the stream loader. It exists because that returned
    tuple is exactly the value ``train_accelerate.py`` throws away.
    """
    monkeypatch.setattr(gate, "_stream_load_strict", lambda *a, **k: (["ghost.key"], []))
    result, text = _run(intact_ckpt, load_mode="stream")
    assert result.status == "REFUSE", text
    assert result.reason == "NONEMPTY_LOAD_RETURN"
    assert result.detail["returned_missing"] == ["ghost.key"]
    monkeypatch.undo()
    assert _run(intact_ckpt, load_mode="stream")[0].status == "PASS"


# --------------------------------------------------------------------------- #
# resolution / labelling / contract
# --------------------------------------------------------------------------- #
def test_relative_weight_path_is_resolved_against_cwd_and_flagged(tmp_path: Path, monkeypatch):
    """Most configs use a relative ``pytorch_weight_path``; the gate must say so."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    src = gate.resolve_weight_source("checkpoints/whatever")
    assert src.is_relative is True
    assert src.cwd == os.getcwd()
    assert src.file == str(tmp_path / "checkpoints" / "whatever" / "model.safetensors")
    assert src.exists is False
    absolute = gate.resolve_weight_source("/tmp/abs/dir")
    assert absolute.is_relative is False


def test_known_source_labels_and_size_check(tmp_path: Path):
    """The two candidate sources are labelled by directory name + size."""
    assert set(gate.KNOWN_SOURCES) == {"pi05_base_pytorch", "pi05-b1kpt50-cs32"}
    directory = tmp_path / "pi05_base_pytorch"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"x")
    src = gate.resolve_weight_source(str(directory))
    assert src.source_id == "pi05_base_pytorch"
    assert src.size_matches_known is False  # 1 byte != 7233650408 -- label, not verdict
    unknown = gate.resolve_weight_source(str(tmp_path))
    assert unknown.source_id == "unknown"


def test_orbax_params_sibling_is_reported(tmp_path: Path):
    directory = tmp_path / "pi05-b1kpt50-cs32"
    (directory / "params").mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"x")
    src = gate.resolve_weight_source(str(directory))
    assert src.sibling_orbax_params is True
    assert gate.resolve_weight_source(str(tmp_path / "nope")).sibling_orbax_params is False


def test_verdict_line_is_single_and_machine_readable(intact_ckpt: Path, tmp_path: Path):
    """Exactly one verdict line; whitespace-delimited ``key=value``; parseable."""
    for directory in (intact_ckpt, tmp_path / "gone"):
        (tmp_path / "gone").mkdir(exist_ok=True)
        _, text = _run(directory)
        lines = _verdict_lines(text)
        assert len(lines) == 1, lines
        fields = lines[0].split(" ")
        assert fields[0] == gate.VERDICT_TOKEN
        assert fields[1] in {"PASS", "REFUSE", "INCONCLUSIVE", "ERROR"}
        parsed = dict(f.split("=", 1) for f in fields[2:])
        for required in ("reason", "config", "strict"):
            assert required in parsed, parsed
        assert all(" " not in v for v in parsed.values())


def test_exit_code_contract():
    assert gate.EXIT_PASS == 0
    assert gate.EXIT_REFUSE == 2
    assert gate.EXIT_ERROR == 3
    assert gate.EXIT_INCONCLUSIVE == 4
    assert gate.GateResult(status="PASS", reason="OK").exit_code == 0
    assert gate.GateResult(status="REFUSE", reason="X").exit_code == 2
    assert gate.GateResult(status="INCONCLUSIVE", reason="X").exit_code == 4
    assert gate.GateResult(status="ERROR", reason="X").exit_code == 3


def test_unregistered_config_name_is_refused_not_silently_substituted(capsys):
    """``get_config`` falls back to ``pi05_b1k-base``; the gate must not accept that."""
    rc = gate.main(["--config", "definitely-not-a-registered-config-name"])
    captured = capsys.readouterr().out
    assert rc == 2, captured
    assert "CONFIG_NOT_REGISTERED" in captured
    assert "SILENTLY fall back" in captured
    # positive control: a real registered name gets past this check (proves the
    # refusal above is about registration, not about every name failing).
    from openpi.training.train_config import _CONFIGS_DICT

    assert "pi05_b1k-base" in _CONFIGS_DICT
    assert gate.load_registered_config("pi05_b1k-base").name == "pi05_b1k-base"


def test_require_openpi_under_rejects_the_wrong_tree(capsys):
    rc = gate.main(["--config", "pi05_b1k-base", "--require-openpi-under", "/nonexistent/tree"])
    captured = capsys.readouterr().out
    assert rc == 3, captured
    assert "WRONG_OPENPI_TREE" in captured
    # positive control: the real worktree is accepted, so the rejection above is
    # about the path and not unconditional. (Uses a bogus weight dir so the run
    # stops quickly at WEIGHT_FILE_MISSING rather than building a 4B-param model.)
    rc_ok = gate.main(
        [
            "--config",
            "pi05_b1k-base",
            "--require-openpi-under",
            str(_REPO_ROOT / "src"),
            "--weight-dir",
            "/tmp/__no_such_weight_dir__",
            "--hash-mode",
            "none",
        ]
    )
    out_ok = capsys.readouterr().out
    assert "WRONG_OPENPI_TREE" not in out_ok, out_ok
    assert rc_ok == 2, out_ok
    assert "WEIGHT_FILE_MISSING" in out_ok, out_ok
