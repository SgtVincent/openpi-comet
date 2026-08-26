# ruff: noqa: SLF001 - contract tests intentionally inspect private state.
"""CPU contracts for the formal 4x8 NVIDIA_H20 BF16 pi0.5-KI Variant A two-arm run.

The experiment holds everything fixed except the warm-start package, so these
tests exist mainly to protect the *controlled* part of "controlled experiment":

  * both arms share one FAST action-token capacity, so capacity can never become
    a third confound alongside weights + normalization
  * each arm's weights are paired with the ``assets`` that ship beside them
  * neither arm references saiwenresearch, which the H20 pool cannot mount
  * the arms differ in exactly the fields we intend, and nothing else
  * the formal arms declare B32 x W32 x GA1 = 1024 and an epoch-DERIVED budget,
    and the trainer's formal contract table agrees with the registered configs
  * the launcher gates on H20 (not A100) and runs the per-device BF16 preflight
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
import re

import pytest

from openpi.training.train_config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _REPO_ROOT / "scripts" / "run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh"
_DS_CONFIG = _REPO_ROOT / "configs" / "deepspeed_zero2_h20_bf16.json"
_ACCEL_CONFIG = _REPO_ROOT / "configs" / "accelerate_ds_zero2_h20_bf16.yaml"

_ARM_A = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16"
_ARM_B = "pi05_ki_joint_fast_b1k-full_task-ki_on_h20_pi05base_bf16"
_ARM_A_SMOKE = f"{_ARM_A}_smoke"
_ARM_B_SMOKE = f"{_ARM_B}_smoke"
_FORMAL = (_ARM_A, _ARM_B)
_SMOKE = (_ARM_A_SMOKE, _ARM_B_SMOKE)
_ALL = (_ARM_A, _ARM_B, _ARM_A_SMOKE, _ARM_B_SMOKE)

_EXPECTED_CAP = 256

# World size of the formal 4x8 topology. Not a config field, so it stays a
# literal here, exactly as the launcher's TOTAL_GPUS does.
_WORLD = 32

# The formal recipe, expressed the way the configs express it. num_train_steps and
# decay_steps are 0 SENTINELS: the budget is derived at runtime as
# num_train_epochs x steps_per_epoch, so there is no step literal to pin. The
# cadences are in samples so they keep their meaning across batch changes.
_FORMAL_STRIDE = 4
_FORMAL_BATCH_PER_GPU = 32
_FORMAL_GLOBAL_BATCH = 1024
_FORMAL_VAL_SAMPLES = 256_000
_FORMAL_SAVE_SAMPLES = 2_560_000
_FORMAL_WARMUP = 250
_FORMAL_PEAK_LR = 2e-5


@pytest.mark.parametrize("name", _ALL)
def test_variant_a_shape(name: str) -> None:
    cfg = get_config(name)
    assert cfg.name == name
    assert cfg.pytorch_model_name == "pi05_ki_joint_fast"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.pytorch_training_precision == "bfloat16"
    assert cfg.accelerate_mixed_precision == "bf16"
    assert cfg.model.knowledge_insulation is True
    assert cfg.model.truncate_expert_kv is True
    assert cfg.gradient_checkpointing is True
    assert cfg.wandb_enabled is True
    # Budget FORM, per arm, and neither may drift into the other's:
    #   formal -> epoch-expressed, with the step fields left as 0 sentinels so the
    #             trainer derives them from the dataset and the global batch.
    #   smoke  -> a small fixed step budget; it must never acquire an epoch budget,
    #             which would make the bounded memory probe unbounded.
    if name in _FORMAL:
        assert cfg.num_train_epochs == 1
        assert cfg.num_train_steps == 0
        assert cfg.lr_schedule.decay_steps == 0
    else:
        assert cfg.num_train_epochs is None
        assert cfg.num_train_steps > 0


@pytest.mark.parametrize(
    ("name", "expected_per_gpu", "expected_global"),
    [
        (_ARM_A, _FORMAL_BATCH_PER_GPU, _FORMAL_GLOBAL_BATCH),
        (_ARM_B, _FORMAL_BATCH_PER_GPU, _FORMAL_GLOBAL_BATCH),
        # The smokes deliberately keep B8: the smoke exists to produce a real
        # measured memory peak for a batch that has already been run, so lowering
        # or raising it would break comparability with the earlier bounded runs.
        (_ARM_A_SMOKE, 8, 256),
        (_ARM_B_SMOKE, 8, 256),
    ],
)
def test_global_batch_matches_each_arms_declared_shape(
    name: str, expected_per_gpu: int, expected_global: int
) -> None:
    cfg = get_config(name)
    assert cfg.batch_size_per_gpu == expected_per_gpu
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.batch_size_per_gpu * _WORLD * cfg.gradient_accumulation_steps == expected_global


def test_cap_is_identical_across_all_four_configs() -> None:
    """The cap must never differ between arms.

    Both arms share one value so FAST capacity cannot act as a third confound.
    This is safe precisely because padded positions carry mask/ar_mask/loss_mask
    False/0/False and the action objective divides by ``shift_loss_mask.sum()``,
    so raising the cap does not rescale CE or accuracy.
    """
    caps = {name: get_config(name).model.action_token_max_len for name in _ALL}
    assert set(caps.values()) == {_EXPECTED_CAP}, caps


@pytest.mark.parametrize("name", _ALL)
def test_no_lq_paths_anywhere(name: str) -> None:
    """H20 mounts behavior-data-hl / navigation-hl / robot-mllm-data-hl only."""
    cfg = get_config(name)
    probes = [str(cfg.pytorch_weight_path)]
    for group in (cfg.data, cfg.val_data):
        for data_cfg in group:
            probes.append(str(data_cfg.assets.assets_dir))
            probes.append(str(data_cfg.base_config.behavior_dataset_root))
    for probe in probes:
        assert "saiwenresearch" not in probe, probe


@pytest.mark.parametrize(("name", "expected_leaf"), [
    (_ARM_A, "pi05-b1kpt50-cs32"),
    (_ARM_A_SMOKE, "pi05-b1kpt50-cs32"),
    (_ARM_B, "pi05_base_pytorch"),
    (_ARM_B_SMOKE, "pi05_base_pytorch"),
])
def test_assets_come_from_the_same_package_as_the_weights(name: str, expected_leaf: str) -> None:
    """Each arm must use the norm_stats that ships with its own weights.

    This is the pairing the whole comparison rests on: the flow expert and action
    head were fit in that normalization space. Crossing them would both corrupt
    the warm start and void the measured token-length bound.
    """
    cfg = get_config(name)
    weights = str(cfg.pytorch_weight_path)
    assert weights.endswith(expected_leaf), weights
    for group in (cfg.data, cfg.val_data):
        for data_cfg in group:
            assert str(data_cfg.assets.assets_dir) == f"{weights}/assets"
            assert data_cfg.assets.asset_id == "behavior-1k/2025-challenge-demos"


def test_the_two_formal_arms_differ_only_in_the_warm_start_package() -> None:
    """Any drift beyond the intended fields turns the A/B into an uncontrolled test."""
    a = dataclasses.asdict(get_config(_ARM_A))
    b = dataclasses.asdict(get_config(_ARM_B))
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    # name/exp_name plus the identity-bearing path fields, and the output roots
    # derived from the name. Notably NOT: model, batch, schedule, stride, budget.
    allowed = {"name", "exp_name", "pytorch_weight_path", "data", "val_data",
               "assets_base_dir", "checkpoint_base_dir", "log_base_dir"}
    assert differing <= allowed, f"unexpected divergence between arms: {differing - allowed}"
    # And prove the model config itself is byte-identical, objective included.
    assert a["model"] == b["model"]


def test_formal_arms_carry_the_derived_single_sweep_recipe() -> None:
    """Both arms must carry one identical, epoch-derived stride-4 recipe.

    Nothing here is a step budget. ``num_train_steps`` and ``decay_steps`` are 0
    sentinels: the trainer derives ``num_train_steps = num_train_epochs x
    steps_per_epoch`` from the dataset and the resolved global batch, and auto-sets
    ``decay_steps`` to it. Asserting the sentinels is what proves the derivation is
    ARMED -- a literal in either field would cap the budget or decouple the cosine
    decay from it, which is exactly the failure the sentinels exist to prevent.

    The two cadences are asserted in samples AND in their derived step equivalents,
    because a sample-valued cadence that rounds to zero steps would silently
    disable validation or checkpointing.
    """
    for name in _FORMAL:
        cfg = get_config(name)
        # Derivation armed, not a remembered budget.
        assert cfg.num_train_epochs == 1
        assert cfg.num_train_steps == 0
        assert cfg.lr_schedule.decay_steps == 0
        # One single sweep: a stride with no per-epoch offset rotation, so the
        # trainer never rebuilds the loader mid-run.
        assert cfg.streaming_anchor_stride == _FORMAL_STRIDE
        assert cfg.epoch_anchor_offsets is None
        # Batch-invariant cadences. The step-valued save_interval /
        # val_log_interval fields are placeholders that train_accelerate.py
        # overwrites from these, so they are deliberately not asserted here.
        assert cfg.val_interval_samples == _FORMAL_VAL_SAMPLES
        assert cfg.save_interval_samples == _FORMAL_SAVE_SAMPLES
        assert cfg.checkpoint_policy == "epoch_with_rolling"
        assert cfg.rolling_checkpoint_interval == 2_500
        global_batch = cfg.batch_size_per_gpu * _WORLD * cfg.gradient_accumulation_steps
        assert global_batch == _FORMAL_GLOBAL_BATCH
        assert cfg.val_interval_samples // global_batch == 250
        assert cfg.save_interval_samples // global_batch == 2_500
        assert cfg.val_num_batches == 20
        assert cfg.project_name == "pi05_ki"
        sched = cfg.lr_schedule
        assert (
            int(sched.warmup_steps),
            float(sched.peak_lr),
            int(sched.decay_steps),
            float(sched.decay_lr),
        ) == (_FORMAL_WARMUP, _FORMAL_PEAK_LR, 0, 0.0)


def test_smoke_arms_are_bounded_and_keep_the_b8_memory_probe() -> None:
    """The smoke exists to measure real memory at B8, so B8 must not be changed.

    B8 is no longer the formal per-GPU batch (the formal arms now run B32), but the
    smoke keeps it deliberately: its purpose is a bounded, already-proven-shape run,
    and it must also keep the step-valued cadence rather than the formal
    sample-valued one, which would round to a single step at this batch.
    """
    for name in _SMOKE:
        cfg = get_config(name)
        assert 0 < cfg.num_train_steps <= 16
        assert cfg.streaming_anchor_stride == 1
        assert cfg.batch_size_per_gpu == 8
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.model.action_token_max_len == _EXPECTED_CAP
        assert cfg.val_interval_samples is None
        assert cfg.save_interval_samples is None
        assert cfg.checkpoint_policy == "step"


def test_smoke_must_actually_reach_validation() -> None:
    """A smoke that never calls compute_eval_metrics is not a gate.

    Variant A's known historical failure is *inside* validation: the trainer
    passes ``deterministic_flow`` to both KI variants through one shared
    ``is_pi05_ki_joint`` branch, and Variant A's override of
    ``compute_eval_metrics`` did not accept it -- which killed an A100 FAST run at
    its first validation after ~2h40m of training. Validation fires on
    ``global_step % val_log_interval == 0 and global_step > 0``, so the interval
    must be strictly below the budget (an equal interval puts the only validation
    on the termination boundary) and must yield at least two passes.
    """
    for name in (_ARM_A_SMOKE, _ARM_B_SMOKE):
        cfg = get_config(name)
        assert cfg.val_log_interval < cfg.num_train_steps, (
            f"{name}: val_log_interval={cfg.val_log_interval} >= "
            f"num_train_steps={cfg.num_train_steps}, so validation would land on or "
            "past the termination boundary and might never run"
        )
        assert cfg.num_train_steps // cfg.val_log_interval >= 2, (
            f"{name}: only {cfg.num_train_steps // cfg.val_log_interval} validation "
            "pass(es) in the budget; need >= 2"
        )


def test_step0_capability_gate_blocks_the_regression_and_the_bad_fixes(tmp_path) -> None:
    """The step-0 gate must reject the original bug AND the tempting bad fixes.

    The cost of this bug class was never the TypeError itself -- it was that it
    fired at the first validation, hours into a 32-GPU run, and that a smoke
    shorter than one validation interval could not reach it. So the gate is
    checked here against four implementations.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_ta_capcheck", _REPO_ROOT / "scripts" / "train_accelerate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ta_capcheck"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass

    cfg = get_config(_ARM_A_SMOKE)
    assert cfg.val_deterministic_flow is True, (
        "this gate is only meaningful while the flag is requested ON"
    )

    # File-backed so inspect.getsource works, exactly as for real model classes.
    mod_path = tmp_path / "_capcheck_models.py"
    mod_path.write_text(
        "class Honoured:\n"
        "    def compute_eval_metrics(self, o, a, *, compute_flow_l1=False,\n"
        "                             num_denoise_steps=10, flow_l1_seed=42,\n"
        "                             deterministic_flow=False):\n"
        "        if deterministic_flow:\n"
        "            import torch\n"
        "            torch.manual_seed(flow_l1_seed)\n"
        "        return {}\n"
        "class Regressed:\n"
        "    def compute_eval_metrics(self, o, a, *, compute_flow_l1=False,\n"
        "                             num_denoise_steps=10, flow_l1_seed=42):\n"
        "        return {}\n"
        "class Absorber:\n"
        "    def compute_eval_metrics(self, o, a, **kwargs):\n"
        "        return {}\n"
        "class NoOp:\n"
        "    def compute_eval_metrics(self, o, a, *, compute_flow_l1=False,\n"
        "                             num_denoise_steps=10, flow_l1_seed=42,\n"
        "                             deterministic_flow=False):\n"
        "        return {}\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        spec2 = importlib.util.spec_from_file_location("_capcheck_models", mod_path)
        models = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(models)

        # The correct implementation passes.
        module._validate_ki_eval_capability(models.Honoured(), cfg, is_main=False)

        # The original bug: TypeError at first validation.
        with pytest.raises(RuntimeError, match="deterministic_flow"):
            module._validate_ki_eval_capability(models.Regressed(), cfg, is_main=False)

        # The forbidden **kwargs "fix": converts a loud crash into total silence.
        with pytest.raises(RuntimeError, match="silently swallowed"):
            module._validate_ki_eval_capability(models.Absorber(), cfg, is_main=False)

        # Signature parity with a no-op body: accepted and ignored.
        with pytest.raises(RuntimeError, match="never referenced in its body"):
            module._validate_ki_eval_capability(models.NoOp(), cfg, is_main=False)
    finally:
        sys.path.remove(str(tmp_path))


def test_both_ki_variants_accept_the_shared_trainer_call_surface() -> None:
    """Guard the bug class, not just the one instance that bit us.

    ``is_pi05_ki_joint`` treats the two classes as interchangeable while they are
    maintained separately, so any kwarg added to one implementation and to the
    call site -- but not to the other implementation -- is a latent TypeError that
    only fires at validation. Assert full signature parity on every method FAST
    overrides, except the two deliberate fail-closed ``**kwargs`` stubs.
    """
    import inspect

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    # These two are intentionally divergent: FAST replaces them with
    # NotImplementedError stubs because they are Variant-B-only code paths.
    intentional_stubs = {"_embed_query_tokens", "_compute_query_mse_loss"}

    q_attrs, f_attrs = vars(Query), vars(Fast)
    overridden = [
        name
        for name in f_attrs
        if name in q_attrs and callable(f_attrs[name]) and not name.startswith("__")
    ]
    assert overridden, "expected FAST to override at least compute_eval_metrics"

    divergent = {}
    for name in overridden:
        if name in intentional_stubs:
            continue
        sig_q = inspect.signature(q_attrs[name])
        sig_f = inspect.signature(f_attrs[name])
        if str(sig_q) != str(sig_f):
            divergent[name] = (str(sig_q), str(sig_f))
    assert not divergent, f"signature divergence between KI variants: {divergent}"

    # And the stubs really are fail-closed rather than silently absorbing calls.
    for name in intentional_stubs:
        assert name in f_attrs, f"{name} should still be overridden in FAST"
        src = inspect.getsource(f_attrs[name])
        assert "NotImplementedError" in src, f"{name} must fail closed, not absorb the call"


def test_fast_eval_metrics_honours_deterministic_flow_the_same_way() -> None:
    """Signature parity is not enough -- the behaviour must match.

    If Variant B makes flow sampling deterministic and Variant A merely absorbs
    the kwarg, the two arms would differ in validation determinism, which is a
    confound in the very comparison this pair exists to run. ``val_deterministic
    _flow`` defaults to True, so this path is live, not hypothetical.
    """
    import inspect

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    for cls in (Query, Fast):
        src = inspect.getsource(cls.compute_eval_metrics)
        assert "deterministic_flow" in src
        # fixed (noise, time) drawn from flow_l1_seed
        assert "torch.manual_seed(flow_l1_seed)" in src, cls.__name__
        # no random image augmentation on the deterministic path
        assert "train_preprocess=False" in src, cls.__name__
        # CUDA generators restored explicitly: manual_seed reseeds CPU *and* CUDA,
        # while get/set_rng_state covers only CPU, so omitting this leaks a CUDA
        # reseed into the training RNG stream.
        assert "set_rng_state_all" in src, cls.__name__


def test_formal_protocol_selects_both_h20_profiles_without_smoke_leakage() -> None:
    source = (_REPO_ROOT / "scripts" / "train_accelerate.py").read_text()
    tree = ast.parse(source)
    policies = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_FORMAL_B1K_CHECKPOINT_POLICIES"
            for target in node.targets
        ):
            policies = ast.literal_eval(node.value)
            break
    assert policies is not None
    for name in _FORMAL:
        assert policies[name] == "epoch_with_rolling"
    for name in _SMOKE:
        assert name not in policies


def test_h20_no_offload_policy_covers_all_four_configs() -> None:
    source = (_REPO_ROOT / "scripts" / "train_accelerate.py").read_text()
    block = source.split("_H20_BF16_NO_OPTIMIZER_OFFLOAD_CONFIGS = {", 1)[1].split("}", 1)[0]
    for name in _ALL:
        assert name in block, f"{name} not covered by the H20 no-offload policy"


def test_deepspeed_config_is_zero2_bf16_without_offload() -> None:
    ds = json.loads(_DS_CONFIG.read_text())
    zero = ds["zero_optimization"]
    assert zero["stage"] == 2
    assert "offload_optimizer" not in zero
    assert "offload_param" not in zero
    assert ds["bf16"]["enabled"] is True
    assert ds["fp16"]["enabled"] is False
    assert ds["gradient_accumulation_steps"] == "auto"


def test_accelerate_config_defers_precision_to_deepspeed() -> None:
    text = _ACCEL_CONFIG.read_text()
    assert "deepspeed_config_file: configs/deepspeed_zero2_h20_bf16.json" in text
    assert re.search(r"^mixed_precision\s*:", text, re.MULTILINE) is None


def test_launcher_gates_on_h20_and_not_a100() -> None:
    text = _LAUNCHER.read_text()
    assert "this BF16 launcher is H20-only" in text
    # Inspect the executable GPU-model gate itself rather than the whole file, so
    # a comment that merely *mentions* A100 cannot fail (or pass) this test.
    gate_lines = [
        line
        for line in text.splitlines()
        if "GPU_MODEL^^" in line and not line.lstrip().startswith("#")
    ]
    assert len(gate_lines) == 1, f"expected exactly one GPU-model gate, got {gate_lines}"
    gate = gate_lines[0]
    assert "*H20*" in gate, gate
    assert "A100" not in gate, f"H20 launcher must not gate on A100: {gate}"


def test_launcher_runs_the_per_device_bf16_preflight() -> None:
    """A GPU-0-only probe would let a node with one bad GPU reach c10d bootstrap."""
    text = _LAUNCHER.read_text()
    assert "cuda_preflight_all_devices.py" in text
    assert "--require-bf16" in text
    assert "--min-driver-major 525" in text
    assert (_REPO_ROOT / "scripts" / "cuda_preflight_all_devices.py").is_file()


def test_launcher_enforces_the_key_invariants() -> None:
    text = _LAUNCHER.read_text()
    # provenance + clean tree + import path
    assert "OPENPI_EXPECTED_CODE_COMMIT" in text
    assert "^[0-9a-f]{40}$" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "openpi import does not resolve inside the pinned tree" in text
    # The registered TrainConfig is resolved exactly and owns the batch recipe.
    assert "openpi.training.launcher_profile" in text
    assert "CFG_BATCH_SIZE_PER_GPU" in text
    assert "CFG_GRADIENT_ACCUMULATION_STEPS" in text
    assert "CFG_EXPECTED_GLOBAL_BATCH" in text
    assert '"${GLOBAL_BATCH_SIZE}" -eq "${CFG_EXPECTED_GLOBAL_BATCH}"' in text
    assert "CFG_EFFECTIVE_SAVE_INTERVAL" in text
    assert "CFG_EFFECTIVE_VAL_LOG_INTERVAL" in text
    # The visible cadence is effective, while exact Tyro argv must preserve the
    # raw registered authorities. The trainer materializes the same pure resolver
    # after world size is known; passing effective values here would look like a
    # forbidden direct override before materialization.
    assert "cadence save=${CFG_EFFECTIVE_SAVE_INTERVAL} val=${CFG_EFFECTIVE_VAL_LOG_INTERVAL}" in text
    assert '--save-interval "${CFG_SAVE_INTERVAL}"' in text
    assert '--val-log-interval "${CFG_VAL_LOG_INTERVAL}"' in text
    assert '--save-interval "${CFG_EFFECTIVE_SAVE_INTERVAL}"' not in text
    assert '--val-log-interval "${CFG_EFFECTIVE_VAL_LOG_INTERVAL}"' not in text
    assert "--batch-size-per-gpu 8" not in text
    # And the derivation must stay armed in the formal-mode preflight.
    assert "steps derived at runtime (num_train_steps==0)" in text
    assert "decay derived at runtime (decay_steps==0)" in text
    assert "no per-epoch anchor offsets (offset machinery removed)" in text
    # occupier handoff
    assert "__GPU_OCCUPY__torch_mm_512" in text
    # warm-start mapping is proven before GPUs are spent
    assert "verify_warm_start_keymap.py" in text
    # the working conda env, and an explicit warning about the broken one
    assert "behavior-data-hl/chenjunting/miniconda3" in text
    assert "GemmaForCausalLM" in text


def test_launcher_pins_each_arm_to_its_own_norm_stats_digest() -> None:
    """The digest assert is the cheap hard proof of the weight/normalization pair."""
    text = _LAUNCHER.read_text()
    assert "d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929" in text
    assert "4dde119e69123ed865072c71a714095ae746c6d294fefba910a842757a7083ce" in text
    assert "norm_stats digest mismatch" in text
    # and the weight-size assert that distinguishes fp32 Arm A from bf16 Arm B
    assert "14467165872" in text
    assert "7233650408" in text


def _make_eval_probe(cls, n_backbone_outputs: int):
    """Lightweight subclass that runs the REAL compute_eval_metrics.

    Only the expensive collaborators are stubbed, so the deterministic_flow code
    path itself -- manual_seed, the CPU/CUDA state save-restore, the (noise, time)
    draw and the train_preprocess passthrough -- is genuinely executed. Building
    the real 4.1B-parameter module would make this test unusable, and asserting
    only on the signature is what allowed the original bug through in the first
    place.
    """
    import torch

    class _Probe(cls):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.calls: list[dict] = []

        def _compute_backbone_eval_metrics(self, observation, actions):
            z = torch.zeros((), dtype=torch.float32)
            return tuple(z.clone() for _ in range(n_backbone_outputs))

        def compute_expert_loss(
            self, observation, actions, noise=None, time=None, *, train_preprocess=True
        ):
            self.calls.append(
                {
                    "noise": None if noise is None else noise.detach().clone(),
                    "time": None if time is None else time.detach().clone(),
                    "train_preprocess": train_preprocess,
                }
            )
            # Make the reported metric a function of (noise, time) so that
            # identical draws must produce bit-identical flow metrics.
            val = (
                (noise.double().sum() + time.double().sum()).float()
                if noise is not None
                else torch.zeros((), dtype=torch.float32)
            )
            return {"expert_loss": val, "flow_loss": val}

    return _Probe()


@pytest.mark.parametrize(("variant", "n_backbone"), [("fast", 5), ("query", 5)])
def test_deterministic_flow_is_actually_honoured_and_leaves_rng_untouched(
    variant: str, n_backbone: int
) -> None:
    """Behaviour test, not a signature test.

    Three properties, all of which the original Variant A silently lacked:
      1. two calls with deterministic_flow=True yield bit-identical flow metrics
      2. the deterministic path disables random image augmentation
      3. the global RNG stream is left byte-identical, so validation cannot
         perturb training. torch.manual_seed() reseeds CPU *and* all CUDA
         generators, so a partial restore would leak into the training stream.
    """
    import torch

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    cls = Fast if variant == "fast" else Query
    probe = _make_eval_probe(cls, n_backbone)
    actions = torch.zeros((2, 4, 8), dtype=torch.float32)

    torch.manual_seed(1234)
    state_before = torch.get_rng_state()

    m1 = probe.compute_eval_metrics(None, actions, deterministic_flow=True, flow_l1_seed=777)
    m2 = probe.compute_eval_metrics(None, actions, deterministic_flow=True, flow_l1_seed=777)

    state_after = torch.get_rng_state()

    # 1. identical (noise, time) => bit-identical flow metrics
    assert torch.equal(probe.calls[0]["noise"], probe.calls[1]["noise"]), (
        f"{variant}: deterministic_flow did not fix the noise draw"
    )
    assert torch.equal(probe.calls[0]["time"], probe.calls[1]["time"]), (
        f"{variant}: deterministic_flow did not fix the time draw"
    )
    assert torch.equal(m1["flow_loss"], m2["flow_loss"]), (
        f"{variant}: flow_loss is not reproducible under deterministic_flow"
    )
    assert torch.equal(m1["expert_loss"], m2["expert_loss"])
    assert torch.equal(m1["total_loss"], m2["total_loss"])

    # 2. no random image augmentation on the deterministic path
    assert probe.calls[0]["train_preprocess"] is False, (
        f"{variant}: deterministic_flow must preprocess with train=False"
    )
    assert probe.calls[1]["train_preprocess"] is False

    # 3. training RNG stream untouched
    assert torch.equal(state_before, state_after), (
        f"{variant}: validation perturbed the global CPU RNG state; manual_seed reseeds "
        "CPU and CUDA, so both must be saved and restored"
    )


@pytest.mark.parametrize(("variant", "n_backbone"), [("fast", 5), ("query", 5)])
def test_deterministic_flow_off_uses_the_random_path(variant: str, n_backbone: int) -> None:
    """With the flag off, the model must delegate to the default random draw.

    This is the control for the test above: if the implementation ignored the flag
    entirely, both branches would look identical and the test above would pass for
    the wrong reason.
    """
    import torch

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    cls = Fast if variant == "fast" else Query
    probe = _make_eval_probe(cls, n_backbone)
    actions = torch.zeros((2, 4, 8), dtype=torch.float32)

    probe.compute_eval_metrics(None, actions, deterministic_flow=False)
    call = probe.calls[0]
    assert call["noise"] is None, f"{variant}: flag off must not inject a fixed noise"
    assert call["time"] is None, f"{variant}: flag off must not inject a fixed time"
    assert call["train_preprocess"] is True, (
        f"{variant}: flag off must keep the default train-time preprocessing"
    )


def test_both_variants_produce_identical_deterministic_draws() -> None:
    """The two arms must be equally deterministic, or determinism is a confound.

    Same seed, same action shape: both variants must draw the same (noise, time),
    otherwise their validation losses are not comparable on equal terms even
    though both nominally honour the flag.
    """
    import torch

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    actions = torch.zeros((3, 4, 8), dtype=torch.float32)
    draws = {}
    for label, cls in (("fast", Fast), ("query", Query)):
        probe = _make_eval_probe(cls, 5)
        torch.manual_seed(99)
        probe.compute_eval_metrics(None, actions, deterministic_flow=True, flow_l1_seed=4242)
        draws[label] = (probe.calls[0]["noise"], probe.calls[0]["time"])

    assert torch.equal(draws["fast"][0], draws["query"][0]), (
        "arms draw different validation noise under the same seed"
    )
    assert torch.equal(draws["fast"][1], draws["query"][1]), (
        "arms draw different validation time under the same seed"
    )


def test_flow_l1_path_rng_handling_is_shared_not_duplicated() -> None:
    """The second RNG site must not become a second place to get this wrong.

    Variant B has TWO independent CPU+CUDA save/restore blocks: one in
    ``compute_eval_metrics`` and one in ``_compute_flow_l1_euler`` (the Euler
    integration path reached when ``compute_flow_l1=True``). Variant A only
    re-implements the first. It gets the second for free by inheritance, which is
    exactly why that site never diverged -- the entire bug class lives in the set
    of methods FAST *overrides*.

    A partially deterministic validation is the worst outcome: it would pass a
    two-call bit-identity test on the fixed path and still leak nondeterminism
    through the unfixed one. So assert the sharing explicitly, and this test fails
    if someone later re-implements the method in FAST without the RNG handling.
    """
    import inspect

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    assert Fast._compute_flow_l1_euler is Query._compute_flow_l1_euler, (
        "FAST has re-implemented _compute_flow_l1_euler; it must then carry its own "
        "CPU+CUDA RNG save/restore, or validation becomes only partially deterministic"
    )
    src = inspect.getsource(Query._compute_flow_l1_euler)
    assert "torch.manual_seed(seed)" in src
    assert "torch.get_rng_state()" in src and "torch.set_rng_state(" in src
    assert "get_rng_state_all" in src and "set_rng_state_all" in src


@pytest.mark.parametrize("variant", ["fast", "query"])
def test_deterministic_flow_holds_with_flow_l1_enabled_too(variant: str) -> None:
    """Cover the compute_flow_l1=True branch, which reaches the second RNG site.

    The heavy Euler integration is stubbed (it needs real model internals), so what
    this asserts is the part living in compute_eval_metrics: the configured seed is
    threaded through to the flow-L1 path, results stay bit-identical across calls,
    and the global RNG stream is still untouched with the slow metric enabled.
    """
    import torch

    from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch as Fast
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch as Query

    cls = Fast if variant == "fast" else Query
    probe = _make_eval_probe(cls, 5)
    seeds: list[int] = []

    def _stub_flow_l1(*, observation, actions, num_steps, seed):
        seeds.append(seed)
        return torch.full((), float(seed), dtype=torch.float32)

    probe._compute_flow_l1_euler = _stub_flow_l1  # noqa: SLF001
    actions = torch.zeros((2, 4, 8), dtype=torch.float32)

    torch.manual_seed(555)
    before = torch.get_rng_state()
    m1 = probe.compute_eval_metrics(
        None, actions, compute_flow_l1=True, deterministic_flow=True, flow_l1_seed=31337
    )
    m2 = probe.compute_eval_metrics(
        None, actions, compute_flow_l1=True, deterministic_flow=True, flow_l1_seed=31337
    )
    after = torch.get_rng_state()

    assert "flow_l1" in m1 and "flow_l1" in m2, f"{variant}: flow_l1 not emitted when requested"
    assert seeds == [31337, 31337], f"{variant}: flow_l1_seed not threaded through: {seeds}"
    assert torch.equal(m1["flow_l1"], m2["flow_l1"])
    assert torch.equal(m1["flow_loss"], m2["flow_loss"])
    assert torch.equal(before, after), (
        f"{variant}: RNG stream perturbed with compute_flow_l1=True"
    )


def test_cap_provenance_comment_is_exhaustive_and_does_not_reuse_the_4dde119e_ids() -> None:
    """The cap comment must state what was actually measured, under which norm_stats.

    Reusing the `4dde119e` exhaustive provenance would be a false claim, and now a
    demonstrably false one: the `d66ed168` train max is 200, which exceeds that
    population's 199. The comment must also stay honest that 208 would have
    sufficed, so nobody reads 256 as vindicated.
    """
    text = (_REPO_ROOT / "src" / "openpi" / "training" / "pi05_ki_joint_query_config.py").read_text()
    block = text.split("_H20_FAST_ACTION_TOKEN_MAX_LEN", 1)[0][-6000:]

    # states its own strength, and the normalization it applies to
    assert "EXHAUSTIVELY VERIFIED" in block
    assert "d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929" in block
    # the measured populations and maxima, in full
    assert "26,857,712" in block and "max 200" in block
    assert "11,398,271" in block and "max 189" in block
    assert "exhaustive_W_cap256_20260824_124043" in block
    # the old provenance may be NAMED, but only as belonging to the other population
    if "0bb9280746" in block:
        assert "4dde119e" in block and "NOT reused" in block
    # and the unflattering half must be stated, not quietly dropped
    assert "208 would in fact have sufficed" in block
    assert "NOT presented as vindicated" in block


def test_cap_comment_does_not_claim_the_old_bound_transfers() -> None:
    """The measured fact that makes the old bound non-transferable must be recorded."""
    text = (_REPO_ROOT / "src" / "openpi" / "training" / "pi05_ki_joint_query_config.py").read_text()
    block = text.split("_H20_FAST_ACTION_TOKEN_MAX_LEN", 1)[0][-6000:]
    assert "EXCEEDS" in block, "the 200 > 199 fact must be explicit"
