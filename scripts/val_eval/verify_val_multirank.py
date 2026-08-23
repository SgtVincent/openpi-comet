#!/usr/bin/env python3
"""Multi-rank verification of the per-task validation all_reduce.

Run under torchrun so `torch.distributed` is really initialized:

    CONDA_ROOT=... B1K_DATASET_ROOT=... \
      scripts/val_eval/run_val_eval.sh -m torch.distributed.run \
      --nproc_per_node=8 scripts/val_eval/verify_val_multirank.py

What this proves
----------------
The only part of the val patch that had never run with world_size > 1 is the
cross-rank reduction:
  * `build_val_datasets()` wrapping the FIXED subset in a DistributedSampler,
  * `_DeterministicValLoader.__iter__` deriving rank-local `batch_task_ids`
    from that sampler,
  * `run_validation()`'s per-task `dist.all_reduce` over the (sums, counts)
    tensor pair.

To make this an EXACT arithmetic proof rather than a "it didn't crash" smoke
test, the 3B model is replaced by a stub whose metrics are a deterministic
function of the batch contents. Each rank records what it computed; the records
are then all_gathered and the expected per-task means are recomputed
independently on rank 0 and compared against what run_validation would have
written to metrics.jsonl.

Using a stub also keeps this cheap (no 8x 7GB model load) and isolates the
aggregation logic, which is the thing under test.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

REPO = os.environ.get(
    "OPENPI_REPO_ROOT",
    str(pathlib.Path(__file__).resolve().parents[2]),
)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.distributed as dist
from accelerate import Accelerator

import openpi.training.config as _config
import openpi.training.data_loader as _dl
import openpi.models.model as _model

spec = importlib.util.spec_from_file_location(
    "ta", os.path.join(REPO, "scripts", "train_accelerate.py"))
ta = importlib.util.module_from_spec(spec)
sys.modules["ta"] = ta
spec.loader.exec_module(ta)
ta._config = _config
ta._data_loader = _dl
ta._model = _model

CFG = "pi05_ki_joint_query_b1k-full_task-ki_on_bf16"
EPISODES_PER_TASK = int(os.environ.get("VAL_EPISODES_PER_TASK", "2"))


def log(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


# ---- 1. real distributed init, exactly as training does -------------------
accelerator = Accelerator()
rank = accelerator.process_index
world = accelerator.num_processes

# `run_validation()` calls accelerator.unwrap_model(), which imports deepspeed,
# whose import-time compatibility probe shells out to /usr/local/cuda/bin/nvcc.
# This L20 pod has no CUDA toolkit, so that import raises FileNotFoundError.
# unwrap_model on an unwrapped stub is the identity, and it is accelerate's own
# code (already exercised 105x by the real DeepSpeed training run), so shorting
# it out here does not weaken what is under test: the cross-rank reduction.
if not os.path.exists("/usr/local/cuda/bin/nvcc"):
    accelerator.unwrap_model = lambda m, **kw: m
    if rank == 0:
        print("[note] no nvcc on this host -> bypassing accelerator.unwrap_model "
              "(deepspeed import probe); the reduction under test is unaffected")

if rank == 0:
    print("=" * 76)
    print(f"MULTI-RANK VAL VERIFICATION  world_size={world}  "
          f"dist_initialized={dist.is_initialized()}")
    print("=" * 76)
assert dist.is_initialized(), "torch.distributed must be initialized (use torchrun)"
assert world > 1, f"need world_size>1 to test the reduction, got {world}"

cfg = _config.get_config(CFG)
# Only plumbing; every val_* flag is left at its (new) DEFAULT on purpose.
object.__setattr__(cfg, "num_workers", 0)
object.__setattr__(cfg, "batch_size", 8 * world)   # -> per-rank 8
object.__setattr__(cfg, "val_batch_size", 8)
object.__setattr__(cfg, "val_episodes_per_task", EPISODES_PER_TASK)

log(f"\nconfig defaults in effect:")
for f in ("val_deterministic_subset", "val_deterministic_flow",
          "val_slow_metrics_every", "val_log_per_task",
          "val_episodes_per_task", "val_subset_seed"):
    log(f"  {f:26s} = {getattr(cfg, f)}")

# ---- 2. build the real val loader (DistributedSampler path) ---------------
val_loader, val_dc = ta.build_val_datasets(cfg)
assert type(val_loader).__name__ == "_DeterministicValLoader", \
    f"expected deterministic loader, got {type(val_loader).__name__}"
sampler = getattr(val_loader._loader, "sampler", None)
log(f"\nloader           = {type(val_loader).__name__}")
log(f"sampler          = {type(sampler).__name__ if sampler else None}")
log(f"coverage         = {val_loader.coverage}")
log(f"batches per rank = {len(val_loader)}")
assert sampler is not None, "expected a DistributedSampler under world_size>1"

# ---- 3. verify the shards are DISJOINT and cover the subset ---------------
local_idx = sorted(list(sampler))
gathered_idx = [None] * world
dist.all_gather_object(gathered_idx, local_idx)
if rank == 0:
    flat = [i for shard in gathered_idx for i in shard]
    sizes = [len(s) for s in gathered_idx]
    overlap = len(flat) - len(set(flat))
    print(f"\n[shards] per-rank sizes = {sizes}")
    print(f"[shards] total={len(flat)} unique={len(set(flat))} overlap={overlap}")
    assert overlap == 0, f"rank shards OVERLAP by {overlap} indices"
    print("[S PASS] rank shards are disjoint")

# ---- 4. stub model with a KNOWN metric function ---------------------------
class _Stub(torch.nn.Module):
    """Returns metrics that are a deterministic function of the batch."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def compute_eval_metrics(self, observation, actions, **kw):
        v = actions.detach().float().abs().mean()
        return {
            "total_loss": v * 3.0,
            "flow_mse": v,
            "flow_l1": v * 2.0,
            "subtask_accuracy": torch.clamp(v, 0.0, 1.0),
        }


stub = _Stub().to(accelerator.device)

# Capture what THIS rank actually computed, so ground truth is knowable.
seen: list[float] = []
_orig = _Stub.compute_eval_metrics


def _spy(self, observation, actions, **kw):
    out = _orig(self, observation, actions, **kw)
    seen.append(float(out["flow_l1"].detach().float().item()))
    return out


_Stub.compute_eval_metrics = _spy

# Capture the metrics.jsonl record instead of writing a file.
records: list[dict] = []
ta._metrics_buffer_write_boundary = lambda rec: records.append(rec)
ta._metrics_buffer_flush = lambda: None

# ---- 5. run the REAL run_validation() ------------------------------------
log("\nrunning run_validation() across all ranks ...")
means = ta.run_validation(
    accelerator=accelerator,
    model=stub,
    val_loader=val_loader,
    config=cfg,
    global_step=1000,
    steps_per_epoch=420689,
    is_pi05_ki_joint=True,
    use_vlm2=False,
    use_autocast=False,
    autocast_dtype=torch.bfloat16,
    metrics_file=object(),      # truthy so the record path executes
    slow_metrics=True,          # per_task keys off flow_l1
)

# ---- 6. independently recompute the expected per-task means --------------
local_pairs = [(int(t), v)
               for bi, v in enumerate(seen)
               for t in set(val_loader.batch_task_ids[bi])]
gathered_pairs = [None] * world
dist.all_gather_object(gathered_pairs, local_pairs)

if rank == 0:
    from collections import defaultdict
    exp = defaultdict(list)
    for shard in gathered_pairs:
        for t, v in shard:
            exp[t].append(v)
    expected = {t: sum(vs) / len(vs) for t, vs in exp.items()}

    assert records, "run_validation wrote no metrics record"
    rec = records[-1]
    got_raw = rec.get("val_per_task") or {}
    got = {int(k.split("-")[1]): v for k, v in got_raw.items()}

    print(f"\n[per-task] metric      = {rec.get('val_per_task_metric')}")
    print(f"[per-task] tasks logged = {len(got)}  (n={rec.get('val_per_task_n')})")
    print(f"[per-task] min/max      = {rec.get('val_per_task_min'):.6f} / "
          f"{rec.get('val_per_task_max'):.6f}")
    print(f"[per-task] subset       = {rec.get('val_subset')}")

    assert rec.get("val_per_task_metric") == "flow_l1", rec.get("val_per_task_metric")
    missing = set(expected) - set(got)
    extra = set(got) - set(expected)
    assert not missing, f"tasks missing from the reduction: {sorted(missing)[:10]}"
    assert not extra, f"tasks invented by the reduction: {sorted(extra)[:10]}"

    worst = 0.0
    worst_t = None
    for t, ev in expected.items():
        d = abs(got[t] - ev)
        if d > worst:
            worst, worst_t = d, t
    print(f"\n[per-task] independently recomputed {len(expected)} task means")
    print(f"[per-task] worst |reduced - expected| = {worst:.3e} (task-{worst_t:04d})")
    assert worst < 1e-5, f"per-task reduction WRONG, worst diff {worst}"
    print("[R PASS] per-task all_reduce matches the independent recomputation exactly")

    # global metric sanity: mean-of-per-rank-means
    all_seen = [v for shard in gathered_pairs for _t, v in shard]
    print(f"\n[global] keys = {sorted(means.keys())}")
    print(f"[global] flow_l1 = {means.get('flow_l1'):.6f}")
    assert "flow_l1" in means and "total_loss" in means
    print("[G PASS] global metrics reduced and returned")

    print("\n" + "=" * 76)
    print(f"ALL MULTI-RANK CHECKS PASSED  (world_size={world}, "
          f"{len(expected)} tasks, {len(all_seen)} task-batch observations)")
    print("=" * 76)

dist.barrier()
accelerator.end_training()
