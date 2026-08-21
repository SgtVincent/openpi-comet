#!/usr/bin/env python3
"""Verify the training-side val patch WITHOUT launching a real training job.

Exercises the exact patched code path:
  build_val_datasets(config) with val_deterministic_subset=True
and checks the three properties the patch is supposed to guarantee:
  P1 task/episode coverage (should be 50 tasks, not 1)
  P2 REPRODUCIBILITY: two independent iterations return the SAME samples
  P3 determinism of flow metrics with val_deterministic_flow=True
"""
from __future__ import annotations
import os, pathlib, sys, importlib.util, time, hashlib
import pathlib

# Repo root is derived from this file's location (scripts/val_eval/<x>.py),
# so the script works from any checkout. Override with OPENPI_REPO_ROOT.
REPO = os.environ.get(
    "OPENPI_REPO_ROOT",
    str(pathlib.Path(__file__).resolve().parents[2]),
)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import openpi.training.config as _config
import openpi.training.data_loader as _dl
import openpi.models.model as _model

# load train_accelerate.py as a module (it is a script, not a package member)
spec = importlib.util.spec_from_file_location("ta", os.path.join(REPO, "scripts", "train_accelerate.py"))
ta = importlib.util.module_from_spec(spec)
sys.modules["ta"] = ta
spec.loader.exec_module(ta)
# populate the lazily-imported globals the script normally sets in main()
ta._config = _config; ta._data_loader = _dl; ta._model = _model
print("[ok] train_accelerate imported; patched helpers present:",
      hasattr(ta, "_build_stratified_val_indices"), hasattr(ta, "_DeterministicValLoader"))

cfg = _config.get_config("pi05_ki_joint_query_b1k-full_task-ki_on_bf16")
# opt in to the new behaviour exactly as a real run would via config
for k, v in [("val_deterministic_subset", True), ("val_episodes_per_task", 10),
             ("val_anchors_per_episode", 1), ("val_subset_seed", 12345),
             ("val_deterministic_flow", True), ("val_log_per_task", True),
             ("val_slow_metrics_every", 1), ("num_workers", 8),
             ("batch_size", 8), ("val_batch_size", 8)]:
    object.__setattr__(cfg, k, v)

t0 = time.time()
val_loader, val_dc = ta.build_val_datasets(cfg)
print(f"[ok] build_val_datasets returned in {time.time()-t0:.1f}s")
print(f"     loader type      = {type(val_loader).__name__}")
print(f"     streaming flag   = {val_dc.chunk_streaming_using_keyframe}  (must be False)")
print(f"     coverage         = {val_loader.coverage}")
print(f"     len(loader)      = {len(val_loader)} batches")

cov = val_loader.coverage
assert cov["n_tasks"] == 50, f"expected 50 tasks, got {cov['n_tasks']}"
assert val_dc.chunk_streaming_using_keyframe is False
print("[P1 PASS] fixed subset covers all 50 tasks "
      f"({cov['n_samples']} samples / {cov['n_episodes']} episodes)")

def fingerprint(loader, n=4):
    """Hash the first n batches' pixel content -> identifies the exact samples."""
    h = hashlib.sha256(); tasks = []
    for i, (obs, act) in enumerate(loader):
        if i >= n: break
        h.update(act.detach().cpu().numpy().tobytes())
        for k in sorted(obs.images):
            h.update(obs.images[k].detach().cpu().numpy().tobytes())
        tasks.append(list(loader.batch_task_ids[i]))
    return h.hexdigest()[:16], tasks

f1, t1 = fingerprint(val_loader)
f2, t2 = fingerprint(val_loader)
print(f"\n[P2] pass#1 fingerprint = {f1}  tasks(batch0) = {t1[0]}")
print(f"[P2] pass#2 fingerprint = {f2}  tasks(batch0) = {t2[0]}")
assert f1 == f2, "REPRODUCIBILITY FAILED: two passes saw different samples"
assert t1 == t2
print("[P2 PASS] two independent iterations scored the IDENTICAL samples")
print(f"          (batch of 8 spans {len(set(t1[0]))} distinct tasks -> stratified)")

# P3: deterministic flow metrics
# Optional: point at any ckpt to also exercise the deterministic-flow check.
# Accepts either the checkpoint DIRECTORY or the model.safetensors file itself.
ck = os.environ.get("VAL_VERIFY_CKPT", "")
if ck and os.path.isdir(ck):
    ck = os.path.join(ck, "model.safetensors")
if ck and not os.path.isfile(ck):
    print(f"\n[P3 SKIP] VAL_VERIFY_CKPT does not resolve to a safetensors file: {ck}")
    ck = ""
if torch.cuda.is_available() and os.path.isfile(ck):
    import safetensors.torch
    import openpi.models_pytorch.pi05_ki_joint_query as _ki
    mc = cfg.model; object.__setattr__(mc, "dtype", cfg.pytorch_training_precision)
    model = _ki.PI05KIJointQueryPytorch(mc)
    safetensors.torch.load_model(model, ck, strict=False)
    dev = torch.device("cuda"); model.to(dev).eval()
    def one(det):
        for obs, act in val_loader:
            obs = obs.replace(
                images={k: v.to(dev) for k, v in obs.images.items()},
                image_masks={k: v.to(dev) for k, v in obs.image_masks.items()},
                state=obs.state.to(dev),
                tokenized_prompt=obs.tokenized_prompt.to(dev) if obs.tokenized_prompt is not None else None,
                tokenized_prompt_mask=obs.tokenized_prompt_mask.to(dev) if obs.tokenized_prompt_mask is not None else None,
                token_ar_mask=obs.token_ar_mask.to(dev) if obs.token_ar_mask is not None else None,
                token_loss_mask=obs.token_loss_mask.to(dev) if obs.token_loss_mask is not None else None,
                subtask_tokens=getattr(obs,"subtask_tokens",None).to(dev) if getattr(obs,"subtask_tokens",None) is not None else None,
                subtask_mask=getattr(obs,"subtask_mask",None).to(dev) if getattr(obs,"subtask_mask",None) is not None else None,
                subtask_loss_mask=getattr(obs,"subtask_loss_mask",None).to(dev) if getattr(obs,"subtask_loss_mask",None) is not None else None,
                subtask_ar_mask=getattr(obs,"subtask_ar_mask",None).to(dev) if getattr(obs,"subtask_ar_mask",None) is not None else None)
            act = act.to(device=dev, dtype=torch.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                m = model.compute_eval_metrics(obs, act, compute_flow_l1=True,
                                               deterministic_flow=det)
            return {k: float(v.item()) for k, v in m.items()}
    a = one(True); b = one(True); c = one(False); d = one(False)
    print(f"\n[P3] deterministic_flow=True : flow_mse {a['flow_mse']:.8f} / {b['flow_mse']:.8f}"
          f"  identical={a['flow_mse']==b['flow_mse']}")
    print(f"[P3] deterministic_flow=False: flow_mse {c['flow_mse']:.8f} / {d['flow_mse']:.8f}"
          f"  identical={c['flow_mse']==d['flow_mse']}")
    print(f"[P3] flow_l1 present = {'flow_l1' in a}  value={a.get('flow_l1'):.6f}")
    if a['flow_mse'] == b['flow_mse'] and c['flow_mse'] != d['flow_mse']:
        print("[P3 PASS] fixed noise/time removes the random component; default path still stochastic")
    else:
        print("[P3 WARN] unexpected determinism pattern - inspect")
else:
    print("\n[P3 SKIP] no CUDA or ckpt missing")
print("\nALL PATCH CHECKS DONE")
