#!/usr/bin/env python3
"""Definitive: evaluate a ckpt on a FIXED STRATIFIED val subset with streaming OFF,
to obtain the TRUE independent-sample variance and thus size the val subset (K).

Why this is needed: all previously measured CVs came from a stateful SEQUENTIAL
stream (adjacent frames are near-duplicates), so they understate the variance of
genuinely independent samples and overstate the effective sample size.
Here every sample is an independent anchor from a distinct (task, episode).
"""
from __future__ import annotations
import os, pathlib, sys, time, json, math, random, statistics as st, argparse
import pathlib

# Repo root is derived from this file's location (scripts/val_eval/<x>.py),
# so the script works from any checkout. Override with OPENPI_REPO_ROOT.
REPO = os.environ.get(
    "OPENPI_REPO_ROOT",
    str(pathlib.Path(__file__).resolve().parents[2]),
)
sys.path.insert(0, os.path.join(REPO, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--episodes-per-task", type=int, default=8)
ap.add_argument("--anchors-per-episode", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--flow-l1", action="store_true", help="also compute Euler-integrated flow_l1")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import behavior.learning.datas.dataset as _bds
_orig = _bds.BehaviorLeRobotDataset.__init__
def _patched(self, *a, **kw):
    kw["chunk_streaming_using_keyframe"] = False   # idx honored
    kw["shuffle"] = False
    return _orig(self, *a, **kw)
_bds.BehaviorLeRobotDataset.__init__ = _patched

import torch, safetensors.torch
import openpi.training.config as _config
import openpi.training.data_loader as _dl
import openpi.models_pytorch.pi05_ki_joint_query as _ki

cfg = _config.get_config("pi05_ki_joint_query_b1k-full_task-ki_on_bf16")
dc = cfg.val_data[0].create(cfg.assets_dirs, cfg.model)
t0=time.time(); ds_raw = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
print(f"[build] {time.time()-t0:.1f}s  len={len(ds_raw)}", flush=True)
inner = getattr(ds_raw, "_dataset", ds_raw)
eps = list(getattr(inner, "episodes", []) or [])
edi = getattr(inner, "episode_data_index", None)
fr, to = edi["from"], edi["to"]
bounds = {ep: (int(fr[i]), int(to[i])) for i, ep in enumerate(eps)}

by_task = {}
for ep,(a,b) in bounds.items(): by_task.setdefault(ep//10000, []).append((ep,a,b))
rng = random.Random(args.seed)
pairs = []   # (idx, task, ep)
for task in sorted(by_task):
    ch = sorted(by_task[task], key=lambda x:x[0])
    if len(ch) > args.episodes_per_task: ch = rng.sample(ch, args.episodes_per_task)
    for ep,a,b in ch:
        if b<=a: continue
        for j in range(args.anchors_per_episode):
            frac=(j+0.5)/args.anchors_per_episode
            pairs.append((min(b-1, a+int((b-a)*frac)), task, ep))
rng.shuffle(pairs)     # mix tasks across batches
print(f"[subset] anchors={len(pairs)} tasks={len(set(p[1] for p in pairs))} "
      f"episodes={len(set(p[2] for p in pairs))}", flush=True)

ds = _dl.transform_dataset(ds_raw, dc, skip_norm_stats=False)
class Sub(torch.utils.data.Dataset):
    def __init__(s, base, pr): s.base, s.pr = base, pr
    def __len__(s): return len(s.pr)
    def __getitem__(s, i): return s.base[s.pr[i][0]]
sub = Sub(ds, pairs)

mc = cfg.model; object.__setattr__(mc, "dtype", cfg.pytorch_training_precision)
model = _ki.PI05KIJointQueryPytorch(mc)
safetensors.torch.load_model(model, os.path.join(args.ckpt,"model.safetensors"), strict=False)
dev = torch.device("cuda"); model.to(dev).eval()
adt = torch.bfloat16 if cfg.pytorch_training_precision=="bfloat16" else torch.float16
print(f"[model] loaded {args.ckpt}", flush=True)

dl = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False,
      num_workers=args.num_workers, collate_fn=_dl._collate_fn, drop_last=True,
      **({"prefetch_factor":2} if args.num_workers>0 else {}))
import openpi.models.model as _model
def _to(x,d): return None if x is None else x.to(d, non_blocking=True)

def _as_tensors(o):
    """raw _collate_fn yields numpy; TorchDataLoader normally wraps with as_tensor."""
    import numpy as _np
    if o is None: return None
    if isinstance(o, dict): return {k:_as_tensors(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return type(o)(_as_tensors(v) for v in o)
    if isinstance(o, _np.ndarray): return torch.as_tensor(o)
    return o

per_batch=[]; per_task={}
t0=time.time()
with torch.no_grad():
    for i,b in enumerate(dl):
        b = _as_tensors(b)
        obs = _model.Observation.from_dict(b); act = b["actions"]
        obs = obs.replace(
            images={k:v.to(dev,non_blocking=True) for k,v in obs.images.items()},
            image_masks={k:v.to(dev,non_blocking=True) for k,v in obs.image_masks.items()},
            state=obs.state.to(dev,non_blocking=True),
            tokenized_prompt=_to(obs.tokenized_prompt,dev),
            tokenized_prompt_mask=_to(obs.tokenized_prompt_mask,dev),
            token_ar_mask=_to(obs.token_ar_mask,dev),
            token_loss_mask=_to(obs.token_loss_mask,dev),
            subtask_tokens=_to(getattr(obs,"subtask_tokens",None),dev),
            subtask_mask=_to(getattr(obs,"subtask_mask",None),dev),
            subtask_loss_mask=_to(getattr(obs,"subtask_loss_mask",None),dev),
            subtask_ar_mask=_to(getattr(obs,"subtask_ar_mask",None),dev),
            pcd_xyz=_to(getattr(obs,"pcd_xyz",None),dev))
        act = act.to(device=dev, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=adt, enabled=True):
            m = model.compute_eval_metrics(obs, act, compute_flow_l1=args.flow_l1,
                                           num_denoise_steps=10, flow_l1_seed=42+9999)
        d = {k: float(v.detach().float().item()) for k,v in m.items()
             if hasattr(v,"numel") and v.numel()==1}
        per_batch.append(d)
        tks = [pairs[i*args.batch_size+j][1] for j in range(args.batch_size)
               if i*args.batch_size+j < len(pairs)]
        for tk in set(tks): per_task.setdefault(tk, []).append(d)
        if (i+1)%10==0:
            print(f"  batch {i+1} cum={time.time()-t0:.1f}s total_loss={d.get('total_loss'):.5f}", flush=True)
tot=time.time()-t0
print(f"[eval] {len(per_batch)} batches in {tot:.1f}s = {tot/max(1,len(per_batch)*args.batch_size):.3f} s/sample\n")

MET=["total_loss","backbone_loss","ce_loss","query_mse_loss","subtask_accuracy",
     "expert_loss","flow_mse","query_l1"]+(["flow_l1"] if args.flow_l1 else [])
print("TRUE INDEPENDENT-SAMPLE VARIANCE (each batch = 8 independent stratified anchors)")
print(f"{'metric':18s} {'mean':>10s} {'CV%':>8s} {'SEM%@this n':>12s} | K for SEM<3% | K for SEM<1%")
print("-"*92)
summ={}
B=len(per_batch); n_eff=B*args.batch_size
for k in MET:
    v=[d[k] for d in per_batch if k in d]
    if len(v)<3: continue
    mu=st.mean(v); cv=st.pstdev(v)/abs(mu)*100
    sem=cv/math.sqrt(len(v))
    k3=math.ceil((cv/3.0)**2)*args.batch_size
    k1=math.ceil((cv/1.0)**2)*args.batch_size
    summ[k]={"mean":mu,"cv_pct":cv,"sem_pct":sem,"n_batches":len(v),
             "n_samples":len(v)*args.batch_size,"K_sem3":k3,"K_sem1":k1}
    print(f"{k:18s} {mu:10.6f} {cv:7.1f}% {sem:11.2f}% | {k3:>12,} | {k1:>12,}")
print("-"*92)
print(f"n = {B} batches x {args.batch_size} = {n_eff} INDEPENDENT samples")

print("\nPER-TASK total_loss (top 8 hardest / 3 easiest)")
tl={t:st.mean([d['total_loss'] for d in ds_]) for t,ds_ in per_task.items() if ds_}
sr=sorted(tl.items(), key=lambda x:-x[1])
for t,v in sr[:8]: print(f"  task-{int(t):04d}  {v:.5f}")
print("  ...")
for t,v in sr[-3:]: print(f"  task-{int(t):04d}  {v:.5f}")
if tl:
    vals=list(tl.values())
    print(f"  => across {len(tl)} tasks: min={min(vals):.5f} max={max(vals):.5f} "
          f"spread={max(vals)/min(vals):.1f}x  CV={st.pstdev(vals)/st.mean(vals)*100:.1f}%")

out = args.out or f"./val_eval_results/cv_{args.label}.json"
os.makedirs(os.path.dirname(out),exist_ok=True)
json.dump({"label":args.label,"ckpt":args.ckpt,"streaming":False,
           "anchors":len(pairs),"tasks":len(set(p[1] for p in pairs)),
           "episodes":len(set(p[2] for p in pairs)),
           "batch_size":args.batch_size,"num_workers":args.num_workers,
           "eval_s":tot,"s_per_sample":tot/max(1,len(per_batch)*args.batch_size),
           "metrics":summ,"per_batch":per_batch,
           "per_task_total_loss":{str(int(k)):v for k,v in tl.items()}},
          open(out,"w"), indent=1)
print(f"\nwrote {out}")
