#!/usr/bin/env python3
"""Evaluate MULTIPLE checkpoints on ONE fixed stratified val subset.

Builds the dataset + index list ONCE (~140s) and reuses it for every ckpt, so
  * all ckpts are scored on byte-identical samples => PAIRED comparison valid,
  * the expensive dataset build is amortised.

Supports the openpi-comet BASELINE (plain pi05, no KI joint-query head):
  * its safetensors is a strict subset of ours, missing
      query_embeddings, query_action_head.{weight,bias}   (KI query head)
      ...language_model.embed_tokens.weight               (tied to lm_head)
    so we (a) report exactly what was missing, (b) restore embed_tokens from
    lm_head (that IS what weight tying means), and (c) mark query_*/ce/subtask
    metrics INVALID for it.
  * baseline was trained WITHOUT subtask conditioning, so `--no-subtask`
    suppresses the subtask tokens to feed it its native input format.
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
ap.add_argument("--spec", action="append", required=True,
                help="label=/path/to/ckpt_dir[:nosubtask]  (repeatable)")
ap.add_argument("--episodes-per-task", type=int, default=10)
ap.add_argument("--anchors-per-episode", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
ap.add_argument("--out", default=None, help="output JSON (default: ./val_eval_results/multi.json)")
args = ap.parse_args()
OUT_PATH = args.out or "./val_eval_results/multi.json"

import behavior.learning.datas.dataset as _bds
_o = _bds.BehaviorLeRobotDataset.__init__
def _p(self, *a, **kw):
    kw["chunk_streaming_using_keyframe"] = False   # idx honored
    kw["shuffle"] = False
    return _o(self, *a, **kw)
_bds.BehaviorLeRobotDataset.__init__ = _p

import torch, safetensors.torch
import openpi.training.config as _config
import openpi.training.data_loader as _dl
import openpi.models.model as _model
import openpi.models_pytorch.pi05_ki_joint_query as _ki

cfg = _config.get_config("pi05_ki_joint_query_b1k-full_task-ki_on_bf16")
dc = cfg.val_data[0].create(cfg.assets_dirs, cfg.model)
t0 = time.time()
raw = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
print(f"[build] {time.time()-t0:.1f}s len={len(raw)}", flush=True)
inner = getattr(raw, "_dataset", raw)
eps = list(getattr(inner, "episodes", []) or [])
edi = getattr(inner, "episode_data_index", None)
fr, to = edi["from"], edi["to"]
bounds = {int(e): (int(fr[i]), int(to[i])) for i, e in enumerate(eps)}
by_task = {}
for e, (a, b) in bounds.items(): by_task.setdefault(e // 10000, []).append((e, a, b))
rng = random.Random(args.seed)
pairs = []
for tk in sorted(by_task):
    cand = sorted(by_task[tk], key=lambda x: x[0])
    if len(cand) > args.episodes_per_task:
        cand = sorted(rng.sample(cand, args.episodes_per_task), key=lambda x: x[0])
    for e, a, b in cand:
        if b <= a: continue
        for j in range(args.anchors_per_episode):
            f = (j + 0.5) / args.anchors_per_episode
            pairs.append((min(b - 1, a + int((b - a) * f)), tk, e))
random.Random(args.seed + 1).shuffle(pairs)
print(f"[subset] anchors={len(pairs)} tasks={len(set(p[1] for p in pairs))} "
      f"episodes={len(set(p[2] for p in pairs))}", flush=True)

ds = _dl.transform_dataset(raw, dc, skip_norm_stats=False)
class Sub(torch.utils.data.Dataset):
    def __init__(s, b, pr): s.b, s.pr = b, pr
    def __len__(s): return len(s.pr)
    def __getitem__(s, i): return s.b[s.pr[i][0]]
loader = torch.utils.data.DataLoader(
    Sub(ds, pairs), batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, collate_fn=_dl._collate_fn, drop_last=True,
    **({"prefetch_factor": 2} if args.num_workers > 0 else {}))
NB = args.max_batches or (len(pairs) // args.batch_size)

def as_t(o):
    import numpy as _np
    if o is None: return None
    if isinstance(o, dict): return {k: as_t(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return type(o)(as_t(v) for v in o)
    if isinstance(o, _np.ndarray): return torch.as_tensor(o)
    return o

# Cache the fixed batches ONCE so every ckpt truly sees identical tensors.
print("[cache] materialising fixed batches ...", flush=True)
t0 = time.time(); cached = []
for i, b in enumerate(loader):
    if i >= NB: break
    b = as_t(b)
    tks = [pairs[i*args.batch_size+j][1] for j in range(args.batch_size)
           if i*args.batch_size+j < len(pairs)]
    cached.append((b, tks))
print(f"[cache] {len(cached)} batches in {time.time()-t0:.1f}s", flush=True)

dev = torch.device("cuda")
adt = torch.bfloat16 if cfg.pytorch_training_precision == "bfloat16" else torch.float16
def _to(x): return None if x is None else x.to(dev, non_blocking=True)

def load_ckpt(path):
    mc = cfg.model; object.__setattr__(mc, "dtype", cfg.pytorch_training_precision)
    m = _ki.PI05KIJointQueryPytorch(mc)
    f = os.path.join(path, "model.safetensors")
    missing, unexpected = safetensors.torch.load_model(m, f, strict=False)
    info = {"missing": list(missing), "unexpected": list(unexpected)}
    # Weight tying: baseline omits embed_tokens.weight because it is shared with
    # lm_head.weight. Restore it explicitly, else the language embedding would be
    # left at random init and EVERY metric would be meaningless.
    fixed = []
    for name in list(missing):
        if name.endswith("embed_tokens.weight"):
            lm = dict(m.named_parameters()).get("paligemma_with_expert.paligemma.lm_head.weight")
            tgt = dict(m.named_parameters()).get(name)
            if lm is not None and tgt is not None and lm.shape == tgt.shape:
                with torch.no_grad(): tgt.copy_(lm)
                fixed.append(f"{name} <- lm_head.weight (tied)")
    info["tie_fixed"] = fixed
    m.to(dev).eval()
    return m, info

def run(model, no_subtask):
    pb, ptask = [], {}
    t0 = time.time()
    with torch.no_grad():
        for i, (b, tks) in enumerate(cached):
            obs = _model.Observation.from_dict(b); act = b["actions"]
            sm = getattr(obs, "subtask_mask", None)
            slm = getattr(obs, "subtask_loss_mask", None)
            if no_subtask:
                # The backbone and expert branches gate on subtask presence via
                # DIFFERENT masks:
                #   _compute_backbone_eval_metrics -> subtask_loss_mask.any()
                #   compute_expert_loss            -> subtask_mask.any()
                # Zeroing only one makes the paths disagree and the CE slice
                # length stops matching the targets. Zero BOTH so the model runs
                # in its native no-subtask configuration.
                if sm is not None:
                    sm = torch.zeros_like(sm)
                if slm is not None:
                    slm = torch.zeros_like(slm)
            obs = obs.replace(
                images={k: v.to(dev, non_blocking=True) for k, v in obs.images.items()},
                image_masks={k: v.to(dev, non_blocking=True) for k, v in obs.image_masks.items()},
                state=obs.state.to(dev, non_blocking=True),
                tokenized_prompt=_to(obs.tokenized_prompt),
                tokenized_prompt_mask=_to(obs.tokenized_prompt_mask),
                token_ar_mask=_to(obs.token_ar_mask),
                token_loss_mask=_to(obs.token_loss_mask),
                subtask_tokens=_to(getattr(obs, "subtask_tokens", None)),
                subtask_mask=_to(sm),
                subtask_loss_mask=_to(slm),
                subtask_ar_mask=_to(getattr(obs, "subtask_ar_mask", None)),
                pcd_xyz=_to(getattr(obs, "pcd_xyz", None)))
            act = act.to(device=dev, dtype=torch.float32, non_blocking=True)
            with torch.autocast("cuda", dtype=adt, enabled=True):
                m = model.compute_eval_metrics(obs, act, compute_flow_l1=True,
                                               num_denoise_steps=10, flow_l1_seed=42+9999,
                                               deterministic_flow=True)
            d = {k: float(v.detach().float().item()) for k, v in m.items()
                 if hasattr(v, "numel") and v.numel() == 1}
            pb.append(d)
            for tk in set(tks): ptask.setdefault(int(tk), []).append(d)
            if (i+1) % 10 == 0:
                print(f"    batch {i+1}/{len(cached)} cum={time.time()-t0:.0f}s", flush=True)
    return pb, ptask

out = {"subset": {"anchors": len(pairs), "tasks": len(set(p[1] for p in pairs)),
                  "episodes": len(set(p[2] for p in pairs)),
                  "batches": len(cached), "batch_size": args.batch_size,
                  "seed": args.seed}, "runs": {}}
for spec in args.spec:
    label, path = spec.split("=", 1)
    no_sub = path.endswith(":nosubtask")
    if no_sub: path = path[: -len(":nosubtask")]
    print(f"\n===== {label}  no_subtask={no_sub}\n      {path}", flush=True)
    model, info = load_ckpt(path)
    print(f"      missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    for x in info["missing"][:8]: print(f"        MISSING {x}")
    for x in info["tie_fixed"]: print(f"        FIXED   {x}")
    pb, ptask = run(model, no_sub)
    mets = sorted({k for d in pb for k in d})
    summ = {}
    for k in mets:
        v = [d[k] for d in pb if k in d]
        if len(v) < 2: continue
        mu = st.mean(v); sd = st.pstdev(v)
        summ[k] = {"mean": mu, "cv_pct": (sd/abs(mu)*100 if mu else None),
                   "sem_pct": (sd/abs(mu)*100/math.sqrt(len(v)) if mu else None),
                   "n": len(v)}
    out["runs"][label] = {
        "ckpt": path, "no_subtask": no_sub,
        "missing_keys": info["missing"], "tie_fixed": info["tie_fixed"],
        "has_query_head": not any("query_action_head" in x for x in info["missing"]),
        "metrics": summ, "per_batch": pb,
        "per_task_flow_l1": {str(t): st.mean([d["flow_l1"] for d in ds_ if "flow_l1" in ds_[0]])
                             for t, ds_ in ptask.items() if ds_ and "flow_l1" in ds_[0]},
    }
    del model; torch.cuda.empty_cache()
    # Incremental dump: a later ckpt failing must not lose earlier results.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"      done: flow_l1={summ.get('flow_l1',{}).get('mean')} "
          f"flow_mse={summ.get('flow_mse',{}).get('mean')}  [saved]", flush=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
json.dump(out, open(OUT_PATH, "w"), indent=1)
print(f"\nwrote {OUT_PATH}")
