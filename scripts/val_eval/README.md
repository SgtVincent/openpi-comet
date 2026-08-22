# Offline validation tools

Tooling that was used to diagnose and verify the deterministic, task-stratified
validation subset (see `TrainConfig.val_deterministic_subset` and friends).

Everything here is **offline / analysis-only** — none of it is imported by the
training loop. The scripts deliberately reuse the production data path
(`create_torch_dataset` → `transform_dataset` → `compute_eval_metrics`) so that
preprocessing, normalization and metric definitions are identical to
training-time validation; only the *sampling* differs.

## Why this exists

The training-time validation loader used to be built with
`chunk_streaming_using_keyframe=True`, where `__getitem__(idx)` ignores `idx`
and returns the frame at an internal, monotonically advancing cursor. On the
BEHAVIOR-1K 50-task split that meant:

* sampler permutations and `shuffle` were no-ops,
* a batch of 8 was 8 *consecutive* frames (~0.27 s) of one episode, so the
  effective sample size was a small fraction of the nominal one,
* with `persistent_workers=True` the cursor survived across `iter()` calls, so
  each validation continued the stream and drifted forward,
* only ~1 of 50 tasks was ever scored.

Consequence: val curves mixed "model improved" with "data changed" and could
not be used for checkpoint selection. The fix builds one fixed, stratified
index list (episodes from every task, anchors spread across each episode's
phases) with streaming disabled so `idx` is honored.

This is **on by default** (`TrainConfig.val_deterministic_subset`,
`val_deterministic_flow`, `val_slow_metrics_every`, `val_log_per_task`), so
individual train configs do not need to opt in. If a val dataset does not
expose per-episode index bounds, the trainer logs a warning and falls back to
the legacy streaming loader. Set `val_deterministic_subset=False` only to
reproduce a historical run's exact val numbers.

## Scripts

| Script | Purpose |
|---|---|
| `verify_val_subset.py` | Regression check for the val patch. Calls the real `build_val_datasets()` and asserts (P1) the subset covers all tasks, (P2) two independent iterations score byte-identical samples, (P3) `deterministic_flow=True` gives bit-identical flow metrics while the default path stays stochastic. Set `VAL_VERIFY_CKPT=/path/to/ckpt_dir` to enable P3. |
| `eval_multi_ckpt.py` | Evaluate several checkpoints on ONE fixed stratified subset. Materializes the batches once and reuses them, so every checkpoint sees byte-identical tensors and a **paired** comparison is valid. |
| `bench_val_variance.py` | Measure true independent-sample variance on the stratified subset and derive the subset size `K` needed for a target SEM, plus a per-task breakdown. |
| `run_val_eval.sh` | Env wrapper reproducing the training-time environment (conda env, offline HF, node-local caches, baseline stride contract). |

## Usage

```bash
export CONDA_ROOT=/path/to/miniconda3
export B1K_DATASET_ROOT=/path/to/2025-challenge-demos/

# 1. verify the val subset is fixed and representative
CUDA_VISIBLE_DEVICES=0 scripts/val_eval/run_val_eval.sh \
    scripts/val_eval/verify_val_subset.py

# 2. compare checkpoints on the same fixed samples (paired)
CUDA_VISIBLE_DEVICES=0 scripts/val_eval/run_val_eval.sh \
    scripts/val_eval/eval_multi_ckpt.py \
    --spec "old=/path/to/ckpt_a" --spec "new=/path/to/ckpt_b"

# 3. size the subset for a target precision
CUDA_VISIBLE_DEVICES=0 scripts/val_eval/run_val_eval.sh \
    scripts/val_eval/bench_val_variance.py \
    --ckpt /path/to/ckpt --label mymodel --flow-l1
```

Results are written under `./val_eval_results/` by default.

## Gotchas worth knowing

* **`num_workers=0` when streaming is ON.** `BehaviorLeRobotDataset` already
  forks its own helper processes; stacking DataLoader workers on top caused a
  single batch to take >10 min. With streaming OFF (what these tools use),
  `num_workers>0` is fine and recommended.
* **Random access is ~10x more expensive per sample than sequential** (video
  seek), but it parallelizes, and the stratified subset needs far fewer samples
  for the same precision — so the net cost is lower.
* **`transform_dataset` drops `episode_index` / `timestamp`.** Build any index
  list from `episode_data_index` on the *raw* dataset, before wrapping.
* **The raw `_collate_fn` returns numpy**, not tensors (`TorchDataLoader`
  normally converts in `__iter__`). Convert before `Observation.from_dict`.
* **`episodes_index` is applied PER TASK**, not globally.
* **Evaluating a foreign checkpoint**: use its own model class and its own
  `assets/norm_stats.json`, and do not route it through a subclass'
  `sample_actions` — subclasses may autoregressively predict conditioning
  tokens using weights the foreign checkpoint does not have.
* **`deterministic_flow=True` values are not comparable to the default path**
  (clean vs augmented images, fixed vs random flow-matching timestep). Compare
  within one mode only.
