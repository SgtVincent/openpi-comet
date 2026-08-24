# H20 Variant A (FAST-CE) two-arm experiment — record and provenance

Everything needed to reconstruct, audit or repeat this experiment. Written for a
reader with **no access to the conversation that produced it**. It records the
failures as well as the results, because most of the cost here was in defects that
would have been invisible until hours into a 32-GPU run.

Code commit: see `git log` for `scripts/h20_train_entrypoint.sh`. Base of this
work: `2ac7cab20c6b93d6a2c02503025bd461c0a92469`.

---

## 1. The question, and the limit on what it can answer

π0.5-KI **Variant A** — FAST discrete action tokens supervised by cross-entropy on
the backbone — exists in this repo but had never been run. This experiment runs it
on 4×8 NVIDIA_H20, warm-started from an **already B1K-fine-tuned** checkpoint
rather than the π0.5 pretrain, as a two-arm comparison of the two base packages.

| | Arm A "comet base" | Arm B "pi05 base" |
|---|---|---|
| config | `pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16` | `..._on_h20_pi05base_bf16` |
| weights | `checkpoints/openpi_comet/pi05-b1kpt50-cs32` | `checkpoints/pi05_base_pytorch` |
| stored dtype | fp32, `model.safetensors` 14,467,165,872 B | bf16, 7,233,650,408 B |
| `norm_stats.json` | `d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929` (6368 B) | `4dde119e69123ed865072c71a714095ae746c6d294fefba910a842757a7083ce` (6361 B) |

Everything else is held identical: model, objective, FAST capacity, data root,
schedule, B8/GA1 → global batch 256, stride-12 three-pass rotation with offsets
(0, 4, 8), and the fixed 104,912-step budget.

### ⚠️ Scoping limit — the arms differ in TWO coupled variables

Each arm pairs a weight set with the `norm_stats` that **ships alongside it**,
because that arm's flow expert and action head were fit in that normalization
space. That pairing is deliberate and correct, and it means base weights *and*
action normalization change together.

**Therefore this experiment answers "which base *package* is the better starting
point for π0.5-KI". It does NOT answer "which weights are better with
normalization held fixed".** Any write-up that implies the latter is wrong. There
is also a minor stored-dtype asymmetry (fp32 vs bf16, both downcast to bf16 for
training).

### Budget asymmetry — do not present convergence as budget-neutral

104,912 steps at cont2's measured 3.474 s/step ≈ **4.22 days × 32 H20 per arm**;
sequential ≈ 8.4 days. Group 947 holds only ~36 H20, so the arms most likely
cannot run concurrently.

A single fixed budget is what makes the arms comparable and is a deliberate
choice — but it is **not neutral between them**. Arm A begins from weights already
fine-tuned on B1K and so starts closer to the target by construction. "Arm A
converged faster" or "reached lower loss sooner" must never be reported as if the
budget were neutral.

---

## 2. Why Arm A's normalization was chosen, and the corroboration

The two `norm_stats.json` files differ by empirical estimate, not algorithm:
identical schema, 32 dims, 22 non-degenerate in both, several dims matching to 5
decimals at hard joint limits (`2.09440` = 2π/3, `3.14160` = π). Arm A's was
shipped by Team Comet; ours was recomputed locally.

Arm A uses `d66ed168` because we continue from those weights and the flow expert
was fit in that space. Feeding a different normalization partly discards the warm
start.

**Independent corroboration:** the verified RGBWrapper easy-8 baseline (mean
q = 0.347044, q>0 in 12/24, full success 5/24) **itself ran under `d66ed168`** —
all 10 lanes logged `Loaded norm stats from .../pi05-b1kpt50-cs32/assets/behavior-1k/2025-challenge-demos`,
and all 24 episodes served `pi05-b1kpt50-cs32`. `pi05-b1kpt12-cs32` was served by
zero lanes.

Quantified effect of the change (same raw action through `[q01,q99] → [-1,1]`, no
clipping): 18 of 22 non-degenerate dims move < 0.02, span-ratio median exactly
1.0000, and only dim0 (gain 0.70, *dampens*) and dim16 (gain 1.0609, the sole
amplifier) exceed 0.10.

---

## 3. FAST action-token cap = 256 — exhaustively verified

`action_token_max_len` is **fail-closed**: `tokenize_action_chunk` in
`src/openpi/models/tokenizer.py` **raises** rather than truncating, because FAST is
byte-pair encoded over DCT coefficients so dropping trailing tokens corrupts the
chunk rather than shortening it. One over-cap window anywhere aborts the run.

Exhaustive scan under `d66ed168`, every window in both populations:

| split | n windows | min | p50 | p99 | p99.9 | **max** | >199 | >208 | >256 |
|---|---|---|---|---|---|---|---|---|---|
| train | 26,857,712 | 19 | 37 | 73 | 98 | **200** | 1 | 0 | 0 |
| val | 11,398,271 | 19 | 38 | 74 | 102 | **189** | 0 | 0 | 0 |

Record: `h20_fastce_scan/exhaustive_W_cap256_20260824_124043/` — per-split JSON
with full histograms, `SAMPLED_NOT_EXHAUSTIVE: false`, `windows_scanned ==
population_windows`, `anchor_rule {stride 12, offsets [0,4,8], drop true}` (the
formal training contract, so the scanned population **is** what the run consumes),
`action_dim: 23`, and the full norm_stats digest.

### The decisive fact, and an honest note on the value

**Train max under `d66ed168` is 200, which EXCEEDS the 199 max of the `4dde119e`
population — on exactly 1 window in 26,857,712 (3.7e-08).** So reusing the
historical cap-208 provenance (`run_id=0bb9280746…`, `manifest=ef4cb52d…`,
`aggregate=51250f15…`) across normalizations would have asserted a max of 199 for a
population whose max is 200 — *literally* false, not merely unproven. That
provenance belongs to `4dde119e` and is deliberately not reused.

**Honest note: 208 would in fact have sufficed** (200 < 208, 8 tokens spare). 256
was chosen before the exhaustive scan existed, from a sampled bound, and is more
conservative than the result requires — 56 tokens of headroom where 8 would have
done, at ~+3% sequence length. It is kept because it is exhaustively proven safe,
and it is **not** presented as vindicated.

For anyone setting a future cap: the sampled maxima were train 166 / val 178, i.e.
they **understated** the exhaustive maxima by 34 and 11 tokens. A sampled max
cannot bound a fail-closed cap.

### Both arms share one cap, and that is metric-neutral

Capacity is held identical so it cannot become a third confound. Safe because
padded positions are emitted with `mask=False`, `ar_mask=0`, `loss_mask=False`, and
the action objective divides by `shift_loss_mask.sum()` — CE and accuracy are
normalized over valid tokens only. Arm B's population is exactly what cap 208 was
proven on, so 256 is strictly slack there.

Re-audit: `h20_fastce_scan/run_exhaustive_gate.sh <P|W|/abs/path> <cap> <workers>`
(measured ~6,958 windows/s at 48 workers; both splits ≈ 95 min).

---

## 4. Warm start — verified, both arms identical

`scripts/train_accelerate.py` loads `pi05_ki_joint_fast` with **`strict=False`**, so
a warm start that matched nothing would still log success and train from noise.
Verified explicitly with `scripts/verify_warm_start_keymap.py --actually-load`:

```
matched 812 | tied 1 | missing 0 | unexpected 0 | shape_mismatch 0
coverage 4,143,404,816 / 4,143,404,816 = 100.0000%
real loader returned missing=[] unexpected=[]
```

The one initially-absent key is `paligemma.model.language_model.embed_tokens.weight`
(257152 × 2048 = 526,647,296 params — exactly the model/checkpoint gap). It is
**documented weight tying**, recorded in each file's safetensors `__metadata__` as
an alias of `paligemma.lm_head.weight`, and proven repopulated: abs-sum
8,403,631 → 66,391,008, then exactly equal to the tie target. This matters because
Variant A's CE objective runs on that vocab tail.

Variant A adds **no** parameters: versus Variant B it simply lacks
`query_action_head.{weight,bias}` and `query_embeddings`.

### "Suspiciously similar" is expected here — do not re-raise it as a defect

Both arms' post-load embedding abs-sums are nearly identical (66,391,008 vs
66,390,988). That is **not** a failed load. Direct checkpoint comparison:

| tensor | relative difference |
|---|---|
| `paligemma.lm_head.weight` (= the tied vocab embedding) | **0.0032%** |
| `action_in_proj.weight` | 2.17% |
| `time_mlp_in.bias` | 12.43% |
| `action_out_proj.weight` | 15.32% |
| `gemma_expert.lm_head.weight` | 141.42% |
| **mean over probe tensors** | **20.68%** |

B1K fine-tuning barely touched the vocab embedding, so near-identical
token-embedding statistics between arms are expected. The action head, expert and
time-MLP genuinely differ, so flow and action losses should diverge.

---

## 5. Defects found and fixed — the expensive part

### 5.1 `deterministic_flow` — would have killed both arms at their first validation

`train_accelerate.py` routes both KI variants through one `is_pi05_ki_joint` branch
and passes `deterministic_flow` to `compute_eval_metrics` unconditionally.
`PI05KIJointQueryPytorch` accepted it; `PI05KIJointFastPytorch`'s **override did
not**. Result: `TypeError: compute_eval_metrics() got an unexpected keyword
argument 'deterministic_flow'` at the first validation.

* **Root cause, inherited from `2ac7cab`:** `git log -S` shows the kwarg entered the
  call site *and* Variant B's signature in the same commit, `1b75e55`
  ("fix(validation): consolidate deterministic distributed evaluation"), and never
  appeared in `pi05_ki_joint_fast.py`. A caller and one of two implementations were
  updated; the other was missed.
* **`val_deterministic_flow` defaults to `True`**, so this was not a dormant branch
  that *might* fire — it **would** have fired on both arms.
* Historical cost: it killed an A100 FAST run at its first validation after
  ~2h40m of training.

**Fixed with semantics, not just a signature.** `deterministic_flow` is a
*behaviour flag*, not a metric selector: with it off, `flow_loss` / `expert_loss` /
`total_loss` carry a random component that does **not** shrink as the validation
subset grows. An arm that merely accepted the kwarg would have an irreducible noise
floor on exactly the metrics this A/B compares — worse than the crash, because it
costs the conclusion rather than the run. Variant A now honours both halves: fixed
`(noise, time)` drawn from `flow_l1_seed`, and `train_preprocess=False`.

**Porting trap, copied deliberately:** `torch.manual_seed()` reseeds the CPU
generator **and all CUDA generators**, while `torch.get/set_rng_state()` covers only
the CPU one. Restoring only CPU state leaks a CUDA reseed into the *training* RNG
stream — converting a validation bug into a training bug. Both CPU and CUDA states
are saved and restored.

### 5.2 The shared call surface, audited in three buckets

`is_pi05_ki_joint` treats the two classes as interchangeable while they are
maintained separately. **Generalisation: the divergence risk is exactly the set of
methods FAST overrides** — which bounds the audit.

* **Behaviour / determinism flags** (must be honoured or hard-fail):
  `deterministic_flow` — was the defect; now honoured.
* **Metric selectors** (a variant may legitimately not emit one, but consumers must
  not index unconditionally): `compute_flow_l1`. FAST omits `query_mse_loss` /
  `query_l1` and emits `action_ce_loss` / `action_token_accuracy` instead. All
  consumers already guard (`"…" in extra_metrics`, `global_means.get("query_l1",
  nan)`), so **`query_l1=nan` in the `[VAL]` line is expected for Variant A**, not a
  failure.
* **Metric parameters** (travel with their metric): `num_denoise_steps`,
  `flow_l1_seed`.

Of 5 overridden methods, 3 signatures diverge: two are deliberate
`NotImplementedError` stubs for Variant-B-only paths (`_embed_query_tokens`,
`_compute_query_mse_loss`), and `compute_eval_metrics` was the only real defect.

The **second** RNG save/restore site, `_compute_flow_l1_euler`, needed no change:
`Fast._compute_flow_l1_euler is Query._compute_flow_l1_euler` → `True`. FAST
inherits it rather than re-implementing it, which is precisely why it never
diverged. A test asserts that sharing, so a future re-implementation without the
RNG handling fails loudly.

### 5.3 The smoke could not have caught it — `val_log_interval` 8 → 4

Validation fires on `global_step % val_log_interval == 0 and global_step > 0`. With
an 8-step budget and interval 8, the only validation landed **exactly on the
termination boundary**, so the smoke never called `compute_eval_metrics` at all. It
would have passed cleanly and let a broken run be promoted to the 104,912-step
budget.

Now 4, so an 8-step smoke validates at steps **4 and 8**. The launcher preflight
asserts `val_log_interval < num_train_steps` with ≥2 passes for smoke configs.
**A smoke that cannot reach the code path with a known failure mode is not a gate.**

### 5.4 Entrypoint defects — and the subsystem that caused most of them

The first entrypoint had three defects. Only the first was inherent; the other two
were created by a design choice that has since been removed.

1. **`set -uo pipefail` released all 32 GPUs.** After `-u`, sourcing
   `extra_bashrc.sh` aborts at its line 33 (`PROMPT_COMMAND` is an
   interactive-shell variable, unset in a job pod) **before** `use_gpu`/`free_gpu`
   are defined; the entrypoint exits, the trial ends, 32 H20 are released — and it
   masquerades as a platform fault. This defect is real independently of any design
   choice. Fixed: no `-u`, no `-e`, plus a `PROMPT_COMMAND` guard, all
   test-enforced.
2. **A 60 s occupancy supervisor raced the launcher.** It restored occupiers during
   the launcher's CPU preflight, and `assert_no_occupiers` then aborted the run.
   Measured: the warm-start keymap check alone takes **261 s** cold and the full
   preflight **>450 s** — many cycles, so this was certain, not a race. (The cost is
   *not* reading the 14.5 GB checkpoint — only its header is parsed — but importing
   openpi/transformers off NFS and building the 4.1B-param module graph.)
3. **The supervisor became a single point of failure**, since it was the only thing
   restoring occupancy after training ended; if it died the allocation survived but
   utilization sat at zero and the 30%/3-hour reaper would take all 32 cards.

**Resolution: the supervisor was deleted, and defects 2 and 3 went with it.** They
existed only to manage it — the race required a launch sentinel, the sentinel
required three-way phase logic, and the single-point-of-failure required a
watchdog. Three of the four fixes were self-inflicted complexity.

The actual requirement — *if training fails, fall back to the matmul occupiers so
the cards are not released* — was already met by
`scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh`, which with
`KEEPALIVE_ON_SUCCESS=1` holds the allocation after success or failure, restarts
dead occupiers (`OCCUPIER_AUTO_RESTART=1`) and never returns. The entrypoint now
delegates to it and follows the house pattern proven by job `6fc66189eb6c5c88`:
set env, then call a repo script.

The **one** genuine difference from a clean-node start is that this job is converted
from a hold job with 32 tagged occupiers already running, and the launcher
hard-fails via `assert_no_occupiers`. That needs a single `free_gpu` before
launching — two lines, not a subsystem.

`tests/scripts/test_h20_train_entrypoint.sh` asserts both that the surviving
properties hold and that the removed machinery (`supervisor()`, `LAUNCHING`,
`kill -0`, `occ_count()`) **stays** removed, including a line-count ceiling, because
the failure mode here is complexity creeping back.

Batch/stride/offset values are deliberately **not** re-exported by the entrypoint:
the launcher defaults and asserts the batch contract, and `train_accelerate.py`
`setdefault`s `FRAME_ANCHOR_STRIDE=12` / `FRAME_ANCHOR_OFFSETS=0,4,8` and then
validates them. Duplicating a contract that is asserted downstream only creates two
places to drift.

### 5.5 `occ_count` over-counting (historical; no longer in the entrypoint)

While the supervisor existed it counted occupiers by the tag alone, which
over-counts — the dangerous direction, since it would conclude occupiers exist,
never start them, and let the reaper take the cards. The path is real:
`extra_bashrc.sh`'s `free_gpu` ends in `pkill -f "$GPU_OCCUPY_TAG"`, so while that
`pkill` lives its argv carries the tag. Measured with a live `pkill`: **tag-only
count 1 → 2, two-condition count 0 → 0.**

Recorded because the same trap applies to anything that counts these processes: use
the launcher's two-condition predicate (tag **AND**
`gpu_occupy_(torch_mm.py|stub.sh)`), not the tag alone. The entrypoint no longer
counts occupiers at all, so this is history rather than live code.

---

## 6. Environment facts that are not obvious

* **The `navigation-hl` conda env cannot build the model.** Its `transformers`
  cannot resolve `GemmaForCausalLM`, so `openpi.models_pytorch.gemma_pytorch` fails
  to import. Same version (4.53.2) as the working env — apparently a partial
  install. **Use `behavior-data-hl`** for both `OPENPI_PREFLIGHT_PYTHON` and
  `CONDA_ROOT`. Do not "simplify" this back.
* **No FAST tokenizer existed on any H20-mounted volume**, and Variant A cannot
  tokenize without one while the pool runs offline. A `physical-intelligence/fast`
  snapshot is staged at `h20_fastce/fast_tokenizer` and verified to load with
  `local_files_only=True`. Calibration: an all-zero chunk yields **1** token, a
  uniform-random one **~601** — FAST length is violently content-sensitive, which is
  why a few-percent cap margin is not valid extrapolation.
* **H20 mounts** `behavior-data-hl`, `navigation-hl`, `robot-mllm-data-hl` and
  **not** `saiwenresearch`, so every path must be HL-side. Note `4dde119e` *is*
  reachable from H20 (via `behavior-data-hl`), so the normalization choice was a
  real scientific decision, not forced by mounts.
* **SSH and WebShell are both unavailable** on this job family (`ssh probe failed`;
  WebShell needs an interactive SSO refresh). The authoritative evidence channel is
  the NAS heartbeat the entrypoint writes.
* **`robust-hot-update` is unusable on this job lineage.** Arnold requires a
  non-empty gitlab `repo_name`; this lineage carries `git_repo: {"repo_name": ""}`
  (as does its parent), because the code lives on GitHub plus NAS worktrees. Four
  variants were tried across the v4 and v1 endpoints and both update modes; all were
  rejected at validation, costing nothing. It also only applies to a **running**
  trial.

---

## 7. Reproducing

```bash
# CPU-only preflight (no GPU, no output, no occupier action)
OPENPI_LAUNCH_PREFLIGHT_ONLY=1 OPENPI_H20_ARM=A OPENPI_H20_MODE=smoke \
OPENPI_EXPECTED_CODE_COMMIT=$(git rev-parse HEAD) \
NUM_NODES=4 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=127.0.0.1 GPU_MODEL=NVIDIA_H20 \
bash scripts/run_pi05_ki_formal_A_fast_bf16_4x8_h20.sh

# Contract tests
pytest tests/test_pi05_ki_h20_bf16_two_arm.py -q
bash tests/scripts/test_h20_train_entrypoint.sh

# Warm-start proof (loads the real checkpoint)
python scripts/verify_warm_start_keymap.py \
  --config pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16 \
  --checkpoint <arm's checkpoint dir> --actually-load --max-unexpected 0
```

The Merlin job's main command is a small shim that sets `H20_ARM`, `H20_MODE` and
`OPENPI_EXPECTED_CODE_COMMIT` and calls `scripts/h20_train_entrypoint.sh`. The shim
must never exit — if it cannot find the repo script it must start the tagged
occupiers and fall into an infinite keepalive, because an exit releases 32 H20 and a
bare `sleep` loop would sit at 0% utilization and be reaped.

---

## 8. What this experiment cannot tell you

* Whether Variant A produces a **capable policy**. `val_flow_loss` has already been
  shown not to rank task success on this family: a prior run reached its global
  minimum flow loss while scoring **0/24** on easy-8. Only closed-loop evaluation
  answers that.
* Which of **weights** or **normalization** drove any difference — they move
  together by design (§1).
* The **world-32 memory peak**, until a 4×8 smoke reports it. An 8-rank probe
  bounds parameters, gradient shard and optimizer shard conservatively (ZeRO-2 at
  8 ranks carries **+4.42 GB** more sharded state per GPU than at 32) and matches
  activations exactly at equal per-GPU batch — but it cannot bound inter-node NCCL
  buffers, world-size-dependent validation gathers, allocator fragmentation, or
  32-rank c10d bootstrap, none of which exist at 8 ranks on one node.
