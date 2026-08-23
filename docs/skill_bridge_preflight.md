# Skill Bridge Baseline — Training Preflight

Checklist before launching a skill-bridge training run.
Phase 1: combined `subtask_text` for valid single-boundary contiguous skill
crossings, emitted only for `annotations_skill` source, only when action chunk
has no padding.

## 1. Dataset Verification

- [ ] B1K dataset root exists and matches the expected SHA / version
- [ ] `subtask_source = "annotations_skill"` is set in the data config
      (Phase 1 bridge only activates for this source)
- [ ] Skill annotation JSONs exist for all episodes in the split
- [ ] Run `audit_episode_segments` on a sample of episodes:
  - No overlaps (hard error if found)
  - Gaps ≤ 1 frame (warn only; > 1 frame gaps are skipped by bridge)
  - All segments have non-empty phrases
- [ ] Run `audit_chunk_crossing_stats` on 1000+ random chunks:
  - `valid_bridge_ratio` is non-zero and reasonable (expect 5–20% for 32-step chunks)
  - `single_skill` + `valid_bridge` + rejections = total samples
  - Most common rejection reason documented

## 2. Config Verification

- [ ] `data.skill_bridge.enabled = True` in the training config
- [ ] `data.skill_bridge.min_pre_boundary_steps` = 1 (default) or tuned
- [ ] `data.skill_bridge.min_post_boundary_steps` = 1 (default) or tuned
- [ ] `data.subtask_source = "annotations_skill"` (bridge only activates for this source)
- [ ] `action_horizon` matches the chunk size used for bridge stats
- [ ] `chunk_streaming_using_keyframe = True` (default B1K — streaming path passes
      actual query indices + pad mask for accurate boundary detection)
- [ ] `streaming_drop_incomplete_horizon = True` (recommended — drops padded
      chunks at episode boundaries; bridge also rejects any padded chunk)

## 3. Implementation Notes

**Wiring:**
- `DataConfig.skill_bridge` → `create_behavior_dataset()` →
  `BehaviorLeRobotDataset(skill_bridge_config=...)` →
  `BehaviorLeRobotDataset._get_bridge_subtask_text()` →
  `get_bridge_subtask_text()` (integration helper) → `compute_bridge_info()` (core)

**Active paths:**
- **Streaming path** (default B1K): uses actual `_get_query_indices` frame
  indices + `action_is_pad` mask for precise boundary detection
- **Non-streaming path**: uses `anchor_frame + action_horizon` fallback

**Bridge rejection rules (Phase 1):**
- `skill_bridge.enabled = False` → no bridge
- `subtask_source != "annotations_skill"` → no bridge
- Any padded action step → no bridge (we do NOT trim to valid prefix)
- Not exactly one contiguous skill crossing → no bridge
- Fewer than `min_pre_boundary_steps` before boundary → no bridge
- Fewer than `min_post_boundary_steps` after boundary → no bridge

## 4. Expected Impact

- **Subtask CE loss**: may increase initially since combined phrases are longer
  and the model must learn "then" transition semantics
- **Action flow loss**: no direct change in Phase 1 (conditioning text only)
- **Convergence**: may take longer because bridge samples have a different
  conditioning distribution; monitor subtask accuracy separately for
  single-skill vs bridge samples

## 5. Hardware / Resource Notes

### V100 (8×8 = 64 GPU, FP32)
- Baseline config: `pi05_ki_joint_query_b1k-multitask-ki_on_500step_fp32`
- Bridge config: `pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32` (smoke)
- Bridge adds no model parameters; data loading overhead is negligible
  (bisect + string concat per sample)
- Expect similar step time and memory footprint to baseline

### A100 (4×8 = 32 GPU, BF16)
- Baseline config: `pi05_ki_joint_query_b1k-full_task-ki_on_bf16`
- Bridge is data-only — no numerical precision implications
- Expect identical step time and memory to baseline

## 6. Sanity Checks (first 100 steps)

- [ ] `subtask_text` contains " then " in ~expected ratio of samples
      (5–20% for 32-step chunks with contiguous skills)
- [ ] No crashes in the data loader (bridge code has zero side effects,
      all rejections fall through to original anchor-frame phrase)
- [ ] Loss is finite and decreasing (same order of magnitude as baseline)
- [ ] No per-sample log spam from bridge code (there shouldn't be any)

## 7. Rollback

To disable the bridge: set `data.skill_bridge.enabled = False` and re-launch.
No checkpoint migration needed — model architecture is unchanged.
When disabled, `_get_bridge_subtask_text` is byte-identical to `_get_subtask_text`.
