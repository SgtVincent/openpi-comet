---
name: "pi3x-3d-reconstruction"
description: "Runs Pi3X on Behavior1K videos to reconstruct a colored point cloud (.ply) and visualizes it in viser. Invoke when you want Pi3/Pi3X reconstruction + web viewer on an episode mp4."
---

# Pi3X 3D Reconstruction (B1K)

## What This Skill Does

- Uses Pi3X (from `src/openpi/third_party/Pi3`) to reconstruct a colored point cloud from a B1K episode mp4.
- Saves `pi3x_reconstruction.ply` and `metadata.json`.
- Launches a viser viewer to inspect the `.ply` interactively in browser.

## When To Invoke

- When the user asks to run Pi3 / Pi3X on a Behavior1K episode (full episode or a quick sample).
- When the user wants a `.ply` output and a simple web viewer comparison.

## Prerequisites

1. Repo clone (already placed in this workspace):
   - `src/openpi/third_party/Pi3`
2. Conda env:
   - `conda activate pi3x-nas`
3. Packages:
   - `pip install -r src/openpi/third_party/Pi3/requirements.txt`
   - `pip install viser`

If weights download is slow:
- Run `proxy_on` per workspace rule, or download `model.safetensors` manually and pass `--ckpt`.

## Commands

### 1) Reconstruct (B1K wrapper)

```bash
conda activate openpi-comet-nas
conda activate pi3x-nas
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet

python scripts/pi3x_reconstruct_b1k.py \
  --episode_id 130010 \
  --camera head \
  --interval 30
```

Optional:

- Use local weights:

```bash
python scripts/pi3x_reconstruct_b1k.py \
  --episode_id 130010 \
  --camera head \
  --interval 30 \
  --ckpt /path/to/model.safetensors
```

### 2) Visualize (viser)

```bash
conda activate openpi-comet-nas
conda activate pi3x-nas
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet

python scripts/visualize_ply_viser.py \
  --ply_path outputs/pi3x_exp/task0013_ep0010_head_pi3x_i30/pi3x_reconstruction.ply \
  --port 8099
```

Open:
- `http://localhost:8099/`

## Outputs

- `outputs/pi3x_exp/<exp_name>/pi3x_reconstruction.ply`
- `outputs/pi3x_exp/<exp_name>/metadata.json`
