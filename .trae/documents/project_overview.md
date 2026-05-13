# OpenPi Comet: Project Overview & Development Walkthrough

This repository implements Team Comet’s OpenPi-based submission for the 2025 BEHAVIOR-1K Challenge. It provides an end-to-end workflow for:
- Pre-training and fine-tuning π0/π0.5 (Pi05) style VLA policies (JAX/Flax and PyTorch variants)
- Serving policies via websocket for simulation/robot consumption
- Evaluating policies in the BEHAVIOR-1K / OmniGibson simulator
- Generating data (teleop + RFT dataset construction)

Primary reference: README.md at the repo root.

## Local Rules (Read First)

- Always activate the conda environment before running repo code/scripts:
  - `conda activate openpi-comet-nas`
- Data lives on NAS and IO can bottleneck training:
  - Increase DataLoader `num_workers` (single GPU: 16+; multi-GPU: 10+ per GPU), avoid `num_workers=0` when possible.

See: .trae/rules/project_rules.md

## Repo Map (What Lives Where)

- scripts/: main entrypoints
  - train.py: JAX/Flax training loop
  - train_dist.py: distributed training + checkpoint management (multi-GPU)
  - train_pytorch.py: PyTorch training entrypoint (used heavily by VLM2 workflow)
  - compute_norm_stats.py: compute normalization statistics (required for most real datasets)
  - serve_policy.py: serve a policy over websocket (generic OpenPi protocol)
  - serve_b1k.py: serve a policy over websocket with BEHAVIOR-1K adapter
  - test_*.py: lightweight sanity checks (dataset loading, inference, etc.)
- src/openpi/: core library code
  - models/: JAX/Flax model configs + modules (Pi0/Pi05, tokenizers, etc.)
  - models_pytorch/: PyTorch implementations (incl. VLM2 components)
  - training/: training configs, dataloaders, checkpointing, sharding
  - policies/: policy abstraction + checkpoint loading for inference/serving
  - serving/: websocket server implementation
  - shared/: normalization utilities + BEHAVIOR-1K wrappers
- src/behavior/learning/: BEHAVIOR-1K / OmniGibson rollout + evaluation integration
  - configs/: Hydra configs used by evaluator
  - eval_custom.py: customized evaluator / rollout runner (used for RFT rollouts too)
- packages/openpi-client/: minimal-dependency client for remote inference
- data_generation/: teleop tools and RFT dataset construction scripts
- docs/: additional notes (remote inference, docker, etc.)

## Configuration Systems (Two Worlds)

### Training configs (Tyro + dataclass “named configs”)

- Core definition: src/openpi/training/train_config.py
  - `TrainConfig` is a frozen dataclass describing model/data/optimizer/checkpoint/logging options.
  - `cli()` uses Tyro’s overridable config CLI to select a named config and override fields.
  - Config registry is assembled from:
    - src/openpi/training/pretrain_config.py
    - src/openpi/training/sft_config.py
    - src/openpi/training/rft_config.py
    - src/openpi/training/test_config.py

Practical impact:
- To add a new training recipe, add/modify a named config in one of the `*_config.py` modules above.

### Evaluation configs (Hydra)

BEHAVIOR-1K simulator evaluation uses Hydra configs under:
- src/behavior/learning/configs/

Entrypoint:
- src/behavior/learning/eval_custom.py

## Data & Transforms (Training ↔ Serving Consistency)

The project is strict about applying the same transforms in training and serving:

- Training data pipeline is built in src/openpi/training/data_loader.py
  - Loads dataset(s)
  - Applies repack transforms, data transforms, Normalize, then model transforms
  - Batches using PyTorch DataLoader (even for JAX training), then converts to JAX arrays when needed

- Serving pipeline loads the same normalization stats from checkpoint assets, not from config-time assets:
  - src/openpi/policies/policy_config.py:create_trained_policy

Common failure mode:
- “Normalization stats not found” → run:
  - `uv run scripts/compute_norm_stats.py --config-name <your-config-name>`

## Core Workflows

### 1) Install / Setup (Typical)

This repo supports a uv-based install (README) and a conda environment spec (environment.yml).

Recommended for this environment:
- `conda activate openpi-comet-nas`

If you need the uv/.venv workflow (README-style), refer to:
- README.md

BEHAVIOR-1K dependencies are required for some dataset/eval utilities; see:
- TRAIN_GUIDE.md (VLM2 workflow)

### 2) Compute Normalization Stats

Most real training requires per-config normalization stats. Compute them before training:

```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

### 3) Train / Finetune

JAX/Flax training:
- Entrypoint: scripts/train.py

Distributed pretrain:
- Entrypoint: scripts/train_dist.py

PyTorch training (VLM2-oriented):
- Entrypoint: scripts/train_pytorch.py
- See: TRAIN_GUIDE.md

Operational tip (NAS):
- Increase `--num_workers` to reduce IO stalls (project rule).

### 4) Serve a Checkpoint (Websocket)

Generic OpenPi websocket serving:
- scripts/serve_policy.py

BEHAVIOR-1K adapter serving (for OmniGibson evaluator):
- scripts/serve_b1k.py
- Uses the BEHAVIOR adapter wrapper in src/openpi/shared/eval_b1k_wrapper.py

Websocket protocol server implementation:
- src/openpi/serving/websocket_policy_server.py

Client package for remote inference:
- packages/openpi-client/
- Docs: docs/remote_inference.md

### 5) Evaluate in BEHAVIOR-1K / OmniGibson

Evaluator:
- src/behavior/learning/eval_custom.py

Hydra config:
- src/behavior/learning/configs/base_config.yaml

Typical flow:
1) Start policy server (serve_b1k.py)
2) Run OmniGibson evaluation with websocket policy (eval_custom.py or BEHAVIOR default eval)


## Where To Make Changes (Common Tasks)

- Add a new training config:
  - src/openpi/training/*_config.py
- Modify dataset loading / batching / worker behavior:
  - src/openpi/training/data_loader.py
- Modify normalization behavior:
  - src/openpi/transforms.py (normalize / unnormalize) and policy_config/data_loader integration
- Modify inference serving:
  - src/openpi/serving/websocket_policy_server.py
  - scripts/serve_policy.py or scripts/serve_b1k.py
- Modify BEHAVIOR-1K eval/rollout logic:
  - src/behavior/learning/eval_custom.py
  - src/openpi/shared/eval_b1k_wrapper.py

## Development Checklist (Fast Sanity)

- Confirm environment:
  - conda env is `openpi-comet-nas`
- Confirm norm stats exist for your config:
  - run compute_norm_stats if needed
- Smoke-test dataset loading / inference:
  - scripts/test_*.py (choose the one matching your task)
- For VLM2 training & offline validation:
  - follow TRAIN_GUIDE.md exactly
