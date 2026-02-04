<p align="center">
  <img src="docs/teaser.png" width="100%">
</p>

<p align="left">
  <a href="https://behavior.stanford.edu/index.html">
    <img
      src="https://img.shields.io/badge/BEHAVIOR--1K-Website-red?logo=googleplay&logoColor=white"
      alt="BEHAVIOR-1K Website"
    />
  </a>
  <a href="https://behavior.stanford.edu/challenge/leaderboard.html#privileged-information-track">
    <img
      src="https://img.shields.io/badge/BEHAVIOR--1K-Leaderboard-5865F2?logo=googleplay&logoColor=white"
      alt="BEHAVIOR-1K Leaderboard"
    />
  </a>
  <a href="https://huggingface.co/sunshk/openpi_comet">
    <img
        src="https://img.shields.io/badge/Model-HuggingFace-green?logo=huggingface&logoColor=brightyellow"
        alt="Model"
    />
  </a>
  <a href="https://huggingface.co/datasets/delinqu/comet-1.5k">
    <img
        src="https://img.shields.io/badge/Data-HuggingFace-green?logo=huggingface&logoColor=brightyellow"
        alt="Data"
    />
  </a>
  <a href="https://arxiv.org/abs/2512.10071">
    <img
      src="https://img.shields.io/badge/Comet--Submission-Paper-red?logo=arxiv&logoColor=red"
      alt="Implementation Report"
    />
  </a>
</p>

# Openpi Comet

> [!TIP]
> OpenPi Comet is the submission of Team Comet for the [2025 BEHAVIOR Challenge](https://behavior.stanford.edu/index.html). This repository provides a unified framework for pre-training, post-training, data generation and evaluation of π0.5 (Pi05) models on BEHAVIOR-1K.

Our [[submission]](https://behavior.stanford.edu/challenge/leaderboard.html#privileged-information-track) achieved a Q-score of **0.2514 (Held-out Test)**, securing 2nd place overall and finishing behind the winning team by a narrow margin—highlighting both the strong competitiveness of our approach and the effectiveness of our end-to-end VLA training strategy. 

**🏆 Post-Challenge Update.**  
Building upon our **competition submission**, we further refined the training strategy, leading to a significantly higher **Q-score of 0.345 on the Public Validation set** using only [two pretrained models](#model-zoo).

<p align="center">
  <img src="docs/leaderboard.png" width="80%">
</p>


This codebase contains:
1. Distributed OpenPi training infrastructure, Support Multi-datasets Sharding.
2. Various pre-training setup, including hierarchical instructions (global, subtask, skill) and multimodal observations (RGB, depth, point cloud, segmentation, bounding boxes, human pointing)
3. Post-training via Rejection Sampling Fine-Tuning (RFT) with automated dataset construction
4. Data generation scripts such as teleoperation and simulation rollouts using existing policy
5. Native OpenPi compatibility, SOTA performance with minimal modifications (can be directly loaded by official OpenPi)


Please check our [[Report]](https://arxiv.org/abs/2512.10071) for more details.

<div align="center">
  <video src="https://github.com/user-attachments/assets/2644bb91-76a0-4329-ab83-06c2f04f4395" controls width="720">
  </video>
</div>

## Updates

- [Dec 06, 2025] Released the full submission codebase and pre-trained weights.
- [Jan 03, 2026] Release our [latest pretrained models](#model-zoo) with **Q Score of 0.345** (Public Validation), Upload our RFT dataset: [comet-1.5k 🤗](https://huggingface.co/datasets/delinqu/comet-1.5k), Support Multi-dataset Loading and Sharding.


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Repo Clone

```bash
git clone https://github.com/mli0603/openpi-comet.git
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```
This finetuning instruction is adapted from the original [openpi repo](https://github.com/Physical-Intelligence/openpi).

## Installation

Openpi uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
cd openpi-comet
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

source .venv/bin/activate

# Install behavior for server deploy 
cd $PATH_TO_BEHAVIOR_1K
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
```

## Model Zoo

We provide a suite of base VLA model checkpoints trained on 1.5K hours robot trajectories, ideal for BEHAVIOR-1K fine-tuning.

|   Model Name | Discription                          | HuggingFace URL                                                                                                                                                                                                                                          |
|----------:|:----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pi05-b1kpt12-cs32 | Pretrained Model in tasks `0,1,6,17,18,22,30,32,34,35,40,45` with action chunk size 32                         | [openpi_comet/pi05-b1kpt12-cs32](https://huggingface.co/sunshk/openpi_comet/tree/main/pi05-b1kpt12-cs32)                                                                  |
| pi05-b1kpt50-cs32 | Pretrained Model in tasks `0-49` with action chunk size 32                                                     | [openpi_comet/pi05-b1kpt50-cs32](https://huggingface.co/sunshk/openpi_comet/tree/main/pi05-b1kpt50-cs32)                                                                  |

### Finetune OpenPi

When resuming training from an existing pretrained checkpoint, we strongly recommend inheriting the normalization statistics from the pretrained model, rather than recomputing them. This ensures consistency in feature scaling and avoids distributional shifts that may destabilize training or degrade performance.

You can also recompute normalization statistics when starting training from scratch:
```bash
# Optional
> uv run scripts/compute_norm_stats.py --config-name pi05_b1k-turning_on_radio
```

This will create `norm_stats.json` under `assets/pi0_b1k/behavior-1k/2025-challenge-demos`, which will be used to normalize the training data.

Update the configs in `src/openpi/training/config.py` to be the task name you want (or None to include all tasks), for example, you can update the configs as follows for the `turning_on_radio` task:

```python
TrainConfig(
    name="pi05_b1k-turning_on_radio",
    exp_name="openpi",
    project_name="B1K",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
    data=LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=list(range(200)),
            behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
            tasks=["turning_on_radio"],
            fine_grained_level=0,  # 0: global instruction, 1: subtask instruction, 2: skill instruction
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "The Model Path you want to finetune from, e.g., gs://openpi-assets/checkpoints/pi05_base/params\
        or the checkpoint from our model zoo"
    ),
    num_train_steps=30_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        peak_lr=2.5e-5,
        decay_steps=30_000,
    ),
    freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
    ema_decay=None,
    checkpoint_base_dir=".",
    num_workers=8,
    batch_size=8 * 32,
),
```

Then run the following command to fintune OpenPi:
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 
uv run scripts/train.py \
    pi05_b1k-turning_on_radio \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)"
```

### Pre-train OpenPi

To support distributed training, we update `src/openpi/training/data_loader.py` for data sharding, and the `src/openpi/training/checkpoints_dist.py` and `scripts/train_dist.py` for distributed checkpointing management and training. To launch the pretrain, run the following command:

```bash
# set dist training envs
export MASTER_ADDR=${SERVICE_PREFIX}-0.${SUBDOMAIN}
export WORLD_SIZE=${LEPTON_JOB_TOTAL_WORKERS}
export WORLD_RANK=${LEPTON_JOB_WORKER_INDEX}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12350

config_name=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40
exp_name=pi05_b1k-pt50_pretrain

python scripts/compute_norm_stats.py --config-name ${config_name}

python scripts/train_dist.py ${config_name} --exp_name=${exp_name} --overwrite
```

### Post-train OpenPi using Rejection Sampling fine-tuning (RFT)

To perform RFT, you need to first deploy the finetuned checkpoint, and then rollout the episodes in the BEHAVIOR-1K Simulator. We also observe that the `pose perturbator` helps improve the robustness of the RFT Algorithm. 

1. Copy the files in `openpi-comet/src/behavior/learning` to `BEHAVIOR-1K/OmniGibson/omnigibson/learning`. Be careful to the latest commit of the BEHAVIOR-1K repo and replace the files in the CheckList:

| Name | Description |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| `configs/base_config.yaml`    | hydra config with additional params, e.g., `env_wrapper`, `perturb_pose`, and `parallel_evaluator`.   |
| `wrappers/rgb_wrapper.py`     | Full Resolution RGB Wrapper, Helpful for evaluation                                                   |
| `wrappers/__init__.py`        | Register the RGBWrapper                                                                               |
| `pose_perturbator.py`         | Pose Perturbator in RFT Rollout                                                                       |
| `eval_custom.py`              | Custom Evaluation Script                                                                              |

2. Run the RFT rollout in parallel:

```bash
python OmniGibson/omnigibson/learning/eval_custom.py policy=websocket \
    save_rollout=true \
    perturb_pose=true \
    task.name=$TASK_NAME \
    log_path=./outputs/rft \
    use_parallel_evaluator=false \
    parallel_evaluator_start_idx=0 \
    parallel_evaluator_end_idx=10 \
    model.port=8000 \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
```
where `parallel_evaluator_start_idx` and `parallel_evaluator_end_idx` are the start and end index of the parallel rollout, we can distribute the rollout to multiple GPUs by splitting the total number of instances into multiple parts.

3. Build the RFT dataset:
After the rollout, you can build the RFT dataset by following [Data Generation README.md](data_generation/rft/README.md)

Then, we can perform RFT training on the RFT dataset. Please refer to the [RFT training config](src/openpi/training/config.py) for more details.

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:

    ```
    source .venv/bin/activate

    uv run scripts/serve_b1k.py \
      --task_name=$TASK_NAME \
      --control_mode=receeding_horizon \
      --max_len=32 \
      policy:checkpoint \
      --policy.config=pi05_b1k-base \
      --policy.dir=$PATH_TO_CKPT
    ```
    This opens a connection listening on 0.0.0.0:8000. Please check the `scripts/serve_b1k.py` for more details.


2. Run the evaluation on BEHAVIOR:

    Assume you have behavior env installed (check https://github.com/StanfordVL/BEHAVIOR-1K for more details), run the following command within the BEHAVIOR-1K directory:
    
    ```bash
    conda activate behavior 
    
    python OmniGibson/omnigibson/learning/eval.py \
      policy=websocket \
      task.name=$TASK_NAME \
      log_path=$LOG_PATH
      # env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
    ```
    NOTE: We recommend to use the RGBWrapper for evaluation, please follow the instructions in [Post-train OpenPi using Rejection Sampling fine-tuning (RFT)](#post-train-openpi-using-rejection-sampling-fine-tuning-rft) to add `RGBWrapper` or custom the evaluation script.


## FAQs

If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions!

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{bai2025openpicometcompetitionsolution,
  title={Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge}, 
  author={Junjie Bai and Yu-Wei Chao and Qizhi Chen and Jinwei Gu and Moo Jin Kim and Zhaoshuo Li and Xuan Li and Tsung-Yi Lin and Ming-Yu Liu and Nic Ma and Kaichun Mo and Delin Qu and Shangkun Sun and Hongchi Xia and Fangyin Wei and Xiaohui Zeng},
  journal={arXiv preprint arXiv:2512.10071},
  year={2025},
  url={https://arxiv.org/abs/2512.10071}, 
}
```

## Third-party dependencies (submodules) ⚙️

This repository uses git submodules for some third-party components located under `src/openpi/third_party` (e.g., `cut3r`, `vggt`). After cloning the repo, run:

```bash
git submodule update --init --recursive
```

To update submodules to latest remote branch:

```bash
git submodule update --remote --merge
```

