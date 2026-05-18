# Debug Session: multinode-dataset-root
- **Status**: [OPEN]
- **Issue**: Arnold 多节点 Accelerate 训练在构建 BEHAVIOR 数据集时解析到了错误且不可写的默认根目录，触发 mkdir PermissionError。
- **Debug Server**: N/A
- **Log File**: N/A

## Reproduction Steps
1. 在 Arnold 4 node x 8 GPU 环境启动 `scripts/run_pi05_sft_accelerate_deepspeed_multinode.sh`
2. 训练初始化进入 `build_datasets()`
3. `behavior/learning/datas/dataset.py` 尝试对解析出的 dataset root 执行 `mkdir(parents=True, exist_ok=True)`
4. 根路径落到 `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos`
5. 当前环境对 `/mnt/bn/robot-mllm-data-lf-3` 无权限，触发 `PermissionError: [Errno 13]`

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | 多节点 Accelerate 启动脚本没有像单机脚本一样显式导出 `OPENPI_BEHAVIOR_DATASET_ROOT`，导致回退到配置默认根路径 | High | Low | Pending |
| B | `data_config.py` 的环境变量覆盖逻辑在 Accelerate 多节点路径中没有生效或生效过晚 | Med | Low | Pending |
| C | `behavior` 数据集类对根目录执行 `mkdir` 是预期行为，但当前默认路径在 Arnold 容器内既不存在也不可写 | High | Low | Pending |
| D | 只有部分 rank 缺少数据根路径环境变量，导致 rank 间看到不同的 root | Med | Med | Pending |

## Log Evidence
- 见 `checkpoints/console_logs/pi05_b1k-make_pizza_lr1e-4_5ep_sft_accel_ds_z2_4n8g_20260417_032543/node0.log`

## Verification Conclusion
- Pending
