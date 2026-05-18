# Optimizer 内存优化指南

本文档介绍了在 `openpi-comet` 项目中如何使用 8-bit Optimizer 来显著降低 GPU 显存占用，并提供了实际测试的效果对比。

## 1. 概述

对于参数量巨大的模型（如 PI05/VLM2，约 2.3B 参数），优化器状态（Optimizer States）往往是显存占用的主要来源之一。标准的 `AdamW` 优化器为每个可训练参数维护两个浮点数状态（动量 $m$ 和方差 $v$），在 FP32 精度下，每个参数需要额外消耗 8 字节显存。

通过使用 `bitsandbytes` 提供的 **8-bit AdamW**，我们可以将优化器状态量化为 8 位，从而将这部分显存占用降低为原来的 1/4（从 8 字节降至 2 字节），且对训练精度的影响几乎可以忽略不计。

## 2. 如何使用

在当前代码框架下，我们已经通过环境变量集成了 8-bit 优化器的开关。

### 开启 8-bit 优化器
在运行训练脚本（如 `scripts/run_*.sh`）时，设置环境变量 `USE_8BIT_OPTIM=1`：

```bash
export USE_8BIT_OPTIM=1
bash scripts/run_pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep.sh
```

### 代码实现
在 `scripts/train_pytorch.py` 中，系统会自动尝试导入 `bitsandbytes`。如果安装成功且环境变量开启，将自动切换优化器。

## 3. 效果对比

我们在单卡 NVIDIA A100 (40GB) 环境下，使用 `pi05_b1k_skill-pt50_pretrain` 配置（`batch_size_per_gpu=2`）进行了实测：

| 优化器类型 | Allocated Memory (实际占用) | Reserved Memory (预留总计) | 显存节省 |
| :--- | :--- | :--- | :--- |
| **Standard AdamW (FP32)** | 28.41 GB | 36.69 GB | - |
| **8-bit AdamW (INT8)** | **21.37 GB** | **24.42 GB** | **~12.27 GB (Reserved)** |

### 关键结论
*   **显存预留 (Reserved) 降低了约 12.27 GB**。
*   这省下的空间足以支持将 `batch_size_per_gpu` 从 2 提升到 4 甚至更多，或者为更长序列的 VLM2 模型腾出空间。

## 4. 常见问题 (FAQ)

**Q: 8-bit 会影响模型收敛吗？**
A: 根据 `bitsandbytes` 官方和业界的广泛测试，8-bit AdamW 在绝大多数视觉和语言模型任务中，其收敛曲线与 FP32 几乎重合，是一种非常成熟的“免费午餐”式优化。

**Q: 如果环境里没有安装 bitsandbytes 怎么办？**
A: 代码中做了平滑降级处理。如果导入失败，系统会打印警告并自动回退到标准的 `torch.optim.AdamW`，不会中断训练。

---
*文档更新日期：2026-03-26*
