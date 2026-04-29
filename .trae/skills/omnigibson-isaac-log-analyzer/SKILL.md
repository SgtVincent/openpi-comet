---
name: "omnigibson-isaac-log-analyzer"
description: "分析 OmniGibson/Isaac Sim 仿真日志与终端报错。遇到 Traceback、[Error]、HydraEngine/renderer/扩展加载、Segmentation fault 或渲染失败时调用。"
---

# OmniGibson Isaac Log Analyzer

## 目标

给定 OmniGibson / Isaac Sim 的终端输出、Kit 日志、评测日志或用户圈选的报错片段，快速完成：
- 提取最后一个真实失败点，而不是被大量 warning 干扰
- 区分 Python 异常、Kit `[Error]`、renderer / driver / extension 问题、以及 native crash
- 结合上下文判断错误发生阶段：启动、extension hydration、环境构建、scene import、sensor / rendering、退出卸载
- 输出下一步最小验证动作与最小修复建议

## 何时调用

出现以下任一情况就调用：
- 用户贴了 Terminal 片段，里面有 `Traceback`、`[Error]`、`Fatal Python error`、`Segmentation fault`
- OmniGibson / Isaac Sim 启动失败、卡住、黑屏、headless 渲染失败
- 报错涉及 `HydraEngine rtx failed creating scene renderer`、driver verify、`No module named 'omni.*'`
- bbox / camera / synthetic data / replicator 相关脚本运行失败
- 日志里 warning 非常多，需要快速找出真正根因

## 输入优先级

优先按以下顺序收集证据：
1. 用户圈选的终端报错片段
2. 本轮运行命令本身及其环境变量
3. 当前脚本的关键阶段日志
4. Kit 日志或 `tee` 落盘日志
5. 相关源码行与最近本地修改

## 工作方式

1. 先定位最后一个明确失败信号
   - 优先找：`Traceback`、`TypeError:`、`RuntimeError:`、`ModuleNotFoundError:`、`Fatal Python error`、`Segmentation fault`
   - 如果同时存在 Python 异常和 segfault，先把 Python 异常当成主故障，segfault 通常是退出阶段连带崩溃

2. 再区分错误所在层
   - Python 层：类型不匹配、数组 dtype、导入失败、参数错误
   - Kit / extension 层：`[Error]`、extension startup failure、`No module named 'omni.*'`
   - Renderer / driver 层：`HydraEngine rtx failed creating scene renderer`、driver verify、display / GLFW
   - Native crash 层：`Fatal Python error: Segmentation fault`

3. 再判断故障阶段
   - 启动前：解释器、环境、依赖
   - Simulation App Starting 前后：Kit / extension / renderer
   - `Imported scene 0.` 前后：scene import / object load / USD → trimesh
   - `viewer_camera` / `cam.get_obs` 附近：sensor / replicator / image dtype
   - 脚本结束时：shutdown / plugin unload

4. 输出必须包含
   - 现象一句话总结
   - 最关键证据 3~8 条
   - 最可能根因 Top 1~3
   - 下一步最小动作

## 快速模式

如果用户只贴了几十行终端错误：
- 直接抓最后一个 `Traceback` 块或最后一个 `[Error] [py stderr]` 块
- 提取最末尾异常类型与异常信息
- 判断它是否是“根因”还是“次生崩溃”

例如：
- `TypeError: Cannot handle this data type: (1, 1, 4), <i8`  
  结论优先指向图像数组 dtype 不符合 Pillow 预期，而不是 segfault
- `HydraEngine rtx failed creating scene renderer`  
  结论优先指向 renderer / driver / headless 配置
- `No module named 'omni.kit.telemetry'`  
  结论优先指向 extension hydration 或启动方式错误

## 推荐脚本

如果有长日志文件，优先运行：

```bash
python .trae/skills/omnigibson-isaac-log-analyzer/scripts/sim_log_inspect.py \
  /path/to/log.txt \
  --topk 60 \
  --context 40
```

如果要分析管道输入：

```bash
cat /path/to/log.txt | python .trae/skills/omnigibson-isaac-log-analyzer/scripts/sim_log_inspect.py -
```

## 常见模式库

- **Driver / RTX**
  - `The currently installed NVIDIA graphics driver is unsupported`
  - `HydraEngine rtx failed creating scene renderer`
  - `verifyDriverVersion`

- **Display / Headless**
  - `failed to open the default display`
  - `carb.windowing-glfw.plugin`

- **Extension / Launcher**
  - `No module named 'omni.*'`
  - `Failed parsing execute string`
  - 某些扩展缺失但 `isaacsim` launcher 路径能自动 hydration

- **Torch / Numpy / Trimesh**
  - `expected np.ndarray (got numpy.ndarray)`
  - `Cannot interpret 'dtype(...)' as a data type`
  - `apply_transform`

- **Image / Sensor**
  - `Cannot handle this data type`
  - `Image.fromarray`
  - `colorize_bboxes`
  - `cam.get_obs`

- **Native Crash**
  - `Fatal Python error: Segmentation fault`
  - 如果前面已经有 Python 异常，优先修前者

## 本次这类错误的分析模板

当看到类似下面的片段：

```text
[Error] [omni.kit.app._impl] [py stderr]: Traceback ...
TypeError: Cannot handle this data type: (1, 1, 4), <i8
Fatal Python error: Segmentation fault
```

按以下顺序分析：
1. 主错误是 `TypeError`
2. 发生位置在 `PIL.Image.fromarray(...)`
3. 上游数据来自 bbox / rgb 图像数组，重点检查 shape 与 dtype
4. `Segmentation fault` 先视为退出阶段次生崩溃，除非没有更早的 Python 异常
5. 最小修复优先尝试把图像数组转成 `uint8`

## 输出格式

- **现象**：一句话总结
- **证据**：列出关键日志片段与文件/行号
- **根因假设**：Top 1~3
- **下一步动作**：最小验证命令或最小代码修改
- **风险**：是否可能还有 shutdown 阶段的次生 segfault
