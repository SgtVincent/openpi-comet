#!/usr/bin/env python3
"""`subtask_max_len` 作用域守卫。

为什么需要它
------------
`subtask_max_len` 定义在 `Pi05SubtaskConfig` 上，而这个类被**正在跑的**
`annotations_skill` / skillbridge 实验共用。MoMA-VLA 要把上限抬到 160，如果有人
图省事直接改类默认值，就会静默改掉在飞实验的 padding 长度 —— 不报错、不崩、
只是别人的实验换了个 padding。

所以这里断言的是**每个配置各自解析出来的值**，逐个列出，而不是「至少有一个是
128」。同时单独断言类默认值没被动过。

用法
----
    PYTHONPATH=<worktree>/src <env-python> config_scope_guard.py \\
        --expect expectations.json

`expectations.json` 形如：

    {
      "class_default_subtask_max_len": 128,
      "per_config_subtask_max_len": {
        "pi05_subtask_b1k-pt50_cs32_bs64_lr2.5e-5_5ep": 128,
        "pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep": 128,
        "pi05_b1k_skill-pt12_pretrain_lr1e-4_2ep": 128
      },
      "require_exact_config_set": true
    }

`--dump` 只打印当前解析值，用来生成/更新期望文件。
退出码 0 = 全部相符，1 = 有不符，3 = 读不到配置（NOT-MEASURED，不当成通过）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def resolve_configs() -> tuple[dict[str, Any], int, list[str]]:
    """返回 (配置名 -> 解析出的字段, 类默认 subtask_max_len, 诊断行)。"""
    diag: list[str] = []
    # 导入顺序很重要：先 train_config 再 pretrain_config，否则循环导入。
    import openpi.training.train_config as tc

    import openpi.models.pi05_subtask_config as sc

    class_default = int(sc.Pi05SubtaskConfig.__dataclass_fields__["subtask_max_len"].default)
    diag.append(f"class default Pi05SubtaskConfig.subtask_max_len = {class_default}")

    # 两个来源都要看：
    #   ① `_CONFIGS_DICT` —— 真正能被 `--config <name>` 选中的全部配置
    #   ② `moma_memory_config.MOMA_MEMORY_CONFIGS` —— 定义了但**可能没并入 _CONFIGS**
    # 只看 ① 会漏掉尚未注册的新配置；只看 ② 会漏掉别人的在飞实验。
    candidates: list[tuple[Any, bool]] = [(c, True) for c in tc._CONFIGS_DICT.values()]
    try:
        import openpi.training.moma_memory_config as mm

        for c in mm.MOMA_MEMORY_CONFIGS:
            candidates.append((c, c.name in tc._CONFIGS_DICT))
        diag.append(f"moma_memory_config.MOMA_MEMORY_CONFIGS: {len(mm.MOMA_MEMORY_CONFIGS)} 个")
    except Exception as exc:  # noqa: BLE001
        diag.append(f"moma_memory_config 不可用（{type(exc).__name__}），只看已注册配置")

    out: dict[str, Any] = {}
    for cfg, registered in candidates:
        model = getattr(cfg, "model", None)
        if not isinstance(model, sc.Pi05SubtaskConfig) or cfg.name in out:
            continue
        data = cfg.data if isinstance(cfg.data, (list, tuple)) else [cfg.data]
        out[cfg.name] = {
            "subtask_max_len": int(model.subtask_max_len),
            # ⚠️ prompt 预算的**配置字段名是 max_token_len**，不是 prompt_max_len。
            # 后者只是 SubtaskTokenizer 的构造参数名，生产在 data_config.py:274 做
            # `prompt_max_len=model_config.max_token_len` 的映射。把它当配置字段会崩。
            "max_token_len": int(model.max_token_len),
            "action_dim": int(model.action_dim),
            "action_horizon": int(model.action_horizon),
            "planner_stride": int(getattr(model, "planner_stride", -1)),
            "num_workers": int(cfg.num_workers),
            "batch_size_per_gpu": cfg.batch_size_per_gpu,
            "subtask_source": [getattr(getattr(d, "base_config", None), "subtask_source", None)
                               for d in data],
            # 没注册 ⇒ `--config <name>` 选不中它；而 get_config() 对未注册名字是
            # **静默回退**到 pi05_b1k-base（train_config.py:325-330），与拼错名字
            # 的表现完全相同。
            "registered_in_CONFIGS": bool(registered),
        }
    diag.append(f"resolved {len(out)} Pi05SubtaskConfig configs "
                f"（已注册 {sum(1 for v in out.values() if v['registered_in_CONFIGS'])}）")
    return out, class_default, diag


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expect", type=Path, default=None)
    p.add_argument("--dump", action="store_true", help="只打印当前解析值")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    try:
        resolved, class_default, diag = resolve_configs()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"verdict": "CANNOT-ASSESS",
                          "error": f"{type(exc).__name__}: {exc}",
                          "note": "读不到训练配置 ⇒ NOT-MEASURED，不当成通过"}, ensure_ascii=False, indent=2))
        return 3

    for line in diag:
        print(f"[resolve] {line}")

    if args.dump or not args.expect:
        payload = {
            "class_default_subtask_max_len": class_default,
            "per_config_subtask_max_len": {k: v["subtask_max_len"] for k, v in resolved.items()},
            "require_exact_config_set": True,
            "_full_resolved": resolved,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.out:
            args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    exp_per = expect.get("per_config_subtask_max_len") or {}
    exp_default = expect.get("class_default_subtask_max_len")

    rows: list[dict[str, Any]] = []
    ok = True

    if exp_default is not None:
        good = class_default == exp_default
        ok &= good
        rows.append({"item": "class_default", "expected": exp_default,
                     "observed": class_default, "ok": good,
                     "why": "类默认值被改会静默改掉所有共用该类的在飞实验"})

    for name, want in exp_per.items():
        got = resolved.get(name, {}).get("subtask_max_len")
        good = got == want
        ok &= good
        rows.append({"item": f"{name}.subtask_max_len", "expected": want, "observed": got, "ok": good,
                     "why": "缺失（配置被删/改名）也算不符，不能当成通过" if got is None else ""})

    for name, want in (expect.get("per_config_max_token_len") or {}).items():
        got = resolved.get(name, {}).get("max_token_len")
        good = got == want
        ok &= good
        rows.append({"item": f"{name}.max_token_len", "expected": want, "observed": got, "ok": good,
                     "why": "字段名是 max_token_len；prompt_max_len 只是 tokenizer 构造参数名"})

    for name, want in (expect.get("per_config_registered") or {}).items():
        got = resolved.get(name, {}).get("registered_in_CONFIGS")
        good = got == want
        ok &= good
        rows.append({"item": f"{name}.registered_in_CONFIGS", "expected": want, "observed": got,
                     "ok": good,
                     "why": "未注册时 get_config() 静默回退到 pi05_b1k-base，与拼错名字无法区分"})

    if expect.get("require_exact_config_set", True):
        extra = sorted(set(resolved) - set(exp_per))
        good = not extra
        ok &= good
        rows.append({"item": "no_unlisted_Pi05SubtaskConfig", "expected": [],
                     "observed": extra, "ok": good,
                     "why": "新增的 Pi05SubtaskConfig 未登记 ⇒ 它的上限没人管"})

    print()
    for r in rows:
        tag = "PASS" if r["ok"] else "FAIL"
        print(f"  [{tag}] {r['item']}: expected={r['expected']} observed={r['observed']}"
              + (f"   ({r['why']})" if r["why"] else ""))

    report = {"verdict": "PASS" if ok else "FAIL", "class_default": class_default,
              "resolved": resolved, "rows": rows}
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"SCOPE_GUARD_VERDICT: {report['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
