#!/usr/bin/env python3
"""MoMA-VLA 正式训练准入 gate（离线、全量、可重跑）。

作用域与分工
------------
本脚本只做**离线数据判定**：在全量数据上证明当前数据不越界。它**不能替代**训练
代码里的运行时 hard error（逐样本、在线）。文档 3.4.5 要求「未命中区间 / token
超长 / EOS 丢失」是 hard error，而训练侧 `tokenizer.py` 目前只有 warning + 截断
（`:515-519` prompt、`:550-557` subtask）；代码侧由训练实现的 worker 负责改。
两层都需要，谁也不覆盖谁。

量的是哪把尺子
--------------
target/prompt 的 token 长度一律用训练侧真正会用的
`openpi.models.tokenizer.SubtaskTokenizer`（配置取自
`openpi.models.pi05_subtask_config.Pi05SubtaskConfig`）。运行时会记录该
tokenizer 解析到的 SentencePiece 模型文件路径、md5、vocab_size 以及
`sentencepiece` 库版本，写进报告，便于和 `il_lib` 的审计结果逐项对照。

用法
----
    PYTHONPATH=<openpi-worktree>/src <env-python> moma_pretrain_gate.py \
        --data-root <fixed_compact_memory_annotations> --out report.json

所有门槛都是命令行参数，便于做「放宽 / 调紧」的三向验证。
"""

from __future__ import annotations

import os as _os

# 必须在 numpy / sentencepiece 之前设置：BLAS/OMP 的线程池按核数开（这台机 nproc
# 报 119，而 cgroup 只给 8 核），多进程 × 每进程上百线程会把机器压垮 —— 实测这台
# 机上一度挂着 19,088 个线程、一个 4 秒的测试跑了 484 秒。gate 全是 SentencePiece
# 和 JSON 解析，不需要任何 BLAS 并行。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 默认值。全部可由命令行覆盖 —— 这是三向验证（通过 / 放宽 / 调紧）的前提。
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(
    "/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/"
    "fixed_compact_memory_annotations"
)
DEFAULT_EXPECT_TASKS = 50
DEFAULT_EXPECT_EPISODES_PER_TASK = 200
DEFAULT_EXPECT_EPISODES = 10_000
DEFAULT_EXPECT_INTERVALS = 261_353

# 视频/帧语料的 LeRobot root。annotation 是它的 derived 层，dataloader 必须按
# (episode_index, frame_idx) 把两边联起来 —— 这份 annotation 里没有任何 state/帧数据。
DEFAULT_FRAMES_META_ROOT = Path("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos")
# `dataset.py:1334` 的硬编码 chunk 大小；chunk 是当前代码里唯一的分片单元。
DEFAULT_CHUNK_SIZE = 250

# 训练配置里硬编码的值。gate 只测不改。
# `subtask_max_len` / `prompt_max_len` **不给默认值**：这两个上限还在决策中
# （128 → 160 → 可能 192；prompt 512 → 可能 320），把任何一个候选值写死在 gate
# 里，就等于让 gate 在一个假设的上限上给出 PASS。
# 另注意 `Pi05SubtaskConfig` 的**类默认值**是 128，且被在飞的 annotations_skill /
# skillbridge 实验共用 —— 抬高只能在 MoMA 配置实例上覆盖，改类默认值会静默改掉
# 别人实验的 padding。作用域由 `--config-scope-expect` 单独断言。
DEFAULT_SUBTASK_MAX_LEN = None
DEFAULT_PROMPT_MAX_LEN = None
# tokenize 时 state 的**真实**维度是 23，不是模型的 padded 宽度 32：
#   `b1k_policy.py:47 extract_state_from_proprio` = base_qvel 3 + trunk_qpos 4
#   + arm_left 7 + 左夹爪 2→求和成 1 + arm_right 7 + 右夹爪 1 = 23
#   （`B1kOutputs.action_dim = 23`，`b1k_policy.py:196`）
# 而 `PadStatesAndActions(model_config.action_dim=32)` 排在 `TokenizeSubtaskInputs`
# **之后**（`data_config.py:322` 在 `:336` 前），`B1kInputs` 里也没有任何 `pad_to_dim`
# ⇒ 那 9 个 padding 维永远不会进 `State:` 文本。
# 用 32 会把 prompt 预算高估（实测 190 vs 154）。32 仍可作为保守上界显式传入。
DEFAULT_ACTION_DIM = 23  # pi05_subtask_config.py:32

# 目标拓扑：4 节点 × 8 卡；num_workers 从训练配置实测（pretrain_config.py:411）。
DEFAULT_WORLD_SIZE = 32
DEFAULT_NUM_WORKERS = 16

# 文档 3.4.5 记录的 fixed compact Memory 分布，用于观察漂移（默认只告警）。
DEFAULT_COMPACT_P99 = 32
DEFAULT_COMPACT_MAX = 49

ALLOWED_TRANSITIONS = (
    "normal",
    "intra_primitive_skill_bridge",
    "inter_primitive_bridge",
    "repaired_parent_primitive",
)

EXPECTED_SCHEMA_VERSION = "b1k_fixed_compact_memory_v1"
EXPECTED_INTERVAL_CONVENTION = "half-open [start,end)"

# episode 首区间的固定 initial Memory（doc 3.4.2）。运行时会核对一致性。
INITIAL_PREVIOUS_MEMORY = "No task steps have been completed; prepare to begin the task."

# 五行 target 的字段顺序，doc 3.4.3 / 3.4.4 固定。
TARGET_LINE_SPEC = (
    ("Memory", "fixed_compact_memory"),
    ("Primitive", "current_primitive"),
    ("Skill", "current_skill"),
    ("Next skill", "next_skill"),
    ("Next primitive", "next_primitive"),
)
ACTION_QUERY_SUFFIX = "\nAction Query:"

# --- oracle 泄漏检测 -------------------------------------------------------
# P0 那批样本的 `State:` 是 `Frame N of M` 帧计数器；`of M`（episode 总长）只有
# 回放 GT 才有。这类泄漏会让训练曲线更好看、只在闭环失效，所以必须在数据上判死。
LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("frame_counter", re.compile(r"\bframe\s+\d+\s+of\s+\d+\b", re.IGNORECASE)),
    ("frame_word", re.compile(r"\bframe\s+(?:index|idx|\d)", re.IGNORECASE)),
    ("of_total", re.compile(r"\bof\s+\d{2,}\b", re.IGNORECASE)),
    ("annotation_index", re.compile(r"annotation[_\s-]?index", re.IGNORECASE)),
    ("segments_field", re.compile(r"\bsegments\b", re.IGNORECASE)),
    ("frame_duration", re.compile(r"frame[_\s-]?duration", re.IGNORECASE)),
    ("memory_idx", re.compile(r"memory[_\s-]?idx", re.IGNORECASE)),
    ("memory_signature", re.compile(r"memory[_\s-]?signature", re.IGNORECASE)),
    ("task_id", re.compile(r"\btask-\d{3,}\b", re.IGNORECASE)),
    ("episode_id", re.compile(r"\bepisode[_\s-]?\d{5,}\b", re.IGNORECASE)),
    ("episode_length", re.compile(r"\b(?:episode[_\s-]?length|total[_\s-]?frames|num[_\s-]?frames|task[_\s-]?duration)\b", re.IGNORECASE)),
)

# 原始 object handle 形如 `radio_89` / `coffee_table_koagbh_0`：至少一个下划线，
# 全小写数字。裸词（如 `robot`）不算 handle，否则会把普通英文判成泄漏。
HANDLE_SHAPE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
# 归一化（下划线换空格）之后还能当判据的，只有**带数字后缀**的 handle：
# `radio_89` -> "radio 89"、`coffee_table_koagbh_0` -> "coffee table koagbh 0"，自然语言不会这么写。
# 而 `grated_cheese` -> "grated cheese" 是一句普通英文，生产 task 文本里就有
# （实测全语料 432 个 handle 里只有它 1 个不带数字后缀，却造成 12,833 条误报）。
# 所以：原串对所有 handle 判，归一化形态只对带数字后缀的判。
HANDLE_HAS_ID_SUFFIX = re.compile(r"_\d+$")

# 组合型 primitive 的 fallback 文案（判据由 worker 7630833c-749 提供并经本 gate
# 全量复算：35,945 个区间命中、1,387 个 episode 受影响，与其读数逐字一致）。
# 判据是**子串**，因为 fallback 串会被包进 `transitioning from X to Y` 里。
FALLBACK_PHRASES: tuple[str, ...] = (
    "pick up from + place on next to",
    "pick up from + place in next to",
    "pick up from + chop + place on next to",
    "pick up from + place on",
    "pick up from + place in",
    "pick up from + chop",
    "pick up from + pour",
    "unknown primitive",
)
# 完备性断言的作用域只能是 primitive 文案字段。扩到 model_target_text 会有 3,794
# 条假警报 —— 那里的 ` + ` 是 LLM 摘要在当「和」用（"Collected trash can + 2 soda cans"）。
FALLBACK_SCOPE_FIELDS: tuple[str, ...] = ("current_primitive", "next_primitive")

# 负对照哨兵：必然为 0。任何一次非 0 都说明扫描/匹配本身坏了。
SENTINEL_TEXT = "zzz_no_such_string_qqq"
SENTINEL_RE = re.compile(r"zzz[0-9]{9}qqq")


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Check:
    """一条检查。`hard=True` 的失败会让 gate 非零退出。"""

    key: str
    hard: bool
    passed: bool
    detail: str
    measured: Any = None
    threshold: Any = None
    evidence: list[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


class Ledger:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        key: str,
        *,
        hard: bool,
        passed: bool,
        detail: str,
        measured: Any = None,
        threshold: Any = None,
        evidence: list[str] | None = None,
    ) -> Check:
        # 同一个 key 只能登记一次。源码里有 6 个 key 出现在互斥分支里（if/else、
        # try/except），正常运行时每个只会走到一支；真出现两次说明分支写漏了或者
        # 补丁把代码块复制了一份，那会让同一条检查有两个结论、后写的覆盖前面的
        # 判定却都留在报告里。宁可当场炸掉。
        if any(x.key == key for x in self.checks):
            raise RuntimeError(f"check key registered twice: {key}（互斥分支同时执行了？）")
        c = Check(key, hard, bool(passed), detail, measured, threshold, list(evidence or []))
        self.checks.append(c)
        return c

    @property
    def hard_failures(self) -> list[Check]:
        return [c for c in self.checks if c.hard and not c.passed]

    @property
    def soft_failures(self) -> list[Check]:
        return [c for c in self.checks if not c.hard and not c.passed]


# ---------------------------------------------------------------------------
# tokenizer 装载（每个 worker 进程一次）
# ---------------------------------------------------------------------------
_TOK: Any = None
_STATE_STR: str = ""
_OPTS: dict[str, Any] = {}


def _clean(text: str) -> str:
    """与 `tokenizer.py:503 / :533` 逐字一致的清洗。"""
    return text.strip().replace("_", " ").replace("\n", " ")


def build_tokenizer(prompt_max_len: int, subtask_max_len: int):
    from openpi.models.tokenizer import SubtaskTokenizer

    return SubtaskTokenizer(prompt_max_len=prompt_max_len, subtask_max_len=subtask_max_len)


def tokenizer_fingerprint(tok) -> dict[str, Any]:
    """记录「量的是哪把尺子」：模型文件、md5、vocab、库版本。"""
    import sentencepiece

    from openpi.shared import download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    return {
        "sentencepiece_model_path": str(path),
        "sentencepiece_model_md5": md5,
        "vocab_size": int(tok.vocab_size),
        "sentencepiece_lib_version": getattr(sentencepiece, "__version__", "unknown"),
        "sentencepiece_lib_file": sentencepiece.__file__,
        "bos_id": int(tok._tokenizer.bos_id()),
        "eos_id": int(tok._tokenizer.eos_id()),
        "pad_id": int(tok._tokenizer.pad_id()),
        "python": sys.version.split()[0],
    }


def worst_case_state_string(tok, action_dim: int) -> tuple[str, int, dict[str, int]]:
    """选一个 token 数最大的 `State:` 串。

    数据集里没有任何 state 数值（这是一份纯标注数据集），所以 prompt 预算只能用
    最坏情况的 state 来给上界。候选覆盖 digitize 的取值边界：-1（越下界）、0、255
    （越上界）以及随机分布。
    """
    sp = tok._tokenizer
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    rng = np.random.default_rng(0)
    candidates = {
        "all_low_out_of_range": np.full(action_dim, -5.0),
        "all_minus_one": np.full(action_dim, -1.0),
        "all_zero": np.zeros(action_dim),
        "all_high_out_of_range": np.full(action_dim, 5.0),
        "uniform_random": rng.uniform(-1.0, 1.0, action_dim),
    }
    lengths: dict[str, int] = {}
    best_str, best_len = "", -1
    for name, state in candidates.items():
        digitized = np.digitize(state, bins=bins) - 1
        s = " ".join(map(str, digitized))
        n = len(sp.encode(s))
        lengths[name] = n
        if n > best_len:
            best_len, best_str = n, s
    return best_str, best_len, lengths


def _init_worker(opts: dict[str, Any]) -> None:
    global _TOK, _STATE_STR, _OPTS
    _OPTS = opts
    _TOK = build_tokenizer(opts["prompt_max_len"], opts["subtask_max_len"])
    _STATE_STR = opts["state_str"]


# ---------------------------------------------------------------------------
# 单 episode 扫描
# ---------------------------------------------------------------------------
MAX_EVIDENCE_PER_KIND = 4


def _collect_handles(episode: dict) -> set[str]:
    handles: set[str] = set()

    def flat(x):
        if isinstance(x, str):
            handles.add(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                flat(y)

    for key in ("skill_annotation", "primitive_annotation"):
        for seg in episode.get(key, []) or []:
            flat(seg.get("object_id"))
            flat(seg.get("manipulating_object_id"))
    return {h for h in handles if HANDLE_SHAPE.match(h)}


def build_planner_target_text(row: dict) -> str:
    """doc 3.4.3 的 `planner_target_text`：五行，不含 `Action Query:`。"""
    return "\n".join(f"{label}: {row[field]}" for label, field in TARGET_LINE_SPEC)


def build_prefix_text(task_instruction: str, state_str: str, previous_memory: str) -> str:
    """doc 3.4.3 的 prefix。

    清洗口径复制 `tokenizer.py:503`：只清洗自由文本槽位，模板里的 `\\n` 保留
    （`tokenize_prompt` 也是先清洗 prompt 再拼模板，`\\n` 进入 encode）。
    """
    return (
        f"Task: {_clean(task_instruction)}, State: {state_str};\n"
        f"Previous memory: {_clean(previous_memory)}"
    )


def scan_episode(path_str: str) -> dict[str, Any]:
    """扫一个 episode，返回聚合量 + 有界的证据样本。"""
    global _TOK, _STATE_STR, _OPTS
    tok = _TOK
    sp = tok._tokenizer
    subtask_max_len = _OPTS["subtask_max_len"]
    prompt_max_len = _OPTS["prompt_max_len"]
    deep = _OPTS["deep_paths"]
    frames_per_episode = _OPTS["frames_per_episode"]
    seed = _OPTS["seed"]

    path = Path(path_str)
    out: dict[str, Any] = {
        "path": path_str,
        "readable": False,
        "n_intervals": 0,
        "target_lengths": [],
        "model_target_lengths": [],
        "prompt_lengths": [],
        "compact_lengths": [],
        "violations": Counter(),
        "evidence": {},
        "transition_counts": Counter(),
        "sentinel_text_hits": 0,
        "sentinel_re_hits": 0,
        "poscontrol_memory_label": 0,
        "poscontrol_nonzero_len": 0,
        "deep_checked": False,
        "deep_frames": 0,
        "leak_kinds": Counter(),
        "handles_without_suffix": set(),
        "fallback_intervals": 0,
        "fallback_model_target": 0,
        "fallback_episode": 0,
        "fallback_by_task": Counter(),
        "plus_without_known_fallback": 0,
        "ep_index": None,
        "video_length": None,
        "uncovered_head": 0,
        "uncovered_tail": 0,
        "overhang": 0,
        "n_chunks": 0,
        "dead_chunks": 0,
        "has_uncovered": 0,
        "has_dead_chunk": 0,
        "join_missing_in_meta": 0,
        "overhang_episode": 0,
        "video_range_probes": 0,
        "video_range_misses": 0,
        "live_chunks": 0,
        "partial_chunks": 0,
        "clamped_empty": 0,
        "annotated_frames": 0,
        "annotated_frames_outside_live_chunks": 0,
        "annotated_frames_lost_if_partial_dropped": 0,
        "clamped_probes": 0,
        "clamped_misses": 0,
        "chunks_without_aligned_anchor": 0,
        "prod_lengths": [],
        "prompt_lengths_other_source": [],
        "path_deltas": Counter(),
        "len_by_transition": {},
        "overhang_by_transition": Counter(),
    }

    def note(kind: str, msg: str) -> None:
        out["violations"][kind] += 1
        bucket = out["evidence"].setdefault(kind, [])
        if len(bucket) < MAX_EVIDENCE_PER_KIND:
            bucket.append(msg)

    try:
        episode = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        note("unreadable_episode", f"{path_str}: {type(exc).__name__}: {exc}")
        return out
    out["readable"] = True

    if episode.get("memory_schema_version") != EXPECTED_SCHEMA_VERSION:
        note("schema_version", f"{path_str}: {episode.get('memory_schema_version')!r}")
    if episode.get("memory_interval_convention") != EXPECTED_INTERVAL_CONVENTION:
        note("interval_convention", f"{path_str}: {episode.get('memory_interval_convention')!r}")

    rows = episode.get("memory_annotation") or []
    out["n_intervals"] = len(rows)
    if not rows:
        note("empty_memory_annotation", path_str)
        return out

    ann_task = episode.get("task_name") or ""
    try:
        _ep_idx = int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        _ep_idx = None
    lerobot_task = (_OPTS.get("episode_tasks") or {}).get(_ep_idx, "")
    # 生产走 prompt_from_task=True，`Task:` 段填的是 LeRobot 的 tasks[0]，不是 annotation 的
    # task_name。两者差很多（'turning on radio' vs 完整句子），用后者会低估 prompt 预算。
    if _OPTS.get("task_text_source") == "lerobot":
        task_instruction = lerobot_task
        if not task_instruction:
            # 不回退到 annotation：回退会让「部分 episode 用错口径」变成一个只在
            # note 里的软信息，而 prompt 预算照常出数。缺就是缺。
            note("missing_lerobot_task_text", f"{path_str}: episode_index={_ep_idx}")
    else:
        task_instruction = ann_task
    if not ann_task:
        note("missing_task_name", path_str)
    handles = _collect_handles(episode)
    # 无数字后缀的 handle 归一化后可能是普通英文（如 grated_cheese -> "grated cheese"），
    # 对这类只按原串判。把它们报出来，免得这条豁免变成一个没人看见的洞。
    out["handles_without_suffix"] = sorted(h for h in handles if not HANDLE_HAS_ID_SUFFIX.search(h))
    valid_duration = (episode.get("meta_data") or {}).get("valid_duration")

    starts: list[int] = []
    for i, row in enumerate(rows):
        # --- 区间结构 -----------------------------------------------------
        fd = row.get("frame_duration")
        if not (isinstance(fd, (list, tuple)) and len(fd) == 2):
            note("frame_duration_shape", f"{path_str}#{i}: {fd!r}")
            continue
        s, e = int(fd[0]), int(fd[1])
        starts.append(s)
        if not s < e:
            note("interval_inverted", f"{path_str}#{i}: [{s},{e})")
        if row.get("memory_idx") != i:
            note("memory_idx_not_sequential", f"{path_str}#{i}: memory_idx={row.get('memory_idx')!r}")
        if i + 1 < len(rows):
            nxt = rows[i + 1].get("frame_duration")
            if isinstance(nxt, (list, tuple)) and len(nxt) == 2 and int(nxt[0]) != e:
                note("interval_not_contiguous", f"{path_str}#{i}: end={e} next_start={nxt[0]}")

        # --- previous / current Memory 时序（doc 3.2 的 t-1 -> t）---------
        prev_mem = row.get("previous_fixed_compact_memory")
        if i == 0:
            if prev_mem != INITIAL_PREVIOUS_MEMORY:
                note("initial_previous_memory", f"{path_str}#0: {prev_mem!r}")
        elif prev_mem != rows[i - 1].get("fixed_compact_memory"):
            note("previous_memory_chain", f"{path_str}#{i}")

        # --- transition / bridge 标签 -------------------------------------
        tt = row.get("transition_type")
        out["transition_counts"][str(tt)] += 1
        if tt not in ALLOWED_TRANSITIONS:
            note("transition_type_unknown", f"{path_str}#{i}: {tt!r}")
        if tt == "intra_primitive_skill_bridge":
            if i == 0:
                note("bridge_at_index_zero", f"{path_str}#{i}: {tt}")
            else:
                if rows[i - 1].get("current_primitive") != row.get("current_primitive"):
                    note("intra_bridge_primitive_changed", f"{path_str}#{i}")
                if rows[i - 1].get("current_skill") == row.get("current_skill"):
                    note("intra_bridge_skill_unchanged", f"{path_str}#{i}")
        elif tt == "inter_primitive_bridge":
            if i == 0:
                note("bridge_at_index_zero", f"{path_str}#{i}: {tt}")
            elif rows[i - 1].get("current_primitive") == row.get("current_primitive"):
                note("inter_bridge_primitive_unchanged", f"{path_str}#{i}")

        # --- current_memory 子结构与顶层字段一致性 -------------------------
        cm = row.get("current_memory") or {}
        for k_top, k_cm in (
            ("current_primitive", "primitive"),
            ("current_skill", "skill"),
            ("next_skill", "next_skill"),
            ("next_primitive", "next_primitive"),
            ("transition_type", "transition_type"),
        ):
            if cm.get(k_cm) != row.get(k_top):
                note("current_memory_subdict_mismatch", f"{path_str}#{i}: {k_top}")

        # --- 文本构造 -------------------------------------------------------
        try:
            planner = build_planner_target_text(row)
        except KeyError as exc:
            note("missing_target_field", f"{path_str}#{i}: {exc}")
            continue
        model_target = row.get("model_target_text", "")

        # `Action Query:` 是序列标记，不能当文本再编码一次（doc 3.4.3）。
        if "Action Query" in planner:
            note("action_query_in_ce_text", f"{path_str}#{i}")
        if model_target != planner + ACTION_QUERY_SUFFIX:
            note("model_target_reconstruction", f"{path_str}#{i}")

        prefix = build_prefix_text(task_instruction, _STATE_STR, row.get("previous_fixed_compact_memory") or "")
        # 泄漏检测只作用于文本槽位；state 段本身是纯整数，单独校验。
        text_slots = planner + "\n" + _clean(task_instruction) + "\n" + _clean(row.get("previous_fixed_compact_memory") or "")

        for kind, pat in LEAK_PATTERNS:
            if pat.search(text_slots):
                out["leak_kinds"][kind] += 1
                note("oracle_leak", f"{path_str}#{i}: {kind}")
                break
        normalized = text_slots.replace("_", " ")
        for h in handles:
            hit = h in text_slots
            if not hit and HANDLE_HAS_ID_SUFFIX.search(h):
                hit = h.replace("_", " ") in normalized
            if hit:
                out["leak_kinds"]["raw_object_handle"] += 1
                note("oracle_leak", f"{path_str}#{i}: raw_object_handle {h}")
                break

        # --- 组合型 primitive fallback 文案 ---------------------------------
        prim_blob = " | ".join(str(row.get(f) or "") for f in FALLBACK_SCOPE_FIELDS)
        prim_hit = any(f in prim_blob for f in FALLBACK_PHRASES)
        stack_blob = " ".join(str(x) for x in (row.get("completed_primitive_stack") or []))
        mtt_hit = any(f in model_target for f in FALLBACK_PHRASES)
        if prim_hit or mtt_hit or any(f in stack_blob for f in FALLBACK_PHRASES):
            out["fallback_intervals"] += 1
            out["fallback_by_task"][path.parent.name] += 1
        if mtt_hit:
            out["fallback_model_target"] += 1
        # 完备性：primitive 文案里出现 ` + ` 却不属于已知的 8 个串 ⇒ 分类表漏了一种
        if " + " in prim_blob and not prim_hit:
            out["plus_without_known_fallback"] += 1
            note("fallback_taxonomy_incomplete", f"{path_str}#{i}: {prim_blob[:120]!r}")

        if SENTINEL_TEXT in text_slots:
            out["sentinel_text_hits"] += 1
        if SENTINEL_RE.search(text_slots):
            out["sentinel_re_hits"] += 1
        if "Memory:" in planner:
            out["poscontrol_memory_label"] += 1

        # --- token 长度（训练侧 tokenizer）---------------------------------
        tokens, mask, ar_mask, loss_mask = tok.tokenize_subtask(planner)
        n_valid = int(mask.sum())
        true_len = 1 + len(sp.encode(_clean(planner))) + 1  # BOS + text + EOS，未截断
        out["target_lengths"].append(true_len)
        if true_len > 0:
            out["poscontrol_nonzero_len"] += 1
        if true_len > subtask_max_len:
            note("target_over_subtask_max_len", f"{path_str}#{i}: {true_len}")

        # EOS 是否还在有效 mask 内（截断会把它吃掉）
        eos_id = sp.eos_id()
        eos_positions = np.flatnonzero(tokens[:n_valid] == eos_id)
        if eos_positions.size == 0:
            note("eos_lost", f"{path_str}#{i}: len={true_len}")
        else:
            last_eos = int(eos_positions[-1])
            if not bool(loss_mask[last_eos]):
                note("eos_not_supervised", f"{path_str}#{i}")
            if last_eos != n_valid - 1:
                note("eos_not_last_valid", f"{path_str}#{i}: eos@{last_eos} n_valid={n_valid}")

        # loss mask 契约：BOS 不监督、pad 不监督、其余全监督
        if bool(loss_mask[0]):
            note("bos_supervised", f"{path_str}#{i}")
        if int(tokens[0]) != sp.bos_id():
            note("bos_missing", f"{path_str}#{i}")
        expected_loss = mask.copy()
        expected_loss[0] = False
        if not np.array_equal(loss_mask, expected_loss):
            note("loss_mask_contract", f"{path_str}#{i}")
        if int(np.min(ar_mask[:n_valid])) != 1 or int(np.max(ar_mask[n_valid:], initial=0)) != 0:
            note("ar_mask_contract", f"{path_str}#{i}")

        # 生产口径：transforms.py:467-473 在 item 带 memory_text 时走 tokenize_memory，
        # 它刻意不把 `\n` 压成空格（tokenizer.py:650-653），所以五行 target 多 4 个换行
        # token。文档 3.4.4 写的是 tokenize_subtask ⇒ 两个口径都要量、都要报。
        try:
            prod_len = tok.memory_token_length(planner)
        except Exception as exc:  # MemoryTextError 等
            prod_len = None
            note("memory_codec_rejects_text", f"{path_str}#{i}: {type(exc).__name__}: {str(exc)[:120]}")
        if prod_len is not None:
            out["prod_lengths"].append(prod_len)
            out["path_deltas"][prod_len - true_len] += 1
            b = out["len_by_transition"].setdefault(str(row.get("transition_type")),
                                                    {"n": 0, "prod_max": 0, "doc_max": 0})
            b["n"] += 1
            b["prod_max"] = max(b["prod_max"], prod_len)
            b["doc_max"] = max(b["doc_max"], true_len)
            if prod_len > subtask_max_len:
                note("prod_target_over_subtask_max_len", f"{path_str}#{i}: {prod_len}")
        # 「两条口径恒差 4」只对不含下划线的文本成立：tokenize_subtask 会把 `_` 换成
        # 空格、tokenize_memory 不会。数据侧正在用 API 重新生成 phrase 文案，新文本一旦
        # 带下划线，+4 就失效，而失效方式是静默的。所以这里直接判死下划线。
        if "_" in planner:
            note("underscore_in_target_text", f"{path_str}#{i}: {planner[:120]!r}")

        # model_target_text 也量一遍，便于与 il_lib 的 104 直接对齐
        out["model_target_lengths"].append(1 + len(sp.encode(_clean(model_target))) + 1)

        # fixed compact Memory 字段本身（观察漂移，不单独截断）
        out["compact_lengths"].append(1 + len(sp.encode(_clean(row.get("fixed_compact_memory") or ""))) + 1)

        # --- prompt 预算 -----------------------------------------------------
        # prompt 从右截断，而 `Previous memory:` 在末尾 ⇒ 溢出时被吃掉的正好是
        # memory 文本。这里量的是未截断的真长度。
        prompt_len = len(sp.encode(prefix, add_bos=True))
        out["prompt_lengths"].append(prompt_len)
        # 另一口径同时量一份：两个数并排报，读者不会以为数据变了
        other_task = ann_task if _OPTS.get("task_text_source") == "lerobot" else lerobot_task
        out["prompt_lengths_other_source"].append(
            len(sp.encode(build_prefix_text(other_task, _STATE_STR,
                                            row.get("previous_fixed_compact_memory") or ""),
                          add_bos=True)))
        if prompt_len > prompt_max_len:
            note("prompt_over_max_len", f"{path_str}#{i}: {prompt_len}")

        # state 段必须是 action_dim 个 [0,255] 整数，而不是帧计数器
        state_field = prefix.split(", State: ", 1)[1].split(";\n", 1)[0]
        parts = state_field.split()
        if len(parts) != _OPTS["action_dim"]:
            note("state_field_arity", f"{path_str}#{i}: {len(parts)}")
        else:
            try:
                vals = [int(x) for x in parts]
            except ValueError:
                note("state_field_not_integer", f"{path_str}#{i}: {state_field[:60]!r}")
            else:
                if not all(-1 <= v <= 255 for v in vals):
                    note("state_field_range", f"{path_str}#{i}")

    # --- 覆盖范围与 valid_duration 对齐 ------------------------------------
    if valid_duration and len(rows) and starts:
        first_start = rows[0]["frame_duration"][0]
        last_end = rows[-1]["frame_duration"][1]
        if int(first_start) != int(valid_duration[0]):
            note("coverage_start_mismatch", f"{path_str}: {first_start} vs {valid_duration[0]}")
        if int(last_end) != int(valid_duration[1]):
            note("coverage_end_mismatch", f"{path_str}: {last_end} vs {valid_duration[1]}")

    # --- 与视频帧长度的联表：未命中区间的帧 --------------------------------
    # `dataset.py:1347` 按 `range(0, L, 250)` 切 chunk，L 取自 meta/episodes.jsonl 的
    # `length`（整段视频），不是 valid_duration。所以落在标注覆盖之外的 anchor 会让
    # doc 3.4.3 的 `memory_at()` 抛 IndexError —— 这正是文档要求的「未命中区间」。
    ep_lengths = _OPTS.get("episode_lengths")
    if ep_lengths is not None and rows:
        try:
            ep_index = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            ep_index = None
        out["ep_index"] = ep_index
        L = ep_lengths.get(ep_index)
        if L is None:
            out["join_missing_in_meta"] = 1
            note("episode_not_in_frames_meta", f"{path_str}: episode_index={ep_index}")
        else:
            out["video_length"] = L
            cov_s = int(rows[0]["frame_duration"][0])
            cov_e = int(rows[-1]["frame_duration"][1])
            out["uncovered_head"] = max(0, cov_s)
            out["uncovered_tail"] = max(0, L - cov_e)
            out["overhang"] = max(0, cov_e - L)
            cs_size = _OPTS["chunk_size"]
            out["n_chunks"] = -(-L // cs_size)
            # live_e 必须与 L 取 min：83 个 episode 的标注伸出了视频结尾。
            live_e = min(cov_e, L)
            out["annotated_frames"] = max(0, live_e - cov_s)
            dead = 0
            covered_by_live = 0
            lost_if_partial_dropped = 0
            for cs in range(0, L, cs_size):
                ce = min(cs + cs_size, L)
                if ce <= cov_s or cs >= live_e:      # 与 [cov_s, live_e) 交集为空
                    dead += 1
                    continue
                out["live_chunks"] += 1
                lo = max(cs, cov_s)
                hi = min(ce, live_e)
                if hi <= lo:
                    out["clamped_empty"] += 1
                covered_by_live += max(0, hi - lo)
                if cs < cov_s or ce > live_e:        # 跨边界：必须保留 + 夹取
                    out["partial_chunks"] += 1
                    lost_if_partial_dropped += max(0, hi - lo)
            out["dead_chunks"] = dead
            out["annotated_frames_outside_live_chunks"] = max(0, out["annotated_frames"] - covered_by_live)
            out["annotated_frames_lost_if_partial_dropped"] = lost_if_partial_dropped
            out["has_uncovered"] = 1 if (out["uncovered_head"] or out["uncovered_tail"]) else 0
            out["has_dead_chunk"] = 1 if dead else 0
            if dead:
                note("dead_chunk", f"{path_str}: L={L} coverage=[{cov_s},{cov_e}) dead_chunks={dead}")
            if out["overhang"]:
                out["overhang_episode"] = 1
                # 按 transition_type 分组：越界若出现在 normal 而不是 bridge，
                # 那是结构性变化而不是 bridge 长尾，必须一眼能区分。
                out["overhang_by_transition"][str(rows[-1].get("transition_type"))] += 1
                note("annotation_overhang", f"{path_str}: L={L} ann_end={cov_e} overhang={out['overhang']}")

    # --- 深度抽样：frame -> interval 命中（doc 3.4.3 的 bisect 查找）--------
    if path_str in deep and len(starts) == len(rows):
        out["deep_checked"] = True
        rng = random.Random(f"{seed}:{path_str}")
        lo = rows[0]["frame_duration"][0]
        hi = rows[-1]["frame_duration"][1]
        probes: list[int] = []
        for r in rows:
            s, e = r["frame_duration"]
            probes.extend([s, e - 1])
            if e - s > 2:
                probes.append(s + 1)
        for _ in range(frames_per_episode):
            probes.append(rng.randrange(lo, hi))
        # 额外在整段视频 [0, L) 上采样：chunk 的 anchor 就是从这个范围来的
        vlen = out.get("video_length")
        if vlen:
            cs_size = _OPTS["chunk_size"]
            cov_s = int(rows[0]["frame_duration"][0])
            live_e = min(int(rows[-1]["frame_duration"][1]), vlen)
            for _ in range(frames_per_episode):
                f = rng.randrange(0, vlen)
                out["video_range_probes"] += 1
                j = bisect_right(starts, f) - 1
                if j < 0:
                    out["video_range_misses"] += 1
                    continue
                vs, ve = rows[j]["frame_duration"]
                if not (vs <= f < ve):
                    out["video_range_misses"] += 1
            # 剔除 dead chunk + anchor 夹取之后：探测的必须是**stride 对齐的真实 anchor**，
            # 不是任意帧。任意帧只能证明「覆盖范围内任取一帧都命中」，证明不了
            # 「训练实际会取的那些帧都命中」。
            a_stride = _OPTS["anchor_stride"]
            a_offset = _OPTS["anchor_offset"]
            chunk_starts = list(range(0, vlen, cs_size))
            picks = chunk_starts if len(chunk_starts) <= 24 else rng.sample(chunk_starts, 24)
            for cs in picks:
                ce = min(cs + cs_size, vlen)
                if ce <= cov_s or cs >= live_e:
                    continue                      # dead chunk：已被剔除，不采样
                lo, hi = max(cs, cov_s), min(ce, live_e)
                if hi <= lo:
                    continue
                # 复刻 dataset.py:107-113 的对齐算法：episode-local 起点是 cs
                first = lo + ((a_offset - lo) % a_stride)
                if first >= hi:
                    # 该 chunk 的夹取范围内没有任何对齐 anchor。训练侧
                    # `_aligned_streaming_chunk_start` 会对它返回 None（不是 raise），
                    # 由 `_select_aligned_streaming_chunk` 在**所有** chunk 都没有时才 raise。
                    out["chunks_without_aligned_anchor"] += 1
                    note("chunk_without_aligned_anchor",
                         f"{path_str}: chunk=[{cs},{ce}) clamped=[{lo},{hi}) stride={a_stride} offset={a_offset}")
                    continue
                cand = list(range(first, hi, a_stride))
                probe = cand if len(cand) <= 3 else [cand[0], cand[-1], rng.choice(cand)]
                for f in probe:
                    out["clamped_probes"] += 1
                    j = bisect_right(starts, f) - 1
                    if j < 0 or not (rows[j]["frame_duration"][0] <= f < rows[j]["frame_duration"][1]):
                        out["clamped_misses"] += 1
                        note("clamped_range_miss",
                             f"{path_str}: anchor={f} chunk=[{cs},{ce}) stride={a_stride}")
        for frame_idx in probes:
            out["deep_frames"] += 1
            i = bisect_right(starts, frame_idx) - 1
            if i < 0:
                note("frame_before_valid_duration", f"{path_str}: frame={frame_idx}")
                continue
            s, e = rows[i]["frame_duration"]
            if not (s <= frame_idx < e):
                note("frame_not_covered", f"{path_str}: frame={frame_idx} -> [{s},{e})")
                continue
            # 与线性扫描交叉核对，证明 bisect 与半开区间语义一致
            linear = [j for j, r in enumerate(rows) if r["frame_duration"][0] <= frame_idx < r["frame_duration"][1]]
            if linear != [i]:
                note("bisect_linear_disagree", f"{path_str}: frame={frame_idx} bisect={i} linear={linear}")

    out["path_deltas"] = dict(out["path_deltas"])
    out["overhang_by_transition"] = dict(out["overhang_by_transition"])
    out["fallback_episode"] = 1 if out["fallback_intervals"] else 0
    out["fallback_by_task"] = dict(out["fallback_by_task"])
    out["violations"] = dict(out["violations"])
    out["transition_counts"] = dict(out["transition_counts"])
    out["leak_kinds"] = dict(out["leak_kinds"])
    out["handles_without_suffix"] = list(out["handles_without_suffix"])
    return out


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------
def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return -1
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def describe(values: list[int]) -> dict[str, int]:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {
        "n": len(v),
        "min": v[0],
        "p50": percentile(v, 0.50),
        "p90": percentile(v, 0.90),
        "p95": percentile(v, 0.95),
        "p99": percentile(v, 0.99),
        "p999": percentile(v, 0.999),
        "max": v[-1],
    }


def data_fingerprint(root: Path, files: list[Path]) -> dict[str, Any]:
    """记录「量的是哪个版本的数据」。数据会被另一个 worker 重新生成。"""
    manifest = root / "manifest.json"
    fp: dict[str, Any] = {
        "data_root": str(root),
        "n_files": len(files),
        "manifest_exists": manifest.exists(),
    }
    if manifest.exists():
        st = manifest.stat()
        fp["manifest_mtime"] = int(st.st_mtime)
        fp["manifest_mtime_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
        fp["manifest_md5"] = hashlib.md5(manifest.read_bytes()).hexdigest()
        try:
            m = json.loads(manifest.read_text())
            fp["manifest_episodes"] = m.get("episodes")
            fp["manifest_intervals"] = (m.get("scan_stats") or {}).get("intervals")
            fp["manifest_schema_version"] = m.get("schema_version")
            fp["manifest_unresolved_phrase_templates"] = (m.get("scan_stats") or {}).get(
                "source_unresolved_phrase_templates"
            )
        except Exception:  # noqa: BLE001
            fp["manifest_parse_error"] = True
    # 跨偏移抽 16 个文件做 md5 指纹（不是 head：按等距取样）
    if files:
        step = max(1, len(files) // 16)
        probe = files[::step][:16]
        fp["probe_files"] = [
            {"path": str(p), "md5": hashlib.md5(p.read_bytes()).hexdigest(), "size": p.stat().st_size}
            for p in probe
        ]
        fp["probe_digest"] = hashlib.md5(
            "".join(x["md5"] for x in fp["probe_files"]).encode()
        ).hexdigest()
    return fp


# ---------------------------------------------------------------------------
# 分片
# ---------------------------------------------------------------------------
def sharding_report(
    *,
    world_size: int,
    num_workers: int,
    unit_counts: dict[str, int],
) -> dict[str, Any]:
    """按 `dataset.py:1014-1025` 的分片算法算每个 global worker 拿到多少单位。

    `indices = range(global_worker_id, len(chunks), global_num_workers)`
    ⇒ global_worker_id >= len(chunks) 的 worker 拿到空集，随后落进 fallback
    `range(worker_id, len(chunks), num_workers)` —— 该 fallback **不含 rank**，
    于是不同 rank 的同号 worker 拿到逐元素相同的数据。这就是 rank-blind 失效。
    """
    gnw = max(1, world_size * num_workers)
    out: dict[str, Any] = {
        "world_size": world_size,
        "num_workers_per_rank": num_workers,
        "global_num_workers": gnw,
        "units": {},
    }
    for name, n in unit_counts.items():
        per_worker = [len(range(gid, n, gnw)) for gid in range(gnw)]
        empty = sum(1 for x in per_worker if x == 0)
        out["units"][name] = {
            "n_units": n,
            "min_per_global_worker": min(per_worker),
            "max_per_global_worker": max(per_worker),
            "empty_global_workers": empty,
            "margin_x": round(n / gnw, 4),
            "fallback_triggered": empty > 0,
            "rank_blind_duplicate_ranks": (world_size if empty > 0 else 0),
        }
    return out


# ---------------------------------------------------------------------------
# 自检（正对照）：每个检测器都要证明「该命中时会命中」
# ---------------------------------------------------------------------------
def run_self_test(args: argparse.Namespace) -> dict[str, Any]:
    """用构造的坏样本证明检测器不是恒真。

    做法：在临时目录里造一个最小 data root，逐项注入一种缺陷，跑真实的
    `scan_episode`，断言对应的 violation key 出现。
    """
    import tempfile

    # 自检不判生产上限，只判「检测器该点亮时会不会点亮」，所以在没显式传值时用
    # 一组具体常数（并打印出来），而不是让 None 一路漏进 tokenizer。
    subtask_max_len = args.subtask_max_len if args.subtask_max_len is not None else 128
    prompt_max_len = args.prompt_max_len if args.prompt_max_len is not None else 512
    print(f"[self-test] subtask_max_len={subtask_max_len} prompt_max_len={prompt_max_len} "
          f"action_dim={args.action_dim}"
          + ("  (未显式传入，用自检默认值)" if args.subtask_max_len is None else ""))
    args = argparse.Namespace(**{**vars(args), "subtask_max_len": subtask_max_len,
                                 "prompt_max_len": prompt_max_len})

    tok = build_tokenizer(args.prompt_max_len, args.subtask_max_len)
    state_str, _, _ = worst_case_state_string(tok, args.action_dim)

    base_row = {
        "memory_idx": 0,
        "frame_duration": [0, 100],
        "previous_fixed_compact_memory": INITIAL_PREVIOUS_MEMORY,
        "fixed_compact_memory": "No steps completed; currently picking up the radio, next step is to press it.",
        "completed_primitive_stack": [],
        "transition_type": "normal",
        "current_primitive": "pick up the radio from the coffee table",
        "current_skill": "move to the radio",
        "next_skill": "pick up the radio from the coffee table",
        "next_primitive": "press the radio",
        "current_memory": {
            "primitive": "pick up the radio from the coffee table",
            "skill": "move to the radio",
            "next_skill": "pick up the radio from the coffee table",
            "next_primitive": "press the radio",
            "summary": "No primitive has been completed.",
            "transition_type": "normal",
        },
    }

    def finish(row: dict) -> dict:
        row = json.loads(json.dumps(row))
        row["model_target_text"] = build_planner_target_text(row) + ACTION_QUERY_SUFFIX
        return row

    def episode(rows: list[dict], **over) -> dict:
        rows = [finish(r) for r in rows]
        ep = {
            "task_name": "turning on radio",
            "memory_schema_version": EXPECTED_SCHEMA_VERSION,
            "memory_interval_convention": EXPECTED_INTERVAL_CONVENTION,
            "meta_data": {"task_duration": rows[-1]["frame_duration"][1], "valid_duration": [rows[0]["frame_duration"][0], rows[-1]["frame_duration"][1]]},
            "skill_annotation": [{"object_id": [["radio_89"]], "manipulating_object_id": ["coffee_table_koagbh_0"]}],
            "primitive_annotation": [],
            "memory_annotation": rows,
        }
        ep.update(over)
        return ep

    cases: list[tuple[str, str, dict]] = []

    # 1) 干净基线：必须零 violation（负对照）
    cases.append(("clean_baseline", "", episode([dict(base_row)])))

    # 2) 区间倒置
    r = dict(base_row); r["frame_duration"] = [100, 100]
    cases.append(("interval_inverted", "interval_inverted", episode([r])))

    # 3) 区间不连续
    r0 = dict(base_row)
    r1 = dict(base_row); r1.update(memory_idx=1, frame_duration=[150, 200],
                                   previous_fixed_compact_memory=base_row["fixed_compact_memory"])
    ep = episode([r0, r1])
    ep["meta_data"]["valid_duration"] = [0, 200]
    cases.append(("interval_not_contiguous", "interval_not_contiguous", ep))

    # 4) memory_idx 不连续
    r = dict(base_row); r["memory_idx"] = 7
    cases.append(("memory_idx_not_sequential", "memory_idx_not_sequential", episode([r])))

    # 5) previous memory 链断裂
    r0 = dict(base_row)
    r1 = dict(base_row); r1.update(memory_idx=1, frame_duration=[100, 200],
                                   previous_fixed_compact_memory="something else entirely")
    ep = episode([r0, r1]); ep["meta_data"]["valid_duration"] = [0, 200]
    cases.append(("previous_memory_chain", "previous_memory_chain", ep))

    # 6) oracle 泄漏：帧计数器
    r = dict(base_row); r["fixed_compact_memory"] = "Frame 3 of 265; currently picking up the radio."
    cases.append(("oracle_leak_frame_counter", "oracle_leak", episode([r])))

    # 7) oracle 泄漏：原始 object handle
    r = dict(base_row); r["current_primitive"] = "pick up the coffee_table_koagbh_0 from the floor"
    cases.append(("oracle_leak_raw_handle", "oracle_leak", episode([r])))

    # 8) oracle 泄漏：task id
    r = dict(base_row); r["current_skill"] = "move to the radio in task-0000"
    cases.append(("oracle_leak_task_id", "oracle_leak", episode([r])))

    # 9) target 超长
    r = dict(base_row); r["fixed_compact_memory"] = ("the robot picked up the object and then placed it down again " * 20)
    cases.append(("target_over_subtask_max_len", "target_over_subtask_max_len", episode([r])))

    # 10) Action Query 混进 CE 文本
    r = dict(base_row); r["next_primitive"] = "press the radio\nAction Query:"
    cases.append(("action_query_in_ce_text", "action_query_in_ce_text", episode([r])))

    # 11) model_target_text 与五行 target 不一致
    ep = episode([dict(base_row)])
    ep["memory_annotation"][0]["model_target_text"] = "Memory: tampered\nAction Query:"
    cases.append(("model_target_reconstruction", "model_target_reconstruction", ep))

    # 12) bridge 标签与结构矛盾
    r0 = dict(base_row)
    r1 = dict(base_row); r1.update(memory_idx=1, frame_duration=[100, 200],
                                   previous_fixed_compact_memory=base_row["fixed_compact_memory"],
                                   transition_type="inter_primitive_bridge")
    r1["current_memory"] = dict(r1["current_memory"]); r1["current_memory"]["transition_type"] = "inter_primitive_bridge"
    ep = episode([r0, r1]); ep["meta_data"]["valid_duration"] = [0, 200]
    cases.append(("inter_bridge_primitive_unchanged", "inter_bridge_primitive_unchanged", ep))

    # 13) schema version 漂移
    ep = episode([dict(base_row)], memory_schema_version="b1k_something_else_v2")
    cases.append(("schema_version", "schema_version", ep))

    # 14) 覆盖范围与 valid_duration 不符
    ep = episode([dict(base_row)]); ep["meta_data"]["valid_duration"] = [0, 999]
    cases.append(("coverage_end_mismatch", "coverage_end_mismatch", ep))

    # 15) fallback 分类表不完整：primitive 文案含 ` + ` 但不属于已知 8 串
    r = dict(base_row); r["current_primitive"] = "pick up from + teleport onto"
    cases.append(("fallback_taxonomy_incomplete", "fallback_taxonomy_incomplete", episode([r])))

    # 16) 已知 fallback 串不得触发完备性告警（负对照：命中分类表 ⇒ 不算 incomplete）
    r = dict(base_row); r["current_primitive"] = "pick up from + place in next to"
    r["current_memory"] = dict(r["current_memory"]); r["current_memory"]["primitive"] = r["current_primitive"]
    cases.append(("known_fallback_not_flagged_incomplete", "", episode([r])))

    # 17) 负对照：无数字后缀的 handle（`grated_cheese`）归一化后是普通英文，
    #     出现在文本里**不算泄漏** —— 这条防的是把检查改松之外的另一头：误报
    r = dict(base_row); r["current_primitive"] = "take the grated cheese from the fridge"
    r["current_memory"] = dict(r["current_memory"]); r["current_memory"]["primitive"] = r["current_primitive"]
    ep = episode([r]); ep["skill_annotation"] = [{"object_id": [["grated_cheese"]],
                                                  "manipulating_object_id": []}]
    cases.append(("plain_word_handle_not_flagged", "", ep))

    #     配套正对照：**原串** `grated_cheese`（带下划线）出现在文本里仍必须抓。
    #     两条一起才说明改的是判据、不是把检查关小：自然语言形态放行、原串形态照抓。
    r = dict(base_row); r["current_primitive"] = "take the grated_cheese from the fridge"
    r["current_memory"] = dict(r["current_memory"]); r["current_memory"]["primitive"] = r["current_primitive"]
    ep = episode([r]); ep["skill_annotation"] = [{"object_id": [["grated_cheese"]],
                                                  "manipulating_object_id": []}]
    cases.append(("plain_word_handle_raw_form_still_flagged", "oracle_leak", ep))

    # 18) 目标文本含下划线：+4 换算的前提被打破
    r = dict(base_row); r["current_skill"] = "move to the coffee_table"
    r["current_memory"] = dict(r["current_memory"]); r["current_memory"]["skill"] = r["current_skill"]
    cases.append(("underscore_in_target_text", "underscore_in_target_text", episode([r])))

    # 19) 空 memory_annotation
    ep = episode([dict(base_row)]); ep["memory_annotation"] = []
    cases.append(("empty_memory_annotation", "empty_memory_annotation", ep))

    results = []
    with tempfile.TemporaryDirectory(prefix="moma_gate_selftest_") as td:
        tmp = Path(td)
        opts = {
            "prompt_max_len": args.prompt_max_len,
            "subtask_max_len": args.subtask_max_len,
            "action_dim": args.action_dim,
            "state_str": state_str,
            "deep_paths": set(),
            "frames_per_episode": 0,
            "seed": args.seed,
            "task_text_source": "annotation",
            "episode_tasks": {},
            "anchor_stride": 1,
            "anchor_offset": 0,
        }
        _init_worker(opts)
        for name, expect_key, ep in cases:
            p = tmp / f"{name}.json"
            p.write_text(json.dumps(ep, ensure_ascii=False), encoding="utf-8")
            res = scan_episode(str(p))
            viol = res["violations"]
            if expect_key == "":
                ok = len(viol) == 0
                results.append({"case": name, "expect": "no violations", "got": viol, "passed": ok})
            else:
                ok = viol.get(expect_key, 0) >= 1
                results.append({"case": name, "expect": expect_key, "got": viol, "passed": ok})
        # --- lerobot 口径的一对：有 task 文本 / 缺 task 文本 ---------------
        # 生产走 prompt_from_task=True 取 meta/episodes.jsonl 的 tasks[0]。
        # fixture 的 stem 解析不出 episode_index，所以键用 None。
        ep_l = episode([dict(base_row)])
        pl = tmp / "lerobot_task_present.json"
        pl.write_text(json.dumps(ep_l, ensure_ascii=False), encoding="utf-8")
        _init_worker({**opts, "task_text_source": "lerobot",
                      "episode_tasks": {None: "Turn on the radio receiver in the living room."}})
        res = scan_episode(str(pl))
        results.append({"case": "lerobot_task_present",
                        "expect": "no missing_lerobot_task_text", "got": res["violations"],
                        "passed": res["violations"].get("missing_lerobot_task_text", 0) == 0})
        _init_worker({**opts, "task_text_source": "lerobot", "episode_tasks": {}})
        res = scan_episode(str(pl))
        results.append({"case": "lerobot_task_missing_is_flagged",
                        "expect": "missing_lerobot_task_text", "got": res["violations"],
                        "passed": res["violations"].get("missing_lerobot_task_text", 0) >= 1})
        _init_worker(opts)

    n_pass = sum(1 for r in results if r["passed"])
    return {"cases": results, "passed": n_pass, "total": len(results), "all_passed": n_pass == len(results)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _default_jobs() -> int:
    """按 cgroup CPU 配额定并行度；读不到再退回 os.cpu_count()。"""
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if q > 0 and per > 0:
            return max(1, min(8, q // per))
    except Exception:  # noqa: BLE001
        pass
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            return max(1, min(8, int(raw[0]) // int(raw[1])))
    except Exception:  # noqa: BLE001
        pass
    return min(8, os.cpu_count() or 4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoMA-VLA 训练准入 gate")
    p.add_argument("--mode", choices=("formal", "diagnostic"), default="formal",
                   help="formal（默认）：所有对照来源必传，缺一个就拒绝运行。"
                        "diagnostic：允许缺，但缺的项一律渲染成 NOT-MEASURED 的告警，"
                        "**绝不渲染成 PASS** —— 没查到不等于通过")
    p.add_argument("--task-text-source", choices=("lerobot", "annotation"), default="lerobot",
                   help="`Task:` 段的文本来源。生产走 prompt_from_task=True，取的是 LeRobot "
                        "meta/episodes.jsonl 的 tasks[0]（默认）；annotation 的 task_name 是"
                        "另一个更短的串，用它会把 prompt 预算低估（实测 max 161 vs 243）")
    p.add_argument("--expect-tokenizer-md5", type=str, default=None,
                   help="必须匹配的 SentencePiece 模型文件 md5。换了尺子而不报错，"
                        "整套数字都会安静地变成另一把尺子量出来的")
    p.add_argument("--expect-vocab-size", type=int, default=None)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--expect-tasks", type=int, default=DEFAULT_EXPECT_TASKS)
    p.add_argument("--expect-episodes-per-task", type=int, default=DEFAULT_EXPECT_EPISODES_PER_TASK)
    p.add_argument("--expect-episodes", type=int, default=DEFAULT_EXPECT_EPISODES)
    p.add_argument("--expect-intervals", type=int, default=DEFAULT_EXPECT_INTERVALS)
    p.add_argument("--subtask-max-len", type=int, default=DEFAULT_SUBTASK_MAX_LEN,
                   help="必传：那次 run 实际生效的 subtask_max_len（决策中：128/160/192）")
    p.add_argument("--prompt-max-len", type=int, default=DEFAULT_PROMPT_MAX_LEN,
                   help="必传：那次 run 实际生效的 prompt_max_len / max_token_len（决策中：512/320）")
    p.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    # 必传，不给默认值：launcher 的默认值不是那次 run 的配置。
    # `pretrain_config.py:411` 是 16，`TrainConfig` 默认是 2，单机启动脚本又把它覆盖回 2。
    # gate 不能在一个假设的 worker 数上给出 PASS。
    p.add_argument("--world-size", type=int, default=None,
                   help="必传：那次 run 实际生效的 WORLD_SIZE")
    p.add_argument("--num-workers", type=int, default=None,
                   help="必传：那次 run 实际生效的 DataLoader num_workers（不是配置默认值）")
    p.add_argument("--frames-meta-root", type=str, default=None,
                   help="视频语料 LeRobot root（需含 meta/episodes.jsonl）；传空串则显式跳过联表检查")
    # anchor 采样步长：训练侧 `dataset.py:_read_streaming_anchor_env` 从
    # OPENPI_B1K_ANCHOR_STRIDE / _OFFSET 读，缺省 1/0；TrainConfig 的
    # `streaming_anchor_stride`(:147) / `epoch_anchor_offsets`(:175) 默认也是 1 / None。
    # ⚠️ 这与 HeldMemory 的 `planner_stride=5` 是**不同层的步长**，不要混。
    p.add_argument("--anchor-stride", type=int, default=None,
                   help="必传：那次 run 生效的 anchor 步长。gate 按它生成**对齐的 anchor**再验命中，"
                        "而不是拿任意帧冒充")
    p.add_argument("--anchor-offset", type=int, default=None, help="必传：那次 run 生效的 anchor 偏移")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="dataset.py:1334 的 chunk 大小，决定分片单元数")
    p.add_argument("--max-uncovered-frames", type=int, default=0,
                   help="按 [0,L) 采样时没有 Memory 区间覆盖的帧数上限")
    p.add_argument("--max-dead-chunks", type=int, default=0,
                   help="整块落在标注覆盖之外的 chunk 数上限")
    p.add_argument("--expect-path-delta", type=int, default=4,
                   help="两条 tokenizer 口径的逐行差值期望取值（当前数据实测为 4）")
    p.add_argument("--report-limits", type=str, default="128,160,192",
                   help="除生效上限外，额外按这些候选上限报出 max 与越界条数；"
                        "上限还在决策中，报三档比报单档有信息量，成本几乎为零")
    p.add_argument("--expect-overhang-episodes", type=int, default=-1,
                   help="`last_end > length` 的 episode 数期望值。数据侧 clamp 后应为 0，"
                        "这是「gate 量的是新数据」最可判定的凭据（chunk 计数对此不敏感）。-1 = 只报不判")
    p.add_argument("--config-scope-expect", type=Path, default=None,
                   help="subtask_max_len 作用域期望文件（见 config_scope_guard.py）；"
                        "断言 MoMA 配置是新上限、其余配置仍是 128、类默认值未被改")
    p.add_argument("--dead-chunks-expected", type=int, default=-1,
                   help="训练侧 run manifest 里的 chunk 剔除计数；两边必须相等。-1 = 只报不判")
    p.add_argument("--baseline-target-max", type=int, default=-1,
                   help="planner_target_text token max 的基线值，用于报出重生成后的变化量")
    p.add_argument("--max-bridge-anomalies", type=int, default=0,
                   help="intra/inter bridge 结构不变量允许的违例条数（软检查阈值）")
    p.add_argument("--max-fallback-intervals", type=int, default=-1,
                   help="组合型 primitive fallback 文案允许命中的区间数；-1 = 只报不判")
    p.add_argument("--distribution-baselines", type=Path, default=None,
                   help="分布观察项的版本化登记基线（按 manifest md5 索引）。提供后，"
                        "compact memory 分布按**该数据版本登记的值**判定；未登记的版本 "
                        "fail-closed 记 NOT-MEASURED，不自动学习")
    p.add_argument("--compact-p99-max", type=int, default=DEFAULT_COMPACT_P99)
    p.add_argument("--compact-max-max", type=int, default=DEFAULT_COMPACT_MAX)
    p.add_argument("--sample-episodes", type=int, default=100, help="深度抽查的 episode 数（>=100）")
    p.add_argument("--frames-per-episode", type=int, default=32)
    # 默认按 cgroup 配额而不是 nproc：这台机 nproc=119 但 cpu.cfs_quota_us/period=8。
    p.add_argument("--jobs", type=int, default=_default_jobs())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-files", type=int, default=0, help="只扫前 N 个文件（调试用；有偏，仅供冒烟）")
    p.add_argument("--file-list", type=Path, default=None,
                   help="预先生成的 episode 文件清单（每行一个绝对路径）。这台机 glob 一万个文件要 "
                        "47 秒，扫描重跑多次时用它省掉重复枚举；文件数仍会被打印和断言")
    p.add_argument("--file-stride", type=int, default=1,
                   help="按步长跨 task 取子集（无偏，用于阈值扫描；配 --skip-count-checks 用）")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--self-test", action="store_true", help="只跑检测器自检（正对照），不扫数据")
    p.add_argument("--skip-count-checks", action="store_true", help="跳过 episode/interval 总数硬检查（子集运行用）")
    p.add_argument("--chunk-stats-json", type=Path, default=None,
                   help="训练侧 BehaviorLeRobotDataset.memory_chunk_stats() 落盘的 json，"
                        "含 memory_chunks_kept / memory_dead_chunks_dropped / memory_chunks_clipped。"
                        "期望值从这里读，不写死在 gate 里")
    p.add_argument("--baseline-report", type=Path, default=None,
                   help="上一次的 gate 报告 json，用于报出重生成后各项的变化量")
    p.add_argument("--fail-on-soft", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test:
        required = [("--world-size", args.world_size),
                    ("--num-workers", args.num_workers),
                    ("--frames-meta-root", args.frames_meta_root),
                    ("--subtask-max-len", args.subtask_max_len),
                    ("--prompt-max-len", args.prompt_max_len)]
        if args.mode == "formal":
            # 正式模式：所有跨实现/跨版本对照的来源都必须显式给出。
            # 「可选参数缺了就当通过」是 fail-open —— gate 的语义必须是「没查到就不能说通过」。
            required += [("--anchor-stride", args.anchor_stride),
                         ("--anchor-offset", args.anchor_offset),
                         ("--chunk-stats-json", args.chunk_stats_json),
                         ("--config-scope-expect", args.config_scope_expect),
                         ("--distribution-baselines", args.distribution_baselines),
                         ("--expect-tokenizer-md5", args.expect_tokenizer_md5),
                         ("--expect-vocab-size", args.expect_vocab_size)]
        missing = [n for n, v in required if v is None]
        # 取值域按训练侧 `dataset.py:_read_streaming_anchor_env` 的同一套约束校验，
        # 免得 gate 用一个训练侧根本不接受的组合算出「通过」。
        if args.anchor_stride is not None and args.anchor_stride < 1:
            p.error(f"--anchor-stride 必须 >= 1，得到 {args.anchor_stride}")
        if args.anchor_offset is not None:
            if args.anchor_stride is None:
                p.error("--anchor-offset 必须与 --anchor-stride 一起给")
            if not 0 <= args.anchor_offset < args.anchor_stride:
                p.error(f"--anchor-offset 必须满足 0 <= offset < stride；"
                        f"得到 offset={args.anchor_offset}, stride={args.anchor_stride}")
        if args.mode == "formal" and not str(args.frames_meta_root).strip():
            # 空串是「显式放弃」，那在 formal 模式下不成立：放弃之后 EPISODE_JOIN_1TO1 /
            # FRAME_COVERAGE / chunk 四条等十项都失去依据，报告会看起来更干净而不是更差。
            p.error("--mode formal 不接受空的 --frames-meta-root：放弃联表会让十项检查失去依据。"
                    "确实要跳过请用 --mode diagnostic，那时它们会登记成 NOT-MEASURED。")
        if missing:
            p.error(
                "以下参数必须显式传入，不提供默认值：" + " ".join(missing) + "。\n"
                "  --world-size / --num-workers：launcher 的默认值不是那次 run 的配置 —— "
                "pretrain_config.py:411 是 16，train_config.py:99 的默认是 2，而单机启动脚本 "
                "run_pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep.sh:51 又把它覆盖回 2。\n"
                "  只能来自那次 run 的生效值（最好是 run manifest），不要从配置文件读。\n"
                "  --subtask-max-len / --prompt-max-len：这两个上限还在决策中"
                "（128/160/192、512/320）。写死任何一个候选值，就等于让 gate 在一个假设的"
                "上限上给出 PASS。\n"
                "  --frames-meta-root 传空串 '' 表示显式放弃联表检查（记为 NOT-MEASURED）。\n"
                "  只想快速看数、不做正式准入判定时用 --mode diagnostic：缺的项会渲染成 "
                "NOT-MEASURED 告警，但**不会渲染成 PASS**。"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    if args.self_test:
        res = run_self_test(args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.out:
            args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSELF_TEST: {'ok' if res['all_passed'] else 'FAILED'} {res['passed']}/{res['total']}")
        return 0 if res["all_passed"] else 1

    ledger = Ledger()
    root: Path = args.data_root

    # --- 文件枚举（先界定集合规模，再谈命中数）-----------------------------
    task_dirs = sorted(d for d in root.glob("task-*") if d.is_dir())
    filelist_audit: dict[str, Any] = {"used": False}
    if args.file_list and not Path(args.file_list).exists():
        print(f"CANNOT-ASSESS: --file-list {args.file_list} 不存在。"
              "此前这里会静默回退到 glob —— 那等于把「清单丢了」渲染成「正常运行」。")
        return 3
    if args.file_list and Path(args.file_list).exists():
        # 「文件存在」不是「文件完整」：被 kill 的写入会留下截断清单，而 exists() 照样为真
        # （实测踩过一次 6,416 行的截断）。所以用清单前要对账：数量、归属、可读性。
        files = [Path(x) for x in Path(args.file_list).read_text().split() if x.strip()]
        truth = 0
        for td in task_dirs:
            with os.scandir(td) as it:
                truth += sum(1 for e in it
                             if e.is_file() and e.name.startswith("episode_")
                             and e.name.endswith(".json"))
        rootr = str(Path(root).resolve())
        outside = [str(f) for f in files if not str(Path(f).resolve()).startswith(rootr + os.sep)]
        gone = [str(f) for f in files[:: max(1, len(files) // 200)] if not Path(f).exists()]
        filelist_audit = {"used": True, "path": str(args.file_list),
                          "listed": len(files), "on_disk": truth,
                          "count_matches": len(files) == truth,
                          "outside_data_root": len(outside), "outside_examples": outside[:5],
                          "sampled_missing": len(gone), "missing_examples": gone[:5]}
        if not filelist_audit["count_matches"] or outside or gone:
            print("CANNOT-ASSESS: --file-list 对账失败 "
                  f"{json.dumps(filelist_audit, ensure_ascii=False)}", flush=True)
            return 3
    else:
        files = sorted(root.glob("task-*/episode_*.json"))
    if args.file_stride > 1:
        files = files[:: args.file_stride]
    if args.limit_files:
        files = files[: args.limit_files]
    n_files = len(files)
    print(f"[enumerate] task_dirs={len(task_dirs)} episode_files={n_files}", flush=True)
    if n_files == 0:
        print("CANNOT-ASSESS: 0 episode files enumerated under " + str(root))
        return 3

    fingerprint = data_fingerprint(root, files)

    # --- tokenizer 尺子 ----------------------------------------------------
    tok = build_tokenizer(args.prompt_max_len, args.subtask_max_len)
    fp_tok = tokenizer_fingerprint(tok)
    state_str, state_tokens, state_variants = worst_case_state_string(tok, args.action_dim)
    print(f"[tokenizer] {fp_tok['sentencepiece_model_path']} md5={fp_tok['sentencepiece_model_md5']} "
          f"vocab={fp_tok['vocab_size']} sp={fp_tok['sentencepiece_lib_version']}", flush=True)
    print(f"[state] worst-case state tokens={state_tokens} variants={state_variants}", flush=True)

    # --- 深度抽查的 episode（每个 task 均摊，保证跨 task 覆盖，不用 head）---
    rng = random.Random(args.seed)
    by_task: dict[str, list[Path]] = {}
    for f in files:
        by_task.setdefault(f.parent.name, []).append(f)
    per_task = max(1, -(-args.sample_episodes // max(1, len(by_task))))
    deep_paths: set[str] = set()
    for _t, group in sorted(by_task.items()):
        deep_paths.update(str(p) for p in rng.sample(group, min(per_task, len(group))))
    print(f"[sample] deep-checked episodes={len(deep_paths)} across {len(by_task)} tasks "
          f"({per_task}/task)", flush=True)

    # --- 视频帧语料联表（annotation 是 derived 层，本身不含帧/state）--------
    episode_lengths: dict[int, int] | None = None
    episode_tasks: dict[int, str] = {}
    frames_meta_status = "not-requested (显式跳过联表检查)"
    frames_meta_path = None
    if args.frames_meta_root.strip():
        frames_meta_path = Path(args.frames_meta_root) / "meta" / "episodes.jsonl"
        if frames_meta_path.exists():
            episode_lengths = {}
            episode_tasks = {}
            with frames_meta_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    idx = int(rec["episode_index"])
                    episode_lengths[idx] = int(rec["length"])
                    tl = rec.get("tasks") or []
                    episode_tasks[idx] = tl[0] if tl else ""
            frames_meta_status = (f"loaded {len(episode_lengths)} episodes, "
                                  f"{sum(1 for v in episode_tasks.values() if v)} 条带 task 文本")
        else:
            frames_meta_status = f"NOT-MEASURED: {frames_meta_path} 不存在"
    print(f"[frames-meta] {frames_meta_status}", flush=True)

    if args.task_text_source == "lerobot" and not episode_tasks:
        print("CANNOT-ASSESS: --task-text-source lerobot 需要 meta/episodes.jsonl 的 tasks 字段，"
              f"但没读到（{frames_meta_status}）。生产的 Task: 文本来自那里，用 annotation 的 "
              "task_name 代替会把 prompt 预算低估（实测 max 161 vs 243），所以这里不静默降级。")
        return 3

    opts = {
        "episode_tasks": episode_tasks,
        "task_text_source": args.task_text_source,
        "episode_lengths": episode_lengths,
        "chunk_size": args.chunk_size,
        "anchor_stride": args.anchor_stride if args.anchor_stride is not None else 1,
        "anchor_offset": args.anchor_offset if args.anchor_offset is not None else 0,
        "prompt_max_len": args.prompt_max_len,
        "subtask_max_len": args.subtask_max_len,
        "action_dim": args.action_dim,
        "state_str": state_str,
        "deep_paths": deep_paths,
        "frames_per_episode": args.frames_per_episode,
        "seed": args.seed,
    }

    # --- 全量扫描 ----------------------------------------------------------
    agg_target: list[int] = []
    agg_model_target: list[int] = []
    agg_prompt: list[int] = []
    agg_compact: list[int] = []
    agg_prod: list[int] = []
    agg_prompt_other: list[int] = []
    path_deltas: Counter = Counter()
    len_by_tt: dict = {}
    overhang_by_tt: Counter = Counter()
    violations: Counter = Counter()
    leak_kinds: Counter = Counter()
    handles_no_suffix: set = set()
    transitions: Counter = Counter()
    evidence: dict[str, list[str]] = {}
    readable = 0
    total_intervals = 0
    sentinel_text = 0
    sentinel_re = 0
    pos_memory_label = 0
    pos_nonzero_len = 0
    deep_done = 0
    deep_frames = 0
    fb_intervals = 0
    fb_model_target = 0
    fb_episodes = 0
    fb_by_task: Counter = Counter()
    unc_head = 0
    unc_tail = 0
    overhang = 0
    n_chunks_total = 0
    dead_chunks = 0
    eps_uncovered = 0
    eps_dead = 0
    join_missing = 0
    total_video_frames = 0
    overhang_eps = 0
    vr_probes = 0
    vr_misses = 0
    live_chunks = 0
    partial_chunks = 0
    clamped_empty = 0
    annotated_frames = 0
    ann_outside_live = 0
    ann_lost_if_partial = 0
    clamped_probes = 0
    clamped_misses = 0
    no_anchor_chunks = 0

    ctx = mp.get_context("fork")
    chunksize = max(1, n_files // (args.jobs * 8) or 1)
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(opts,)) as pool:
        for k, res in enumerate(pool.imap_unordered(scan_episode, [str(f) for f in files], chunksize=chunksize), 1):
            if res["readable"]:
                readable += 1
            total_intervals += res["n_intervals"]
            agg_target.extend(res["target_lengths"])
            agg_model_target.extend(res["model_target_lengths"])
            agg_prompt.extend(res["prompt_lengths"])
            agg_compact.extend(res["compact_lengths"])
            agg_prod.extend(res.get("prod_lengths", []))
            agg_prompt_other.extend(res.get("prompt_lengths_other_source", []))
            path_deltas.update(res.get("path_deltas", {}))
            for _tt, _b in (res.get("len_by_transition") or {}).items():
                _cur = len_by_tt.setdefault(_tt, {"n": 0, "prod_max": 0, "doc_max": 0})
                _cur["n"] += _b["n"]
                _cur["prod_max"] = max(_cur["prod_max"], _b["prod_max"])
                _cur["doc_max"] = max(_cur["doc_max"], _b["doc_max"])
            overhang_by_tt.update(res.get("overhang_by_transition", {}))
            violations.update(res["violations"])
            leak_kinds.update(res["leak_kinds"])
            handles_no_suffix.update(res.get("handles_without_suffix", []))
            transitions.update(res["transition_counts"])
            sentinel_text += res["sentinel_text_hits"]
            sentinel_re += res["sentinel_re_hits"]
            pos_memory_label += res["poscontrol_memory_label"]
            pos_nonzero_len += res["poscontrol_nonzero_len"]
            deep_done += int(res["deep_checked"])
            deep_frames += res["deep_frames"]
            fb_intervals += res.get("fallback_intervals", 0)
            fb_model_target += res.get("fallback_model_target", 0)
            fb_episodes += res.get("fallback_episode", 0)
            fb_by_task.update(res.get("fallback_by_task", {}))
            unc_head += res.get("uncovered_head", 0)
            unc_tail += res.get("uncovered_tail", 0)
            overhang += res.get("overhang", 0)
            n_chunks_total += res.get("n_chunks", 0)
            dead_chunks += res.get("dead_chunks", 0)
            eps_uncovered += res.get("has_uncovered", 0)
            eps_dead += res.get("has_dead_chunk", 0)
            join_missing += res.get("join_missing_in_meta", 0)
            overhang_eps += res.get("overhang_episode", 0)
            vr_probes += res.get("video_range_probes", 0)
            vr_misses += res.get("video_range_misses", 0)
            live_chunks += res.get("live_chunks", 0)
            partial_chunks += res.get("partial_chunks", 0)
            clamped_empty += res.get("clamped_empty", 0)
            annotated_frames += res.get("annotated_frames", 0)
            ann_outside_live += res.get("annotated_frames_outside_live_chunks", 0)
            ann_lost_if_partial += res.get("annotated_frames_lost_if_partial_dropped", 0)
            clamped_probes += res.get("clamped_probes", 0)
            clamped_misses += res.get("clamped_misses", 0)
            no_anchor_chunks += res.get("chunks_without_aligned_anchor", 0)
            total_video_frames += res.get("video_length") or 0
            for kind, msgs in res["evidence"].items():
                b = evidence.setdefault(kind, [])
                for m in msgs:
                    if len(b) < 12:
                        b.append(m)
            if k % 2000 == 0:
                print(f"[scan] {k}/{n_files} files, intervals={total_intervals}, "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    scan_secs = time.time() - t0
    print(f"[scan] done {n_files} files in {scan_secs:.0f}s", flush=True)

    # --- 可读性先断言，读数不足即 CANNOT-ASSESS ----------------------------
    if readable != n_files:
        ledger.add(
            "READABLE_EPISODES",
            hard=True,
            passed=False,
            detail=f"只有 {readable}/{n_files} 个 episode JSON 可读 ⇒ 其余检查的分母不完整",
            measured=readable,
            threshold=n_files,
            evidence=evidence.get("unreadable_episode", []),
        )
        report = {"verdict": "CANNOT-ASSESS", "readable": readable, "files": n_files}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3
    ledger.add("READABLE_EPISODES", hard=True, passed=True,
               detail="全部 episode JSON 可读", measured=readable, threshold=n_files)

    # --- 尺子必须是登记的那一把 ----------------------------------------------
    # 之前只把 md5/vocab 记进报告、不断言 ⇒ 换了模型文件，整套数字会安静地变成
    # 另一把尺子量出来的，而 gate 照样 PASS。
    if args.expect_tokenizer_md5 or args.expect_vocab_size is not None:
        bad_ruler = []
        if args.expect_tokenizer_md5 and fp_tok["sentencepiece_model_md5"] != args.expect_tokenizer_md5:
            bad_ruler.append(f"md5 {fp_tok['sentencepiece_model_md5']} != {args.expect_tokenizer_md5}")
        if args.expect_vocab_size is not None and int(fp_tok["vocab_size"]) != args.expect_vocab_size:
            bad_ruler.append(f"vocab_size {fp_tok['vocab_size']} != {args.expect_vocab_size}")
        ledger.add("TOKENIZER_FINGERPRINT", hard=True, passed=not bad_ruler,
                   detail="SentencePiece 模型文件 md5 与 vocab_size 必须等于登记值",
                   measured={k: fp_tok[k] for k in
                             ("sentencepiece_model_path", "sentencepiece_model_md5",
                              "vocab_size", "sentencepiece_lib_version")},
                   threshold={"md5": args.expect_tokenizer_md5, "vocab_size": args.expect_vocab_size},
                   evidence=bad_ruler)
    else:
        ledger.add("TOKENIZER_FINGERPRINT", hard=False, passed=False,
                   detail="NOT-MEASURED：未提供 --expect-tokenizer-md5 / --expect-vocab-size。"
                          "没查到不等于通过，所以这里不渲染成 PASS",
                   measured={k: fp_tok[k] for k in ("sentencepiece_model_md5", "vocab_size")},
                   threshold="NOT-MEASURED")

    # --- 正/负对照 ----------------------------------------------------------
    ledger.add(
        "NEGATIVE_CONTROL_SENTINELS", hard=True,
        passed=(sentinel_text == 0 and sentinel_re == 0),
        detail=f"必然为 0 的哨兵：literal={sentinel_text} regex={sentinel_re}",
        measured={"literal": sentinel_text, "regex": sentinel_re}, threshold=0,
    )
    ledger.add(
        "POSITIVE_CONTROL_COVERAGE", hard=True,
        passed=(pos_memory_label == total_intervals and pos_nonzero_len == total_intervals and total_intervals > 0),
        detail=(f"每行 target 都含 `Memory:` 标签且 token 长度 >0 ⇒ 扫描确实覆盖了每一行："
                f"{pos_memory_label}/{total_intervals} 与 {pos_nonzero_len}/{total_intervals}"),
        measured={"memory_label": pos_memory_label, "nonzero_len": pos_nonzero_len},
        threshold=total_intervals,
    )

    # --- 计数类 -------------------------------------------------------------
    if not args.skip_count_checks:
        ledger.add("EPISODE_FILE_COUNT", hard=True, passed=(n_files == args.expect_episodes),
                   detail=f"episode 文件数 {n_files}", measured=n_files, threshold=args.expect_episodes)
        bad_tasks = {d.name: len(list(d.glob("episode_*.json"))) for d in task_dirs
                     if len(list(d.glob("episode_*.json"))) != args.expect_episodes_per_task}
        ledger.add("TASK_LAYOUT", hard=True,
                   passed=(len(task_dirs) == args.expect_tasks and not bad_tasks),
                   detail=f"{len(task_dirs)} 个 task 目录，每个应有 {args.expect_episodes_per_task} 个 episode",
                   measured={"task_dirs": len(task_dirs), "off_spec": bad_tasks},
                   threshold={"task_dirs": args.expect_tasks, "per_task": args.expect_episodes_per_task})
        if args.expect_intervals < 0:
            ledger.add("INTERVAL_TOTAL", hard=False, passed=True,
                       detail=f"memory interval 总数 {total_intervals}（--expect-intervals=-1 ⇒ 只报不判）",
                       measured=total_intervals, threshold="report-only")
        else:
            ledger.add("INTERVAL_TOTAL", hard=True, passed=(total_intervals == args.expect_intervals),
                       detail=(f"memory interval 总数 {total_intervals}"
                               f"（与期望差 {total_intervals - args.expect_intervals:+d}）"),
                       measured=total_intervals, threshold=args.expect_intervals)

    ledger.add("INTERVAL_NONEMPTY", hard=True, passed=(violations.get("empty_memory_annotation", 0) == 0),
               detail="每个 episode 的 memory_annotation 非空",
               measured=violations.get("empty_memory_annotation", 0), threshold=0,
               evidence=evidence.get("empty_memory_annotation", []))

    structure_keys = (
        "interval_inverted", "interval_not_contiguous", "memory_idx_not_sequential",
        "frame_duration_shape", "coverage_start_mismatch", "coverage_end_mismatch",
        "schema_version", "interval_convention", "missing_task_name",
    )
    struct_total = sum(violations.get(k, 0) for k in structure_keys)
    ledger.add("INTERVAL_STRUCTURE", hard=True, passed=(struct_total == 0),
               detail="区间连续、无倒置、memory_idx 递增、覆盖范围等于 valid_duration、schema 字段正确",
               measured={k: violations.get(k, 0) for k in structure_keys}, threshold=0,
               evidence=[m for k in structure_keys for m in evidence.get(k, [])][:12])

    # --- token 长度（训练侧尺子）-------------------------------------------
    tgt = describe(agg_target)
    mtgt = describe(agg_model_target)
    pmt = describe(agg_prompt)
    cmp_ = describe(agg_compact)

    prod = describe(agg_prod)
    tgt_max = tgt.get("max", 1 << 30)
    prod_max = prod.get("max", 1 << 30)
    delta = None if args.baseline_target_max < 0 else prod_max - args.baseline_target_max

    # hard 判定用**生产口径**（tokenize_memory，保留换行）。文档 3.4.4 写的是
    # tokenize_subtask，两者恒差 4 个换行 token；两列都报，差值也报。
    ledger.add("TARGET_TOKEN_MAX", hard=True,
               passed=(prod_max <= args.subtask_max_len),
               detail=("planner_target_text 经**生产路径** tokenize_memory 的 token 最大值"
                       "（BOS+text+EOS，未截断，保留换行）。数据重新生成后即使仍 ≤ 上限，"
                       "也必须看到它涨了多少"),
               measured={"max_production_tokenize_memory": prod_max,
                         "max_doc_path_tokenize_subtask": tgt_max,
                         "delta_two_paths": (None if prod_max >= (1 << 30) or tgt_max >= (1 << 30)
                                             else prod_max - tgt_max),
                         "headroom": args.subtask_max_len - prod_max,
                         "baseline_max": (args.baseline_target_max if args.baseline_target_max >= 0 else None),
                         "delta_vs_baseline": delta,
                         "p99": prod.get("p99"), "p999": prod.get("p999")},
               threshold=args.subtask_max_len)
    # 候选上限逐档报：用户还没拍板 128 / 160 / 192，报三档才回答得了
    # 「在哪个上限下会有多少行越界」。同一批长度比较三次，成本可忽略。
    try:
        cand = sorted({int(x) for x in args.report_limits.split(",") if x.strip()})
    except ValueError:
        cand = []
    limit_table = {
        str(lim): {
            "production_over": sum(1 for x in agg_prod if x > lim),
            "doc_path_over": sum(1 for x in agg_target if x > lim),
            "production_headroom": lim - prod_max,
            "doc_path_headroom": lim - tgt_max,
        }
        for lim in cand
    }
    delta_values = sorted(path_deltas)
    ledger.add("TWO_PATH_DELTA_INVARIANT", hard=True,
               passed=(delta_values == [args.expect_path_delta]
                       and violations.get("underscore_in_target_text", 0) == 0),
               detail=("生产口径 tokenize_memory 与文档口径 tokenize_subtask 的**逐行**差值必须"
                       f"只有一个取值 {args.expect_path_delta}，且目标文本不得含下划线。"
                       "两条清洗是两套（tokenize_subtask 压换行且把 `_` 换成空格，"
                       "tokenize_memory 两样都不做），所以「+4」只是当前数据上的实测值、"
                       "不是转换公式；文本一旦带下划线它就静默失效"),
               measured={"delta_value_counts": {str(k): v for k, v in sorted(path_deltas.items())},
                         "distinct_deltas": delta_values,
                         "underscore_rows": violations.get("underscore_in_target_text", 0)},
               threshold={"distinct_deltas": [args.expect_path_delta], "underscore_rows": 0},
               evidence=evidence.get("underscore_in_target_text", []))

    ledger.add("TARGET_TOKEN_LIMIT_TABLE", hard=False, passed=True,
               detail=("在每个候选上限下会有多少行越界（生产口径 tokenize_memory / "
                       "文档口径 tokenize_subtask）。只报不判 —— 生效上限由 "
                       "--subtask-max-len 决定，判定在 TARGET_TOKEN_MAX 那条"),
               measured={"effective_limit": args.subtask_max_len,
                         "production_max": prod_max, "doc_path_max": tgt_max,
                         "by_candidate_limit": limit_table,
                         "by_transition_type": len_by_tt},
               threshold="report-only")
    ledger.add("TARGET_TOKEN_MAX_DOC_PATH", hard=False,
               passed=True,
               detail=("文档 3.4.4 口径（tokenize_subtask，换行压成空格）的同一组数，只报不判。"
                       "保留它是因为文档与实现在「用哪个函数」上不一致，这个差异要回流文档，"
                       "不能靠换函数掩掉"),
               measured=tgt, threshold="report-only")
    ledger.add("MEMORY_CODEC_ACCEPTS_ALL_TARGETS", hard=True,
               passed=(violations.get("memory_codec_rejects_text", 0) == 0),
               detail=("每一行 planner_target_text 都要能过 MemoryTextCodec 的 "
                       "validate_field_structure（五个标签、顺序固定、标签不得出现在字段正文里）"),
               measured=violations.get("memory_codec_rejects_text", 0), threshold=0,
               evidence=evidence.get("memory_codec_rejects_text", []))
    if args.baseline_target_max >= 0:
        # ⚠️ 这条比的是**生产口径**（tokenize_memory）的 max。若传进来的基线是文档口径
        # （tokenize_subtask）的数，delta 会凭空多出两条口径的固定差值，看起来像「数据变长了」
        # 而其实什么都没变。这里主动把嫌疑标出来，而不是让下一个读者去翻文档。
        looks_like_doc_path = (delta is not None and delta == (prod_max - tgt_max) != 0)
        caveat = ""
        if looks_like_doc_path:
            caveat = (f"⚠️ delta 恰好等于两条口径的固定差值 {prod_max - tgt_max}，"
                      f"且 baseline({args.baseline_target_max}) == 本次的文档口径 max({tgt_max})"
                      if args.baseline_target_max == tgt_max else "")
        ledger.add("TARGET_TOKEN_MAX_NO_REGRESSION", hard=False,
                   passed=(delta is not None and delta <= 0),
                   detail=("相对基线的变化量（比的是**生产口径** tokenize_memory 的 max）。"
                           "涨了不一定是错，但必须被看到 —— 补 phrase 模板会让文本变长。"
                           + (f" {caveat} ⇒ 这更像是**口径变更**而不是数据变化："
                              "基线值应当也取生产口径，或者改用 --baseline-report"
                              "（它逐字段对齐口径）。" if caveat else "")),
                   measured={"baseline": args.baseline_target_max,
                             "baseline_path": ("疑似文档口径 tokenize_subtask" if caveat
                                               else "调用方声明为生产口径"),
                             "now_production": prod_max, "now_doc_path": tgt_max,
                             "delta": delta,
                             "two_path_fixed_delta": prod_max - tgt_max},
                   threshold={"delta": "<= 0"})
    ledger.add("TARGET_OVER_MAX_COUNT", hard=True,
               passed=(violations.get("prod_target_over_subtask_max_len", 0) == 0),
               detail="生产口径下超过 subtask_max_len 的 target 条数必须为 0",
               measured={"production_path": violations.get("prod_target_over_subtask_max_len", 0),
                         "doc_path": violations.get("target_over_subtask_max_len", 0)},
               threshold=0,
               evidence=evidence.get("prod_target_over_subtask_max_len", []))
    ledger.add("EOS_PRESENT_AND_SUPERVISED", hard=True,
               passed=(violations.get("eos_lost", 0) == 0
                       and violations.get("eos_not_supervised", 0) == 0
                       and violations.get("eos_not_last_valid", 0) == 0),
               detail="EOS 必须落在有效 mask 内、被 loss_mask 监督、且是最后一个有效 token",
               measured={k: violations.get(k, 0) for k in ("eos_lost", "eos_not_supervised", "eos_not_last_valid")},
               threshold=0,
               evidence=(evidence.get("eos_lost", []) + evidence.get("eos_not_supervised", []))[:12])
    ledger.add("LOSS_MASK_CONTRACT", hard=True,
               passed=(violations.get("bos_supervised", 0) == 0
                       and violations.get("bos_missing", 0) == 0
                       and violations.get("loss_mask_contract", 0) == 0
                       and violations.get("ar_mask_contract", 0) == 0),
               detail="BOS 不监督、pad 不监督、其余 AR token 全监督；ar_mask 有效段全 1、padding 全 0",
               measured={k: violations.get(k, 0) for k in
                         ("bos_supervised", "bos_missing", "loss_mask_contract", "ar_mask_contract")},
               threshold=0)

    pmt_other = describe(agg_prompt_other)
    _miss_task = violations.get("missing_lerobot_task_text", 0)
    ledger.add("TASK_TEXT_COMPLETE", hard=(args.task_text_source == "lerobot"),
               passed=(_miss_task == 0),
               detail=("生产口径下每个 episode 都必须能从 meta/episodes.jsonl 取到 tasks[0]。"
                       "缺失时**不回退** annotation —— 回退会让「部分 episode 用了另一个口径」"
                       "退化成 note 里的软信息，而 prompt 预算照常出数"),
               measured={"missing_rows": _miss_task, "source": args.task_text_source},
               threshold=0, evidence=evidence.get("missing_lerobot_task_text", []))
    ledger.add("PROMPT_TASK_TEXT_SOURCE", hard=False, passed=True,
               detail=("`Task:` 段的两个文本来源并排量。生产走 prompt_from_task=True ⇒ 取 LeRobot "
                       "meta/episodes.jsonl 的 tasks[0]；annotation 的 task_name 是另一个更短的串。"
                       "两个数并排给出，读者不会把口径差异误读成数据变化"),
               measured={"active_source": args.task_text_source,
                         "active": {k: pmt.get(k) for k in ("p50", "p90", "p99", "max")},
                         "other_source": ("annotation" if args.task_text_source == "lerobot"
                                          else "lerobot"),
                         "other": {k: pmt_other.get(k) for k in ("p50", "p90", "p99", "max")},
                         "missing_lerobot_task_rows": violations.get("missing_lerobot_task_text", 0)},
               threshold="report-only")
    ledger.add("PROMPT_TOKEN_BUDGET", hard=True,
               passed=(violations.get("prompt_over_max_len", 0) == 0
                       and pmt.get("max", 1 << 30) <= args.prompt_max_len),
               detail=("doc 3.4.3 的 prefix（含最坏情况 state）不得超过 prompt_max_len；"
                       "prompt 从右截断，溢出时被吃掉的正好是末尾的 Previous memory"),
               measured={"max": pmt.get("max"), "over_count": violations.get("prompt_over_max_len", 0),
                         "state_tokens_worst_case": state_tokens},
               threshold=args.prompt_max_len,
               evidence=evidence.get("prompt_over_max_len", []))

    ledger.add("STATE_IS_DISCRETIZED_NOT_FRAME_COUNTER", hard=True,
               passed=(violations.get("state_field_arity", 0) == 0
                       and violations.get("state_field_not_integer", 0) == 0
                       and violations.get("state_field_range", 0) == 0),
               detail=f"State 段必须是 {args.action_dim} 个 [-1,255] 整数（256 档离散化），不是 `Frame N of M`",
               measured={k: violations.get(k, 0) for k in
                         ("state_field_arity", "state_field_not_integer", "state_field_range")},
               threshold=0)

    # --- Action Query / 监督目标口径 ---------------------------------------
    ledger.add("ACTION_QUERY_NOT_IN_CE_TEXT", hard=True,
               passed=(violations.get("action_query_in_ce_text", 0) == 0),
               detail="用于文本 CE 的 planner_target_text 里不得出现 `Action Query`（doc 3.4.3）",
               measured=violations.get("action_query_in_ce_text", 0), threshold=0,
               evidence=evidence.get("action_query_in_ce_text", []))
    ledger.add("MODEL_TARGET_RECONSTRUCTION", hard=True,
               passed=(violations.get("model_target_reconstruction", 0) == 0),
               detail="model_target_text 必须逐字节等于五行 planner_target_text + `\\nAction Query:`",
               measured=violations.get("model_target_reconstruction", 0), threshold=0,
               evidence=evidence.get("model_target_reconstruction", []))

    # --- oracle 泄漏 --------------------------------------------------------
    ledger.add("NO_ORACLE_LEAK", hard=True, passed=(violations.get("oracle_leak", 0) == 0),
               detail=("模型可见文本不得含帧计数器、annotation_index、segments、原始 object handle、"
                       "task ID、episode ID、frame index（doc 3.1 / 3.2）"),
               measured={"total": violations.get("oracle_leak", 0), "by_kind": dict(leak_kinds),
                         "handles_without_id_suffix": sorted(handles_no_suffix)[:20],
                         "handles_without_id_suffix_count": len(handles_no_suffix)},
               threshold=0, evidence=evidence.get("oracle_leak", []))

    # --- Memory 时序 / bridge ----------------------------------------------
    ledger.add("MEMORY_CHAIN_ORDERING", hard=True,
               passed=(violations.get("previous_memory_chain", 0) == 0
                       and violations.get("initial_previous_memory", 0) == 0),
               detail="previous_fixed_compact_memory[i] == fixed_compact_memory[i-1]，首区间用固定 initial Memory",
               measured={k: violations.get(k, 0) for k in ("previous_memory_chain", "initial_previous_memory")},
               threshold=0,
               evidence=(evidence.get("previous_memory_chain", []) + evidence.get("initial_previous_memory", []))[:12])

    # 标签取值域与内部一致性：文档明确写死的部分 ⇒ hard
    label_keys = ("transition_type_unknown", "bridge_at_index_zero", "current_memory_subdict_mismatch")
    ledger.add("BRIDGE_LABEL_VALUE_SET", hard=True,
               passed=(sum(violations.get(k, 0) for k in label_keys) == 0),
               detail=("transition_type 取值必须落在文档 3.4.2 的四个值里；bridge 不能出现在首区间；"
                       "current_memory 子结构必须与顶层字段一致"),
               measured={k: violations.get(k, 0) for k in label_keys}, threshold=0,
               evidence=[m for k in label_keys for m in evidence.get(k, [])][:12])

    # 结构不变量：**从数据归纳**出来的，不是文档写的 ⇒ 软检查 + 可调阈值。
    # 它的价值是数据重新生成后的回归检测，不是替文档下定义。
    struct_bridge_keys = ("intra_bridge_primitive_changed", "intra_bridge_skill_unchanged",
                          "inter_bridge_primitive_unchanged")
    bridge_anomalies = sum(violations.get(k, 0) for k in struct_bridge_keys)
    ledger.add("BRIDGE_STRUCTURAL_INVARIANT", hard=False,
               passed=(bridge_anomalies <= args.max_bridge_anomalies),
               detail=("归纳自数据的不变量：intra_primitive_skill_bridge ⇒ primitive 不变且 skill 变；"
                       "inter_primitive_bridge ⇒ primitive 变。违例说明标签与相邻行结构矛盾，"
                       "需要数据侧解释，不由本 gate 定性"),
               measured={k: violations.get(k, 0) for k in struct_bridge_keys},
               threshold=args.max_bridge_anomalies,
               evidence=[m for k in struct_bridge_keys for m in evidence.get(k, [])][:12])

    # --- frame -> interval 命中 ---------------------------------------------
    frame_keys = ("frame_before_valid_duration", "frame_not_covered", "bisect_linear_disagree")
    enough_samples = deep_done >= min(args.sample_episodes, n_files)
    ledger.add("FRAME_TO_INTERVAL_HIT", hard=True,
               passed=(sum(violations.get(k, 0) for k in frame_keys) == 0 and enough_samples and deep_frames > 0),
               detail=(f"在 {deep_done} 个随机 episode 上探测 {deep_frames} 个 frame："
                       f"bisect 查找必须命中且与线性扫描一致（doc 3.4.3 半开区间）"),
               measured={"episodes": deep_done, "frames": deep_frames,
                         **{k: violations.get(k, 0) for k in frame_keys}},
               threshold={"episodes_min": min(args.sample_episodes, n_files), "violations": 0},
               evidence=[m for k in frame_keys for m in evidence.get(k, [])][:12])

    # --- 与视频帧语料的联表 / 未命中区间 ------------------------------------
    uncovered_total = unc_head + unc_tail
    if episode_lengths is None:
        requested = bool(args.frames_meta_root.strip())
        # NOT-MEASURED 一律 passed=False：显式 opt-out 记 WARN、请求了读不到记 hard。
        # 写成 passed=True 会让人读输出里显示 [PASS] —— 那是「整条消失」的另一种形态。
        ledger.add("EPISODE_JOIN_1TO1", hard=requested, passed=False,
                   detail=(f"{frames_meta_status}。annotation 不含帧数据，不联表就无法判定 frame 覆盖"
                           + ("（已请求但读不到 ⇒ NOT-MEASURED，按 hard 记）" if requested
                              else "（调用方显式跳过）")),
                   measured="NOT-MEASURED", threshold="meta/episodes.jsonl 可读")
        # 其余 8 条本来整条消失 ⇒ 报告会因为「少查了东西」而看起来更干净。
        # 逐条登记成 NOT-MEASURED，让缺口在报告里留下痕迹。
        for _k, _why in (
            ("FRAME_COVERAGE_VS_VIDEO", "需要 meta/episodes.jsonl 的 length 才能判帧覆盖"),
            ("CHUNK_STATS_AGREE_WITH_TRAINING_SIDE", "chunk 计数依赖 episode 长度"),
            ("DEAD_CHUNK_COUNT_MANUAL_OVERRIDE", "同上"),
            ("SHARDING_AFTER_DEAD_CHUNK_EXCLUSION", "live chunk 数依赖 episode 长度"),
            ("CLAMPED_RANGE_HITS_100PCT", "夹取范围与 anchor 依赖 episode 长度"),
            ("NO_ANNOTATED_FRAME_DROPPED", "同上"),
            ("NO_DEAD_CHUNKS", "同上"),
            ("ANNOTATION_WITHIN_VIDEO", "越界判定需要视频长度"),
        ):
            ledger.add(_k, hard=requested, passed=False,
                       detail=f"NOT-MEASURED：{_why}（{frames_meta_status}）",
                       measured="NOT-MEASURED", threshold="需要 --frames-meta-root")
    else:
        ann_missing = join_missing
        ledger.add("EPISODE_JOIN_1TO1", hard=True,
                   passed=(ann_missing == 0 and len(episode_lengths) == n_files),
                   detail=("每个 annotation episode 必须能在 meta/episodes.jsonl 里找到同名 "
                           "episode_index，且两侧数量一致（dataloader 要按 (episode, frame) 联表）"),
                   measured={"annotation_files": n_files, "meta_episodes": len(episode_lengths),
                             "annotation_not_in_meta": ann_missing},
                   threshold={"annotation_not_in_meta": 0, "counts_equal": True},
                   evidence=evidence.get("episode_not_in_frames_meta", []))

        ledger.add("FRAME_COVERAGE_VS_VIDEO", hard=False,
                   passed=(uncovered_total <= args.max_uncovered_frames),
                   detail=("按 dataset.py:1347 的 `range(0, L, chunk_size)`（L 取自 "
                           "meta/episodes.jsonl 的整段视频长度，不是 valid_duration）采样时，"
                           "落在 Memory 标注覆盖之外的帧没有区间可查 ⇒ doc 3.4.3 的 memory_at() "
                           "会抛 IndexError，即文档要求判死的「未命中区间」"),
                   measured={"uncovered_frames": uncovered_total,
                             "head": unc_head, "tail": unc_tail,
                             "total_video_frames": total_video_frames,
                             "pct": round(100.0 * uncovered_total / max(1, total_video_frames), 4),
                             "episodes_affected": eps_uncovered,
                             "sampled_probes_over_full_video_range": vr_probes,
                             "sampled_probes_that_missed": vr_misses,
                             "sampled_miss_pct": round(100.0 * vr_misses / max(1, vr_probes), 3)},
                   threshold=args.max_uncovered_frames)

        # --- 与训练侧 memory_chunk_stats() 逐字段比对 -------------------------
        # 期望值从训练侧产物读，不写死在 gate 里：写死等于把当前状态固化成基线。
        stats = None
        stats_err = None
        if args.chunk_stats_json:
            try:
                raw = json.loads(Path(args.chunk_stats_json).read_text(encoding="utf-8"))
                for key in ("memory_chunk_stats", "chunk_stats", "stats"):
                    if isinstance(raw.get(key), dict):
                        raw = raw[key]
                        break
                if raw is None or any(raw.get(k) is None for k in
                                      ("memory_chunks_kept", "memory_dead_chunks_dropped",
                                       "memory_chunks_clipped")):
                    stats_err = f"字段缺失或为 None（memory 源未启用？）: {raw}"
                else:
                    stats = {k: int(raw[k]) for k in
                             ("memory_chunks_kept", "memory_dead_chunks_dropped",
                              "memory_chunks_clipped")}
            except Exception as exc:  # noqa: BLE001
                stats_err = f"{type(exc).__name__}: {exc}"

        live_measured = n_chunks_total - dead_chunks
        if stats is not None:
            agree = {
                "kept_plus_dropped_eq_total": (stats["memory_chunks_kept"]
                                               + stats["memory_dead_chunks_dropped"] == n_chunks_total),
                "dropped_eq_gate_dead": stats["memory_dead_chunks_dropped"] == dead_chunks,
                "kept_eq_gate_live": stats["memory_chunks_kept"] == live_measured,
                "clipped_eq_gate_partial": stats["memory_chunks_clipped"] == partial_chunks,
            }
            ledger.add("CHUNK_STATS_AGREE_WITH_TRAINING_SIDE", hard=True,
                       passed=all(agree.values()),
                       detail=("训练侧 memory_chunk_stats() 与 gate 独立算出的四个量必须逐个相等。"
                               "两边独立算、结果相等才有信息量；不等说明 chunk 判定口径不同"),
                       measured={"training_side": stats,
                                 "gate": {"total_chunks": n_chunks_total, "live": live_measured,
                                          "dead": dead_chunks, "clipped": partial_chunks},
                                 "agreement": agree},
                       threshold="逐个相等")
        else:
            ledger.add("CHUNK_STATS_AGREE_WITH_TRAINING_SIDE",
                       hard=bool(args.chunk_stats_json),
                       passed=False,
                       detail=("NOT-MEASURED：未提供 --chunk-stats-json ⇒ 没有跨实现对照。"
                               "没查到不等于通过，所以不渲染成 PASS"
                               if not args.chunk_stats_json
                               else f"提供了 --chunk-stats-json 但读不到有效字段 ⇒ NOT-MEASURED：{stats_err}"),
                       measured={"gate": {"total_chunks": n_chunks_total, "live": live_measured,
                                          "dead": dead_chunks, "clipped": partial_chunks},
                                 "error": stats_err},
                       threshold="逐个相等")

        # 处置已定：走「anchor 采样限制在标注覆盖范围内」，不把标注扩到整段视频。
        # 所以「覆盖 == 整段视频」不再是判据（降为观察项），判据换成下面四条。
        ledger.add("DEAD_CHUNK_COUNT_MANUAL_OVERRIDE",
                   hard=(args.dead_chunks_expected >= 0),
                   passed=(args.dead_chunks_expected < 0 or dead_chunks == args.dead_chunks_expected),
                   detail=("本 gate 独立算出的 dead chunk 数，必须与训练侧 run manifest 里的剔除计数"
                           "逐个相等。不等说明两边 chunk 判定口径不同（live_e 是否与 L 取 min、"
                           "边界用 <= 还是 <、末块 ce 是否 min(cs+250, L)）"),
                   measured=dead_chunks,
                   threshold=(args.dead_chunks_expected if args.dead_chunks_expected >= 0
                              else "report-only（未提供训练侧剔除计数）"))

        live_total = n_chunks_total - dead_chunks
        gnw = max(1, args.world_size * args.num_workers)
        per_worker_live = [len(range(g, live_total, gnw)) for g in range(gnw)] if live_total else [0]
        ledger.add("SHARDING_AFTER_DEAD_CHUNK_EXCLUSION", hard=True,
                   passed=(min(per_worker_live) >= 1),
                   detail=(f"剔除 dead chunk 后 {live_total} 个 chunk 分给 "
                           f"{args.world_size}×{args.num_workers}={gnw} 个 global worker，"
                           "每人至少 1 个；低于 1 会落进 dataset.py:1025 不含 rank 的 fallback"),
                   measured={"live_chunks": live_total, "global_workers": gnw,
                             "min_per_global_worker": min(per_worker_live),
                             "max_per_global_worker": max(per_worker_live),
                             "empty_global_workers": sum(1 for x in per_worker_live if x == 0),
                             "margin_x": round(live_total / gnw, 4)},
                   threshold={"min_per_global_worker": 1})

        ledger.add("CLAMPED_RANGE_HITS_100PCT", hard=True,
                   passed=(clamped_misses == 0 and clamped_empty == 0 and clamped_probes > 0
                           and no_anchor_chunks == 0),
                   detail=("剔除 dead chunk、anchor 夹到 [max(cs,cov_s), min(ce,live_e)) 之后，"
                           "**stride 对齐的真实 anchor** 的 frame → interval 命中率必须 100%；"
                           "任一未命中即 hard error。"
                           "⚠️ `chunks_without_aligned_anchor` 比训练语义**更严格**："
                           "训练侧 `_select_aligned_streaming_chunk` 只在某 worker 的**全部** chunk "
                           "都找不到对齐 anchor 时才 raise，单个 chunk 没有只会被跳过；"
                           "gate 对单个也报。stride=1 时该计数恒为 0，两者无差异"
                           "（文档 3.4.5「未命中区间 = hard error」在剔除之后的形式）"),
                   measured={"probes": clamped_probes, "misses": clamped_misses,
                             "anchor_stride": opts["anchor_stride"], "anchor_offset": opts["anchor_offset"],
                             "chunks_without_aligned_anchor": no_anchor_chunks,
                             "empty_clamped_ranges": clamped_empty,
                             "hit_rate_pct": round(100.0 * (clamped_probes - clamped_misses) / max(1, clamped_probes), 4)},
                   threshold={"misses": 0, "empty_clamped_ranges": 0},
                   evidence=evidence.get("clamped_range_miss", []))

        ledger.add("NO_ANNOTATED_FRAME_DROPPED", hard=True,
                   passed=(ann_outside_live == 0),
                   detail=("反方向的漏：剔除之后，落在 [cov_s, live_e) 里的标注帧不能有任何一帧"
                           "没有 live chunk 覆盖。跨边界 chunk 必须保留并夹取 anchor，"
                           "若改成「剔除部分覆盖的 chunk」这条会立刻报红"),
                   measured={"annotated_frames": annotated_frames,
                             "outside_live_chunks": ann_outside_live,
                             "partial_chunks": partial_chunks,
                             "annotated_frames_lost_if_partial_chunks_dropped": ann_lost_if_partial},
                   threshold=0)

        ledger.add("NO_DEAD_CHUNKS", hard=False,
                   passed=(dead_chunks <= args.max_dead_chunks),
                   detail=("整块落在标注覆盖之外的 chunk。处置已定为在 dataloader 侧剔除，"
                           "所以这里降为观察项；判定改由 DEAD_CHUNK_COUNT_AGREES 承担"),
                   measured={"dead_chunks": dead_chunks, "total_chunks": n_chunks_total,
                             "pct": round(100.0 * dead_chunks / max(1, n_chunks_total), 4),
                             "episodes_affected": eps_dead},
                   threshold=args.max_dead_chunks,
                   evidence=evidence.get("dead_chunk", []))

        ledger.add("ANNOTATION_WITHIN_VIDEO",
                   hard=(args.expect_overhang_episodes >= 0),
                   passed=(overhang_eps == args.expect_overhang_episodes
                           if args.expect_overhang_episodes >= 0
                           else (overhang == 0 and overhang_eps == 0)),
                   detail="标注区间不应伸出视频结尾（伸出不影响查找，但说明源标注与视频长度不一致）",
                   measured={"overhang_frames": overhang, "episodes": overhang_eps,
                             "by_last_interval_transition_type": dict(overhang_by_tt),
                             "expected_episodes": (args.expect_overhang_episodes
                                                   if args.expect_overhang_episodes >= 0 else None)},
                   threshold=0, evidence=evidence.get("annotation_overhang", []))

    # --- 分片 ---------------------------------------------------------------
    shard_units = {
        "episode": n_files,
        "memory_interval": total_intervals,
        "task_dir": len(task_dirs),
    }
    if n_chunks_total:
        shard_units["chunk_ceil_L_over_%d" % args.chunk_size] = n_chunks_total
    shard = sharding_report(
        world_size=args.world_size,
        num_workers=args.num_workers,
        unit_counts=shard_units,
    )
    chunk_key = "chunk_ceil_L_over_%d" % args.chunk_size
    # 当前代码里真正的分片单元是 chunk（dataset.py:287 -> :1334）；episode 只是备用视角。
    authoritative = chunk_key if chunk_key in shard["units"] else "episode"
    shard["authoritative_unit"] = authoritative
    ep_unit = shard["units"][authoritative]
    ledger.add("SHARDING_EVERY_WORKER_NONEMPTY", hard=True,
               passed=(ep_unit["min_per_global_worker"] >= 1),
               detail=(f"world_size={args.world_size} × num_workers={args.num_workers} = "
                       f"{shard['global_num_workers']} 个 global worker；按当前代码的分片单元 "
                       f"`{authoritative}`（dataset.py:1023 `range(g, N, W)`）每个 worker 至少 "
                       f"{ep_unit['min_per_global_worker']} 个。低于 1 会落进 dataset.py:1025 "
                       f"不含 rank 的 fallback ⇒ 不同 rank 拿到同一批数据"),
               measured=ep_unit, threshold={"min_per_global_worker": 1})
    ledger.add("SHARDING_TASK_LEVEL_WOULD_FAIL", hard=False,
               passed=(shard["units"]["task_dir"]["min_per_global_worker"] >= 1),
               detail=("若分片单位退化到 task 目录级别，会有 worker 拿到空集并落进不含 rank 的 "
                       "fallback ⇒ 不同 rank 拿到相同数据。这是提示项，不是当前实现的判定"),
               measured=shard["units"]["task_dir"], threshold={"min_per_global_worker": 1})

    # --- 组合型 primitive fallback 文案 -------------------------------------
    ledger.add("FALLBACK_PHRASE_TAXONOMY_COMPLETE", hard=True,
               passed=(violations.get("fallback_taxonomy_incomplete", 0) == 0),
               detail=(f"primitive 文案字段 {list(FALLBACK_SCOPE_FIELDS)} 里出现 ` + ` 却不属于已知的 "
                       f"{len(FALLBACK_PHRASES)} 个 fallback 串的条数必须为 0（说明分类表覆盖完整）"),
               measured=violations.get("fallback_taxonomy_incomplete", 0), threshold=0,
               evidence=evidence.get("fallback_taxonomy_incomplete", []))
    fb_top = sorted(fb_by_task.items(), key=lambda kv: -kv[1])[:10]
    ledger.add("FALLBACK_PHRASE_COUNT", hard=False,
               passed=(args.max_fallback_intervals < 0 or fb_intervals <= args.max_fallback_intervals),
               detail=("组合型 primitive fallback 文案命中的区间数（doc 3.4.6：训练前保留并记录 task 分布）；"
                       "阈值 -1 表示只报不判"),
               measured={"intervals": fb_intervals,
                         "pct_of_intervals": round(100.0 * fb_intervals / max(1, total_intervals), 2),
                         "model_target_text_hits": fb_model_target,
                         "episodes_affected": fb_episodes,
                         "top_tasks": fb_top},
               threshold=(args.max_fallback_intervals if args.max_fallback_intervals >= 0 else "report-only"))

    # --- 观察项（软）--------------------------------------------------------
    # 阈值来源：优先用**版本化登记基线**（按 manifest md5 索引），没有提供才退回命令行常量。
    # 直接把阈值改成当前实测值会让 gate 永远 PASS —— 那是跟着数据走的橡皮图章。
    reg_p99, reg_max = args.compact_p99_max, args.compact_max_max
    reg_src = "NOT-MEASURED：未提供版本化登记基线，退回命令行常量"
    reg_missing = not args.distribution_baselines
    reg_entry: dict[str, Any] | None = None
    reg_unknown = False
    if args.distribution_baselines:
        rp = Path(args.distribution_baselines)
        if not rp.exists() or rp.stat().st_size == 0:
            reg_unknown = True
            reg_src = f"登记基线文件不可用: {rp}"
        else:
            try:
                table = json.loads(rp.read_text(encoding="utf-8")).get("by_manifest_md5") or {}
            except Exception as exc:  # noqa: BLE001
                reg_unknown = True
                reg_src = f"登记基线解析失败: {type(exc).__name__}: {exc}"
            else:
                key = fingerprint.get("manifest_md5")
                reg_entry = table.get(key)
                if reg_entry is None:
                    reg_unknown = True
                    reg_src = (f"manifest md5 {key} 未登记 ⇒ fail-closed，NOT-MEASURED。"
                               f"已登记版本: {sorted(v.get('label', k) for k, v in table.items())}")
                elif str(reg_entry.get("status")) != "accepted":
                    # status 此前只被打印、不参与判定 ⇒ 拿已被取代的版本跑也会 PASS。
                    reg_unknown = True
                    reg_src = (f"登记基线 [{reg_entry.get('label')}] 的 status="
                               f"{reg_entry.get('status')!r}，不是 accepted ⇒ 拒绝用它放行")
                    reg_entry = None
                else:
                    reg_p99 = int(reg_entry["fixed_compact_memory_p99_max"])
                    reg_max = int(reg_entry["fixed_compact_memory_max_max"])
                    reg_src = f"登记基线 [{reg_entry.get('label')}] status=accepted"

    ledger.add("COMPACT_MEMORY_DISTRIBUTION",
               hard=reg_unknown,
               passed=(not reg_unknown and not reg_missing
                       and cmp_.get("p99", 1 << 30) <= reg_p99
                       and cmp_.get("max", 1 << 30) <= reg_max),
               detail=("fixed compact Memory 字段的 token P99 / Max（观察漂移，不单独截断该字段）。"
                       "阈值来自该数据版本的登记基线；未登记的版本记 NOT-MEASURED 而不是"
                       "拿别的版本的值去卡，也不自动学习成当前值"),
               measured={"p99": cmp_.get("p99"), "max": cmp_.get("max"),
                         "threshold_source": reg_src,
                         "registry_note": (reg_entry or {}).get("note")},
               threshold=({"p99": reg_p99, "max": reg_max} if not reg_unknown else "NOT-MEASURED"))

    # --- subtask_max_len 作用域守卫 ------------------------------------------
    if not args.config_scope_expect:
        ledger.add("SUBTASK_MAX_LEN_SCOPE", hard=False, passed=False,
                   detail="NOT-MEASURED：未提供 --config-scope-expect ⇒ 作用域完全没查。"
                          "此前这条在缺参时整条不加入报告，看起来像一切正常",
                   measured="NOT-MEASURED", threshold="需要 --config-scope-expect")
    if args.config_scope_expect:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import config_scope_guard as _csg

            resolved, class_default, _diag = _csg.resolve_configs()
            exp = json.loads(Path(args.config_scope_expect).read_text(encoding="utf-8"))
            bad = []
            if exp.get("class_default_subtask_max_len") is not None and \
                    class_default != exp["class_default_subtask_max_len"]:
                bad.append(f"class_default {class_default} != {exp['class_default_subtask_max_len']}")
            for name, want in (exp.get("per_config_subtask_max_len") or {}).items():
                got = resolved.get(name, {}).get("subtask_max_len")
                if got != want:
                    bad.append(f"{name}: {got} != {want}")
            if exp.get("require_exact_config_set", True):
                extra = sorted(set(resolved) - set(exp.get("per_config_subtask_max_len") or {}))
                if extra:
                    bad.append(f"未登记的 Pi05SubtaskConfig: {extra}")
            ledger.add("SUBTASK_MAX_LEN_SCOPE", hard=True, passed=not bad,
                       detail=("subtask_max_len 定义在 Pi05SubtaskConfig 上，而该类被在飞的 "
                               "annotations_skill / skillbridge 实验共用。抬高只能在 MoMA 配置"
                               "实例上覆盖；改类默认值会静默改掉别人实验的 padding"),
                       measured={"class_default": class_default,
                                 "per_config": {k: v["subtask_max_len"] for k, v in resolved.items()}},
                       threshold=exp, evidence=bad[:12])
        except Exception as exc:  # noqa: BLE001
            ledger.add("SUBTASK_MAX_LEN_SCOPE", hard=True, passed=False,
                       detail=f"请求了作用域守卫但解析不到训练配置 ⇒ NOT-MEASURED：{type(exc).__name__}: {exc}",
                       measured="NOT-MEASURED", threshold="可解析")

    # --- 与上一次报告的差异（数据会被重新生成，涨了多少必须被看到）-----------
    baseline_diff: dict[str, Any] = {}
    if args.baseline_report:
        bp = Path(args.baseline_report)
        base = None
        err = None
        # `cp` 可以退出码 0、文件存在、内容为空（md5 d41d8cd9...）。所以「文件存在」
        # 不是可读的证据：先断言字节数 > 0，再断言能解析成含 checks 的 dict。
        if not bp.exists():
            err = "文件不存在"
        elif bp.stat().st_size == 0:
            err = "文件存在但是 0 字节（cp 可能退出码 0 却产出空文件）"
        else:
            try:
                cand_base = json.loads(bp.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            else:
                if not isinstance(cand_base, dict) or "checks" not in cand_base:
                    err = f"解析成功但不是 gate 报告（缺 checks 字段，keys={list(cand_base)[:8]}）"
                else:
                    base = cand_base
        if base is None:
            ledger.add("BASELINE_DIFF", hard=True, passed=False,
                       detail=f"提供了 --baseline-report 但不可用：{err} ⇒ NOT-MEASURED，不当成没差异",
                       measured={"path": str(bp), "size": bp.stat().st_size if bp.exists() else None,
                                 "error": err},
                       threshold="非空且可解析为 gate 报告")
        else:
            def pick(d, *path, default=None):  # noqa: D401
                cur = d
                for k in path:
                    cur = cur.get(k) if isinstance(cur, dict) else None
                    if cur is None:
                        return default
                return cur

            # 旧报告没有 task_text_source 字段，那时用的一律是 annotation 的 task_name
            _base_task_src = (pick(base, "thresholds", "task_text_source") or "annotation")
            pairs = [
                ("baseline_task_text_source", _base_task_src, args.task_text_source),
                ("intervals", pick(base, "counts", "memory_intervals"), total_intervals),
                ("planner_target_max_production", pick(base, "token_lengths",
                                                       "planner_target_text__production_tokenize_memory",
                                                       "max"), prod_max),
                ("planner_target_max_doc_path", pick(base, "token_lengths",
                                                     "planner_target_text__training_tokenizer", "max"), tgt_max),
                ("planner_target_p99", pick(base, "token_lengths",
                                            "planner_target_text__training_tokenizer", "p99"), tgt.get("p99")),
                # prompt 只在**口径相同**时才比。旧报告用 annotation 的 task_name、
                # 新默认用 LeRobot 的 tasks[0]，两者差最多 100 个 token；直接相减会把
                # 口径变更渲染成数据变化 —— 这个坑本轮已经踩过一次（190 vs 161 的 -29）。
                ("prompt_max" + ("" if _base_task_src == args.task_text_source
                                 else f"__NOT-COMPARABLE(baseline={_base_task_src})"),
                 (pick(base, "token_lengths", f"prefix_prompt__active_source_{args.task_text_source}", "max")
                  or (pick(base, "token_lengths", "prefix_prompt__training_tokenizer_worst_case_state", "max")
                      if _base_task_src == args.task_text_source else None)),
                 pmt.get("max")),
                ("compact_p99", pick(base, "token_lengths", "fixed_compact_memory_field", "p99"), cmp_.get("p99")),
                ("compact_max", pick(base, "token_lengths", "fixed_compact_memory_field", "max"), cmp_.get("max")),
                ("dead_chunks", pick(base, "frames_meta", "dead_chunks"), dead_chunks),
                ("clipped_chunks", pick(base, "frames_meta", "partial_boundary_chunks"), partial_chunks),
                ("uncovered_frames", pick(base, "frames_meta", "uncovered_frames_total"), uncovered_total),
                ("annotation_overhang_frames", pick(base, "frames_meta", "annotation_overhang_frames"), overhang),
                ("annotation_overhang_episodes", pick(base, "frames_meta",
                                                      "episodes_with_annotation_overhang"), overhang_eps),
                ("fallback_intervals", pick(base, "fallback_phrases", "intervals"), fb_intervals),
                ("bridge_anomalies", pick(base, "violations", "intra_bridge_primitive_changed"), bridge_anomalies),
            ]
            for name, was, now in pairs:
                delta = None
                if isinstance(was, (int, float)) and isinstance(now, (int, float)):
                    delta = now - was
                baseline_diff[name] = {"was": was, "now": now, "delta": delta}
            not_comparable = {k: v for k, v in baseline_diff.items() if "NOT-COMPARABLE" in k}
            changed = {k: v for k, v in baseline_diff.items()
                       if v["delta"] not in (None, 0) and "NOT-COMPARABLE" not in k}
            ledger.add("BASELINE_DIFF", hard=False, passed=True,
                       detail=("与上一次报告的逐项差异。这里永远 PASS —— 它的作用是把变化摆出来，"
                               "而不是判定；真正的判定在各自的检查里"),
                       measured={"baseline_generated_at": base.get("generated_at"),
                                 "baseline_manifest_mtime": pick(base, "data_fingerprint", "manifest_mtime_iso"),
                                 "changed": changed,
                                 "not_comparable": not_comparable,
                                 "unchanged_keys": sorted(set(baseline_diff) - set(changed)
                                                          - set(not_comparable))},
                       threshold="report-only")
            # 数据换没换：manifest 指纹必须变，否则量的还是旧数据
            base_manifest = pick(base, "data_fingerprint", "manifest_md5")
            now_manifest = fingerprint.get("manifest_md5")
            base_probe = pick(base, "data_fingerprint", "probe_digest")
            now_probe = fingerprint.get("probe_digest")
            ledger.add("DATA_ACTUALLY_CHANGED", hard=False,
                       passed=(base_manifest != now_manifest or base_probe != now_probe),
                       detail=("与基线相比 manifest md5 或抽样文件摘要必须至少有一个变化，"
                               "否则说明本次量的还是同一份数据（用于确认重生成真的落地了）"),
                       measured={"manifest_md5_changed": base_manifest != now_manifest,
                                 "probe_digest_changed": base_probe != now_probe,
                                 "baseline_manifest_md5": base_manifest, "now_manifest_md5": now_manifest},
                       threshold="至少一项变化")

    # --- 汇总 ---------------------------------------------------------------
    hard_fail = ledger.hard_failures
    soft_fail = ledger.soft_failures
    verdict = "PASS" if not hard_fail and (not soft_fail or not args.fail_on_soft) else "FAIL"

    report = {
        "verdict": verdict,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_seconds": round(scan_secs, 1),
        "data_fingerprint": fingerprint,
        "tokenizer_fingerprint": fp_tok,
        "state_token_cost": {"worst_case_tokens": state_tokens, "variants": state_variants,
                             "action_dim": args.action_dim},
        "thresholds": {
            "expect_tasks": args.expect_tasks,
            "expect_episodes_per_task": args.expect_episodes_per_task,
            "expect_episodes": args.expect_episodes,
            "expect_intervals": args.expect_intervals,
            "mode": args.mode,
            "task_text_source": args.task_text_source,
            "expect_tokenizer_md5": args.expect_tokenizer_md5,
            "expect_vocab_size": args.expect_vocab_size,
            "expect_overhang_episodes": args.expect_overhang_episodes,
            "config_scope_expect": str(args.config_scope_expect) if args.config_scope_expect else None,
            "subtask_max_len": args.subtask_max_len,
            "prompt_max_len": args.prompt_max_len,
            "action_dim": args.action_dim,
            "world_size": args.world_size,
            "num_workers": args.num_workers,
            "frames_meta_root": str(args.frames_meta_root),
            "chunk_size": args.chunk_size,
        "anchor_stride": args.anchor_stride if args.anchor_stride is not None else 1,
        "anchor_offset": args.anchor_offset if args.anchor_offset is not None else 0,
            "max_uncovered_frames": args.max_uncovered_frames,
            "max_dead_chunks": args.max_dead_chunks,
            "dead_chunks_expected": args.dead_chunks_expected,
            "baseline_target_max": args.baseline_target_max,
            "chunk_stats_json": str(args.chunk_stats_json) if args.chunk_stats_json else None,
            "baseline_report": str(args.baseline_report) if args.baseline_report else None,
            "max_bridge_anomalies": args.max_bridge_anomalies,
            "max_fallback_intervals": args.max_fallback_intervals,
            "compact_p99_max": args.compact_p99_max,
            "compact_max_max": args.compact_max_max,
            "sample_episodes": args.sample_episodes,
            "frames_per_episode": args.frames_per_episode,
            "seed": args.seed,
            "file_stride": args.file_stride,
        },
        "counts": {
            "task_dirs": len(task_dirs),
            "episode_files": n_files,
            "readable_episodes": readable,
            "memory_intervals": total_intervals,
            "deep_checked_episodes": deep_done,
            "deep_probed_frames": deep_frames,
        },
        "candidate_limit_table": limit_table,
        "token_length_by_transition_type": len_by_tt,
        "two_path_delta_value_counts": {str(k): v for k, v in sorted(path_deltas.items())},
        "token_lengths": {
            "planner_target_text__production_tokenize_memory": prod,
            "planner_target_text__training_tokenizer": tgt,
            "model_target_text__training_tokenizer": mtgt,
            "prefix_prompt__active_source_%s" % args.task_text_source: pmt,
            "prefix_prompt__other_source": pmt_other,
            "fixed_compact_memory_field": cmp_,
        },
        "filelist_audit": filelist_audit,
        "frames_meta": {
            "status": frames_meta_status,
            "sampled_probes_over_full_video_range": vr_probes,
            "sampled_probes_that_missed": vr_misses,
            "episodes_with_annotation_overhang": overhang_eps,
            "path": str(frames_meta_path) if frames_meta_path else None,
            "meta_episodes": len(episode_lengths) if episode_lengths is not None else None,
            "total_video_frames": total_video_frames,
            "chunk_size": args.chunk_size,
        "anchor_stride": args.anchor_stride if args.anchor_stride is not None else 1,
        "anchor_offset": args.anchor_offset if args.anchor_offset is not None else 0,
            "total_chunks": n_chunks_total,
            "uncovered_frames_head": unc_head,
            "uncovered_frames_tail": unc_tail,
            "uncovered_frames_total": uncovered_total,
            "episodes_with_uncovered_frames": eps_uncovered,
            "dead_chunks": dead_chunks,
            "episodes_with_dead_chunks": eps_dead,
            "annotation_overhang_frames": overhang,
            "annotation_overhang_by_last_transition_type": dict(overhang_by_tt),
            "live_chunks": n_chunks_total - dead_chunks,
            "partial_boundary_chunks": partial_chunks,
            "empty_clamped_ranges": clamped_empty,
            "annotated_frames": annotated_frames,
            "annotated_frames_outside_live_chunks": ann_outside_live,
            "annotated_frames_lost_if_partial_chunks_dropped": ann_lost_if_partial,
            "clamped_probes": clamped_probes,
            "clamped_misses": clamped_misses,
        },
        "transition_counts": dict(transitions),
        "fallback_phrases": {
            "criterion": "substring match on " + str(list(FALLBACK_SCOPE_FIELDS))
                         + " + completed_primitive_stack + model_target_text",
            "phrases": list(FALLBACK_PHRASES),
            "intervals": fb_intervals,
            "model_target_text_hits": fb_model_target,
            "episodes_affected": fb_episodes,
            "by_task": dict(sorted(fb_by_task.items(), key=lambda kv: -kv[1])),
        },
        "violations": dict(violations),
        "leak_kinds": dict(leak_kinds),
        "evidence": evidence,
        "sharding": shard,
        "baseline_diff": baseline_diff,
        "checks": [c.as_dict() for c in ledger.checks],
        "hard_failures": [c.key for c in hard_fail],
        "soft_failures": [c.key for c in soft_fail],
    }

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                            encoding="utf-8")

    # --- 人读输出 -----------------------------------------------------------
    print()
    print("=" * 78)
    print(f"MoMA-VLA PRETRAIN GATE  verdict={verdict}  mode={args.mode}  "
          f"task_text_source={args.task_text_source}")
    print("=" * 78)
    print(f"data_root      : {root}")
    print(f"frames_meta    : {frames_meta_status} (root={args.frames_meta_root!r})")
    print(f"topology       : world_size={args.world_size} num_workers={args.num_workers} "
          f"(必传，取自那次 run 的生效值)")
    print(f"manifest       : mtime={fingerprint.get('manifest_mtime_iso')} "
          f"episodes={fingerprint.get('manifest_episodes')} intervals={fingerprint.get('manifest_intervals')}")
    print(f"tokenizer      : md5={fp_tok['sentencepiece_model_md5']} vocab={fp_tok['vocab_size']} "
          f"sentencepiece={fp_tok['sentencepiece_lib_version']} py={fp_tok['python']}")
    print(f"scanned        : {n_files} episodes / {total_intervals} intervals in {scan_secs:.0f}s")
    print()
    for c in ledger.checks:
        tag = "PASS" if c.passed else ("HARD-FAIL" if c.hard else "WARN")
        print(f"  [{tag:9s}] {c.key}")
        print(f"              measured={json.dumps(c.measured, ensure_ascii=False)}  "
              f"threshold={json.dumps(c.threshold, ensure_ascii=False)}")
        if not c.passed and c.evidence:
            for m in c.evidence[:4]:
                print(f"              ! {m}")
    print()
    print("token lengths (training-side SubtaskTokenizer, BOS+text+EOS, untruncated):")
    for name, d in report["token_lengths"].items():
        print(f"  {name:52s} {json.dumps(d)}")
    print()
    print("sharding (authoritative unit = %s):" % shard.get("authoritative_unit"))
    print(f"  global_num_workers = {shard['world_size']} x {shard['num_workers_per_rank']} = {shard['global_num_workers']}")
    for name, u in shard["units"].items():
        print(f"  unit={name:16s} n={u['n_units']:>7d} min/worker={u['min_per_global_worker']:>4d} "
              f"empty_workers={u['empty_global_workers']:>4d} margin={u['margin_x']}x "
              f"fallback={'TRIGGERED' if u['fallback_triggered'] else 'no'}")
    if baseline_diff:
        print()
        print("baseline diff (was -> now, delta):")
        for k, v in baseline_diff.items():
            mark = "  " if v["delta"] in (None, 0) else "* "
            print(f"  {mark}{k:32s} {v['was']} -> {v['now']}  ({v['delta']:+d})"
                  if isinstance(v["delta"], int) else
                  f"  {mark}{k:32s} {v['was']} -> {v['now']}  (n/a)")
    print()
    print(f"GATE_VERDICT: {verdict} hard_failures={len(hard_fail)} soft_failures={len(soft_fail)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
