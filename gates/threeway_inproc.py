#!/usr/bin/env python3
"""三向验证 + 变异测试，单进程内跑完。

为什么不是 shell 脚本反复起子进程
---------------------------------
实测 `import openpi.models.tokenizer` 在这台机（8 核 cgroup，多 worker 争抢，
load 80+）要 **294 秒**，而一次子集扫描本身只要几十秒。二十次子进程调用等于
98 分钟纯 import。这里把 gate 当模块导入一次，然后在同一个进程里用不同 argv
反复调用 `main()`，并用 `--file-list` 省掉每次 47 秒的 glob。

三向：每条阈值都要 ① 当前数据通过 ② 放宽仍通过 ③ 调紧时精确命中**事先算好**
的条数。预期条数由独立于 gate 的一次性统计给出，先落盘再跑 gate。

用法：
    PYTHONPATH=<worktree>/src <env-python> threeway_inproc.py [输出目录]
"""

from __future__ import annotations

import os

# 与 gate 同样的线程约束，必须在任何科学计算库之前设置。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "threeway"
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/"
            "fixed_compact_memory_annotations")
FRAMES = Path("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos")
STRIDE = 10
# 配额是 8 核（cpu.cfs_quota_us=800000 / period=100000），nproc 报的 119 是宿主机核数。
# 数据侧的 API run 也占配额，所以这里只用 2。
JOBS = int(os.environ.get("MOMA_JOBS", "2"))
# 全量扫描由 Master 排队，默认不在这里跑（section 6）。
SKIP_FULL = os.environ.get("MOMA_SKIP_FULL", "0") == "1"

sys.path.insert(0, str(HERE))

ROWS: list[dict] = []


def record(check, direction, arg, expected, observed, verdict):
    ROWS.append({"check": check, "direction": direction, "arg": arg,
                 "expected": expected, "observed": observed, "verdict": verdict})
    print(f"    [{verdict}] {check} | {direction} | {arg} | expected={expected} observed={observed}",
          flush=True)


def env_reading() -> str:
    """每个测量数字都要带环境读数：重启前那些 import 耗时是多 worker 争抢时测的，
    不带环境读数就分不清是环境还是文件系统固有成本。"""
    try:
        load = Path("/proc/loadavg").read_text().split()[:3]
    except Exception:  # noqa: BLE001
        load = ["?"]
    try:
        threads = sum(1 for _ in Path("/proc").glob("[0-9]*/task/[0-9]*"))
        procs = sum(1 for _ in Path("/proc").glob("[0-9]*"))
    except Exception:  # noqa: BLE001
        threads = procs = -1
    return f"load={'/'.join(load)} procs={procs} threads={threads}"


def banner(t):
    print("\n" + "=" * 70 + f"\n {t}\n  [env] {env_reading()}\n" + "=" * 70, flush=True)


def fingerprint() -> dict:
    files = sorted(DATA.glob("task-*/episode_*.json"))
    h = hashlib.md5()
    for p in files:
        st = p.stat()
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return {
        "n_files": len(files),
        "tree_digest_path_size_mtime": h.hexdigest(),
        "manifest_md5": hashlib.md5((DATA / "manifest.json").read_bytes()).hexdigest(),
        "sample_md5": {str(p.relative_to(DATA)): hashlib.md5(p.read_bytes()).hexdigest()
                       for p in files[::1500]},
    }


t_start = time.time()
banner("0. 数据指纹（跑之前）")
FP_BEFORE = fingerprint()
(OUT / "fingerprint_before.json").write_text(json.dumps(FP_BEFORE, indent=2, sort_keys=True))
print(json.dumps({k: v for k, v in FP_BEFORE.items() if k != "sample_md5"}, indent=2))

# 文件清单：这台机 glob 一万个文件要 47 秒，所以缓存起来。
# 但**「文件存在」不是「文件完整」**：上一轮 harness 被 kill 时正在写这个清单，
# 留下了一份 6,416 行的截断文件，而 `exists()` 照样为真。所以
#   ① 一律写临时文件再 os.replace（原子），被 kill 只会留下 .tmp
#   ② 复用之前先拿它的行数与本轮 fingerprint 的 n_files 对账，不符就重建
FILELIST = OUT / "filelist_all.txt"
N_TRUTH = FP_BEFORE["n_files"]


def _write_list(path: Path, items) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(str(x) for x in items))
    os.replace(tmp, path)


def _load_list(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    got = [x for x in path.read_text().split() if x.strip()]
    return got or None


ALL_FILES = _load_list(FILELIST)
if ALL_FILES is None or len(ALL_FILES) != N_TRUTH:
    why = "不存在/为空" if ALL_FILES is None else f"行数 {len(ALL_FILES)} != {N_TRUTH}（截断）"
    print(f"  重建文件清单（{why}）", flush=True)
    ALL_FILES = [str(x) for x in sorted(DATA.glob("task-*/episode_*.json"))]
    _write_list(FILELIST, ALL_FILES)
SUBSET = ALL_FILES[::STRIDE]
FILELIST_SUB = OUT / f"filelist_stride{STRIDE}.txt"
_write_list(FILELIST_SUB, SUBSET)
n_tasks = len({Path(x).parent.name for x in SUBSET})
print(f"file list: all={len(ALL_FILES)} subset(stride={STRIDE})={len(SUBSET)} "
      f"tasks_in_subset={n_tasks}", flush=True)
assert len(ALL_FILES) == N_TRUTH, f"CANNOT-ASSESS: 清单 {len(ALL_FILES)} != 实际 {N_TRUTH}"
assert n_tasks == 50, f"CANNOT-ASSESS: 子集只覆盖 {n_tasks} 个 task，抽样有偏"

banner("1. 预期条数（独立统计，先落盘再跑 gate）")
EXP_PATH = OUT / "expected.json"
if EXP_PATH.exists():
    EXPECTED = json.loads(EXP_PATH.read_text())
    print("(复用已落盘的 expected.json)")
else:
    rc = subprocess.run(
        [sys.executable, str(HERE / "_expected_counts.py"), str(FILELIST_SUB), str(FRAMES), str(EXP_PATH)],
        capture_output=True, text=True)
    if rc.returncode != 0:
        print(rc.stdout[-3000:]); print(rc.stderr[-3000:]); raise SystemExit("预期统计失败 ⇒ CANNOT-ASSESS")
    EXPECTED = json.loads(EXP_PATH.read_text())
print(json.dumps({k: v for k, v in EXPECTED.items() if k != "drop_one_phrase_exposes"}, indent=2))
print("两条口径的逐行差值取值集合（前 8 个）:", EXPECTED.get("delta_two_paths_distinct_values"))
print("drop_one_phrase_exposes:", json.dumps(EXPECTED["drop_one_phrase_exposes"], ensure_ascii=False, indent=2))

banner("2. 导入 gate 模块（只此一次）")
t0 = time.time()
import moma_pretrain_gate as G  # noqa: E402
print(f"import moma_pretrain_gate: {time.time() - t0:.1f}s", flush=True)

# 上限现在必传。三向验证里它们会被逐项覆盖；这里给的是「当前生效值」基线。
SUBTASK_MAX_LEN = os.environ.get("MOMA_SUBTASK_MAX_LEN", "160")
PROMPT_MAX_LEN = os.environ.get("MOMA_PROMPT_MAX_LEN", "512")

BASE = ["--data-root", str(DATA), "--frames-meta-root", str(FRAMES),
        "--world-size", "32", "--num-workers", "16",
        "--subtask-max-len", SUBTASK_MAX_LEN, "--prompt-max-len", PROMPT_MAX_LEN,
        "--file-list", str(FILELIST_SUB), "--skip-count-checks",
        "--sample-episodes", "50", "--jobs", str(JOBS),
        "--max-uncovered-frames", "999999999", "--max-dead-chunks", "999999999"]


def run_gate(extra, tag):
    out = OUT / f"{tag}.json"
    argv = BASE + extra + ["--out", str(out)]
    buf = io.StringIO()
    t = time.time()
    try:
        with redirect_stdout(buf):
            rc = G.main(argv)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    (OUT / f"{tag}.log").write_text(buf.getvalue())
    print(f"  [{tag}] rc={rc} {time.time() - t:.0f}s  [env] {env_reading()}", flush=True)
    rep = json.loads(out.read_text()) if out.exists() else None
    return rc, rep


def check_of(rep, key):
    for c in rep["checks"]:
        if c["key"] == key:
            return c
    return None


banner("3. 自检（正对照）")
buf = io.StringIO()
with redirect_stdout(buf):
    st_rc = G.main(["--self-test", "--out", str(OUT / "selftest.json")])
(OUT / "selftest.log").write_text(buf.getvalue())
st = json.loads((OUT / "selftest.json").read_text())
for c in st["cases"]:
    print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['case']} -> {c['expect'] or '(expect no violations)'}")
record("SELF_TEST", "positive_control", "--self-test", "all pass",
       f"{st['passed']}/{st['total']}", "MATCH" if st["all_passed"] else "MISMATCH")

banner(f"4.1 subtask_max_len（当前生效值 {SUBTASK_MAX_LEN}）")
for thr in (int(SUBTASK_MAX_LEN), 256, 90, 83):
    exp = EXPECTED.get(f"target_over_{thr}")
    if exp is None:
        print(f"  跳过 --subtask-max-len {thr}：预期文件里没有这个阈值的独立统计")
        continue
    _, rep = run_gate(["--subtask-max-len", str(thr)], f"subtask_{thr}")
    obs = rep["violations"].get("prod_target_over_subtask_max_len", 0)
    record("TARGET_OVER_MAX_COUNT",
           "current_or_relaxed" if thr >= int(SUBTASK_MAX_LEN) else "tightened",
           f"--subtask-max-len {thr}", exp, obs, "MATCH" if obs == exp else "MISMATCH")

banner(f"4.2 prompt_max_len（当前生效值 {PROMPT_MAX_LEN}）")
for thr in (int(PROMPT_MAX_LEN), 1024, 180, 162):
    exp = EXPECTED.get(f"prompt_over_{thr}")
    if exp is None:
        print(f"  跳过 --prompt-max-len {thr}：预期文件里没有这个阈值的独立统计")
        continue
    _, rep = run_gate(["--prompt-max-len", str(thr)], f"prompt_{thr}")
    obs = rep["violations"].get("prompt_over_max_len", 0)
    record("PROMPT_TOKEN_BUDGET",
           "current_or_relaxed" if thr >= int(PROMPT_MAX_LEN) else "tightened",
           f"--prompt-max-len {thr}", exp, obs, "MATCH" if obs == exp else "MISMATCH")

banner("4.3 分片：剔除 dead chunk 后每个 global worker 至少 1 个")
LIVE = EXPECTED["live_chunks"]
for ws, nw, want in ((32, 16, "PASS"), (8, 1, "PASS"), (LIVE, 1, "PASS"),
                     (LIVE + 1, 1, "FAIL"), (LIVE * 2, 1, "FAIL")):
    _, rep = run_gate(["--world-size", str(ws), "--num-workers", str(nw)], f"shard_{ws}x{nw}")
    c = check_of(rep, "SHARDING_AFTER_DEAD_CHUNK_EXCLUSION")
    got = "PASS" if c["passed"] else "FAIL"
    record("SHARDING_AFTER_DEAD_CHUNK_EXCLUSION",
           "current_or_relaxed" if want == "PASS" else "tightened",
           f"--world-size {ws} --num-workers {nw} (live={LIVE})",
           want, f"{got}(min={c['measured']['min_per_global_worker']})",
           "MATCH" if got == want else "MISMATCH")

banner("4.4 与训练侧 chunk stats 的一致性")
_, probe = run_gate([], "stats_probe")
gate_dead = probe["frames_meta"]["dead_chunks"]
gate_live = probe["frames_meta"]["live_chunks"]
gate_clip = probe["frames_meta"]["partial_boundary_chunks"]
print(f"  gate 自测（子集）: total={probe['frames_meta']['total_chunks']} live={gate_live} "
      f"dead={gate_dead} clipped={gate_clip}")
print(f"  独立统计对照     : chunks={EXPECTED['chunks']} live={EXPECTED['live_chunks']} "
      f"dead={EXPECTED['dead_chunks']}")
record("CHUNK_COUNTS_VS_INDEPENDENT", "cross_check", "gate vs 独立统计",
       f"dead={EXPECTED['dead_chunks']},live={EXPECTED['live_chunks']}",
       f"dead={gate_dead},live={gate_live}",
       "MATCH" if (gate_dead == EXPECTED["dead_chunks"] and gate_live == EXPECTED["live_chunks"]) else "MISMATCH")
for name, dead_val, want in (("right", gate_dead, "PASS"), ("wrong", gate_dead + 6, "FAIL")):
    sp = OUT / f"stats_{name}.json"
    sp.write_text(json.dumps({"memory_chunks_kept": gate_live,
                              "memory_dead_chunks_dropped": dead_val,
                              "memory_chunks_clipped": gate_clip}))
    _, rep = run_gate(["--chunk-stats-json", str(sp)], f"stats_cmp_{name}")
    c = check_of(rep, "CHUNK_STATS_AGREE_WITH_TRAINING_SIDE")
    got = "PASS" if c["passed"] else "FAIL"
    record("CHUNK_STATS_AGREE_WITH_TRAINING_SIDE",
           "current" if name == "right" else "injected_mismatch",
           f"dropped={dead_val}", want, got, "MATCH" if got == want else "MISMATCH")

banner("4.5 计数类（子集真值 vs 偏离 1）")
n_sub, iv_sub = EXPECTED["files"], EXPECTED["intervals"]
for ep, iv, want in ((n_sub, iv_sub, "PASS"), (n_sub, iv_sub - 1, "FAIL"),
                     (n_sub - 1, iv_sub, "FAIL"), (n_sub, -1, "PASS")):
    argv = [a for a in BASE if a != "--skip-count-checks"] + \
           ["--expect-episodes", str(ep), "--expect-intervals", str(iv),
            "--expect-tasks", "50", "--expect-episodes-per-task", "20",
            "--out", str(OUT / f"count_{ep}_{iv}.json")]
    buf = io.StringIO()
    t = time.time()
    try:
        with redirect_stdout(buf):
            rc = G.main(argv)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    (OUT / f"count_{ep}_{iv}.log").write_text(buf.getvalue())
    rep = json.loads((OUT / f"count_{ep}_{iv}.json").read_text())
    hard = [k for k in rep["hard_failures"] if k in ("EPISODE_FILE_COUNT", "INTERVAL_TOTAL")]
    got = "PASS" if not hard else "FAIL"
    print(f"  [count_{ep}_{iv}] rc={rc} {time.time() - t:.0f}s hard={hard}")
    record("COUNT_CHECKS", "current_or_relaxed" if want == "PASS" else "tightened",
           f"--expect-episodes {ep} --expect-intervals {iv}", want, got,
           "MATCH" if got == want else "MISMATCH")

banner("4.6 fallback 分类表完备性：逐个删串")
drops = EXPECTED["drop_one_phrase_exposes"]
killers = [k for k, v in drops.items() if v > 0]
print(f"  完整分类表下未归类 = {EXPECTED['fallback_taxonomy_incomplete']}（必须为 0）")
for k, v in drops.items():
    print(f"    drop {k!r:46s} -> {v:6d}   {'可单独杀死该检查' if v else '被更短的串吸收，删了不暴露'}")
record("FALLBACK_PHRASE_TAXONOMY_COMPLETE", "current", "full taxonomy", 0,
       EXPECTED["fallback_taxonomy_incomplete"],
       "MATCH" if EXPECTED["fallback_taxonomy_incomplete"] == 0 else "MISMATCH")
record("FALLBACK_PHRASE_TAXONOMY_COMPLETE", "mutated_taxonomy", "drop each of 8 phrases",
       ">=1 phrase kills it", f"{len(killers)}/8 kill",
       "MATCH" if killers else "MISMATCH")

banner("5. 变异测试")
MUT = OUT / "mutant_root"
if MUT.exists():
    shutil.rmtree(MUT)
(MUT / "task-0000").mkdir(parents=True)
SRC = DATA / "task-0000" / "episode_00000010.json"
md5_before = hashlib.md5(SRC.read_bytes()).hexdigest()
shutil.copy2(SRC, MUT / "task-0000" / "episode_00000010.json")
shutil.copy2(DATA / "task-0000" / "episode_00000020.json", MUT / "task-0000" / "episode_00000020.json")
shutil.copy2(DATA / "manifest.json", MUT / "manifest.json")

MARGV = ["--data-root", str(MUT), "--frames-meta-root", "", "--world-size", "1", "--num-workers", "1",
         "--subtask-max-len", SUBTASK_MAX_LEN, "--prompt-max-len", PROMPT_MAX_LEN,
         "--skip-count-checks", "--sample-episodes", "2", "--jobs", "2"]


def run_mut(tag):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            G.main(MARGV + ["--out", str(OUT / f"{tag}.json")])
    except SystemExit:
        pass
    (OUT / f"{tag}.log").write_text(buf.getvalue())
    return json.loads((OUT / f"{tag}.json").read_text())


rep = run_mut("mutant_clean")
c = check_of(rep, "INTERVAL_STRUCTURE")
record("MUTATION", "baseline_copy(negative control)", "clean copy", "INTERVAL_STRUCTURE passed",
       f"passed={c['passed']}", "MATCH" if c["passed"] else "MISMATCH")

mp_ = MUT / "task-0000" / "episode_00000010.json"
d = json.loads(mp_.read_text())
before = [r["frame_duration"][:] for r in d["memory_annotation"][:2]]
d["memory_annotation"][1]["frame_duration"][0] += 7
mp_.write_text(json.dumps(d, ensure_ascii=False))
print(f"  mutated: {before} -> {[r['frame_duration'] for r in d['memory_annotation'][:2]]}")
rep = run_mut("mutant_dirty")
c = check_of(rep, "INTERVAL_STRUCTURE")
killed = (not c["passed"]) and c["measured"].get("interval_not_contiguous", 0) >= 1
print(f"  measured={json.dumps(c['measured'], ensure_ascii=False)}")
record("MUTATION", "interval_continuity", "rows[1].start += 7", "KILLED",
       "KILLED" if killed else "SURVIVED", "MATCH" if killed else "MISMATCH")

shutil.rmtree(MUT)
md5_after = hashlib.md5(SRC.read_bytes()).hexdigest()
record("MUTATION", "source_untouched", "md5(task-0000/episode_00000010.json)",
       md5_before, md5_after, "MATCH" if md5_before == md5_after else "MISMATCH")

banner("6. 全量基线（生产阈值）")
if SKIP_FULL:
    print("  跳过：MOMA_SKIP_FULL=1（全量扫描由 Master 排队，不在这里自行发起）")
    rc_full = None
else:
    full_out = HERE / "gate_report_full.json"
    argv = ["--data-root", str(DATA), "--frames-meta-root", str(FRAMES),
            "--world-size", "32", "--num-workers", "16",
            "--subtask-max-len", SUBTASK_MAX_LEN, "--prompt-max-len", PROMPT_MAX_LEN,
            "--file-list", str(FILELIST),
            "--jobs", str(JOBS), "--sample-episodes", "100", "--baseline-target-max", "101",
            "--out", str(full_out)]
    buf = io.StringIO()
    t = time.time()
    try:
        with redirect_stdout(buf):
            rc_full = G.main(argv)
    except SystemExit as e:
        rc_full = int(e.code) if e.code is not None else 0
    (HERE / "gate_run_full.log").write_text(buf.getvalue())
    print(f"  full run rc={rc_full} {time.time() - t:.0f}s  [env] {env_reading()}")
    for line in buf.getvalue().splitlines():
        if line.strip().startswith("[") or line.startswith("GATE_VERDICT"):
            print("  " + line.strip())

banner("7. 数据指纹（跑之后）")
FP_AFTER = fingerprint()
(OUT / "fingerprint_after.json").write_text(json.dumps(FP_AFTER, indent=2, sort_keys=True))
same = FP_BEFORE == FP_AFTER
print(f"  tree_digest before={FP_BEFORE['tree_digest_path_size_mtime']}")
print(f"  tree_digest after ={FP_AFTER['tree_digest_path_size_mtime']}")
record("DATA_UNTOUCHED", "fingerprint", "whole tree (10000 files)", "IDENTICAL",
       "IDENTICAL" if same else "CHANGED", "MATCH" if same else "MISMATCH")

banner("汇总")
tsv = OUT / "threeway_results.tsv"
with tsv.open("w") as fh:
    fh.write("check\tdirection\targ\texpected\tobserved\tverdict\n")
    for r in ROWS:
        fh.write("\t".join(str(r[k]) for k in ("check", "direction", "arg", "expected",
                                               "observed", "verdict")) + "\n")
w = [max(len(str(r[k])) for r in ROWS + [{"check": "check", "direction": "direction", "arg": "arg",
                                          "expected": "expected", "observed": "observed",
                                          "verdict": "verdict"}])
     for k in ("check", "direction", "arg", "expected", "observed", "verdict")]
hdr = ("check", "direction", "arg", "expected", "observed", "verdict")
print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
for r in ROWS:
    print("  ".join(str(r[k]).ljust(w[i]) for i, k in enumerate(hdr)))
mis = sum(1 for r in ROWS if r["verdict"] == "MISMATCH")
print()
print(f"THREEWAY_VERDICT: mismatches={mis}  rows={len(ROWS)}  total_seconds={time.time() - t_start:.0f}")
(OUT / "threeway_verdict.txt").write_text(f"mismatches={mis} rows={len(ROWS)}\n")
sys.exit(0 if mis == 0 else 1)
