from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import shutil


MAX_POINTS = 900


def _parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or list(default)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_sample_readme(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    demo_match = re.search(r"- Demo:\s*`(\d+)`", text)
    if demo_match is None:
        raise ValueError(f"在 {path} 中找不到 Demo")
    episode_index = int(demo_match.group(1))

    frame_range_match = re.search(r"- Frame range:\s*`\[(\d+),\s*(\d+)\)`", text)
    frame_start = int(frame_range_match.group(1)) if frame_range_match else 0
    frame_end = int(frame_range_match.group(2)) if frame_range_match else None

    skill_index_match = re.search(r"- Skill index:\s*`(\d+)`", text)
    skill_index = int(skill_index_match.group(1)) if skill_index_match else None

    skill_type_match = re.search(r"- Skill type:\s*`([^`]+)`", text)
    skill_type = skill_type_match.group(1) if skill_type_match else None

    suspicion_match = re.search(r"- Suspicion score:\s*`([^`]+)`", text)
    suspicion_score = suspicion_match.group(1) if suspicion_match else None

    anomaly_match = re.search(r"- Anomaly step fraction:\s*`([^`]+)`", text)
    anomaly_step_fraction = anomaly_match.group(1) if anomaly_match else None

    prefilter_match = re.search(r"- Prefilter reasons:\s*`(\[[^\n`]+\])`", text)
    prefilter_reasons = []
    if prefilter_match:
        try:
            prefilter_reasons = list(ast.literal_eval(prefilter_match.group(1)))
        except Exception:
            prefilter_reasons = [prefilter_match.group(1)]

    clip_paths = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        clip_match = re.search(rf"- {camera}:\s*`([^`]+)`", text)
        if clip_match:
            clip_paths[camera] = Path(clip_match.group(1))

    metrics = {}
    for label, key in (
        ("Main explainer", "trajectory_metrics_explainer"),
        ("Outlier detail", "outlier_detail"),
        ("Raw arrays", "trajectory_metrics_arrays"),
        ("Top anomaly hits", "top_step_anomaly_hits"),
    ):
        m = re.search(rf"- {re.escape(label)}:\s*`([^`]+)`", text)
        if m:
            metrics[key] = Path(m.group(1))

    title = path.parent.name.replace("_", " ")
    return {
        "title": title,
        "episode_index": episode_index,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "skill_index": skill_index,
        "skill_type": skill_type,
        "suspicion_score": suspicion_score,
        "anomaly_step_fraction": anomaly_step_fraction,
        "prefilter_reasons": prefilter_reasons,
        "clip_paths": clip_paths,
        "metrics": metrics,
        "readme_path": path,
    }


def _task_name_to_id(dataset_root: Path) -> dict[str, int]:
    task_rows = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    return {row["task_name"]: int(row["task_index"]) for row in task_rows}


def _resolve_episode_index(dataset_root: Path, task: str | None, episode_local_index: int | None) -> int:
    if task is None or episode_local_index is None:
        raise ValueError("未提供 sample README 时，需要同时传入 --task 和 --episode-local-index")
    task_id_by_name = _task_name_to_id(dataset_root)
    if task not in task_id_by_name:
        examples = ", ".join(sorted(task_id_by_name)[:10])
        raise ValueError(f"未知 task={task}。可用 task 示例: {examples}")
    task_id = task_id_by_name[task]
    episode_rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    candidates = sorted(int(row["episode_index"]) for row in episode_rows if int(row["episode_index"]) // 10000 == task_id)
    if not candidates:
        raise ValueError(f"task={task} 没有 episode")
    if episode_local_index < 0 or episode_local_index >= len(candidates):
        raise ValueError(f"--episode-local-index 超界，当前 task 共 {len(candidates)} 个 episode")
    return candidates[episode_local_index]


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _video_path(dataset_root: Path, episode_index: int, camera: str) -> Path:
    task_id = episode_index // 10000
    return (
        dataset_root
        / "videos"
        / f"task-{task_id:04d}"
        / f"observation.images.rgb.{camera}"
        / f"episode_{episode_index:08d}.mp4"
    )


def _relative_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _load_episode_frame(dataset_root: Path, episode_index: int) -> dict:
    task_id = episode_index // 10000
    data_path = dataset_root / "data" / f"task-{task_id:04d}" / f"episode_{episode_index:08d}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到 parquet: {data_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("需要 pandas 才能构建时序 viewer") from exc

    df = pd.read_parquet(
        data_path,
        columns=["timestamp", "episode_index", "action", "observation.state", "observation.cam_rel_poses"],
    )
    return {"df": df, "data_path": data_path}


def _to_matrix(series) -> list[list[float]]:
    matrix = []
    for row in series.to_numpy():
        matrix.append([float(x) for x in row])
    return matrix


def _select_dims_by_variance(matrix: list[list[float]], limit: int) -> list[int]:
    if not matrix:
        return []
    dim = len(matrix[0])
    means = [0.0] * dim
    for row in matrix:
        for i, value in enumerate(row):
            means[i] += value
    n = len(matrix)
    means = [value / n for value in means]
    vars_ = [0.0] * dim
    for row in matrix:
        for i, value in enumerate(row):
            diff = value - means[i]
            vars_[i] += diff * diff
    ranked = sorted(range(dim), key=lambda i: vars_[i], reverse=True)
    return ranked[: min(limit, dim)]


def _downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= max_points:
        return list(range(length))
    stride = math.ceil(length / max_points)
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def _project_series(matrix: list[list[float]], dims: list[int], sample_indices: list[int]) -> dict[str, list]:
    projected = {}
    for dim in dims:
        projected[str(dim)] = [matrix[i][dim] for i in sample_indices]
    return projected


def _write_index_html(
    *,
    out_dir: Path,
    title: str,
    episode_index: int,
    cameras: list[str],
    frame_start: int,
    frame_end: int | None,
    fps: int,
    video_rel_paths: dict[str, str],
    readme_rel_path: str | None,
    metrics_rel_paths: dict[str, str],
    meta: dict,
    sequence_payload: dict,
) -> None:
    reason_items = "".join(f"<li><code>{reason}</code></li>" for reason in meta.get("prefilter_reasons", []))
    video_cards = []
    for camera in cameras:
        rel_path = video_rel_paths[camera]
        video_cards.append(
            f"""
            <section class="video-card">
              <div class="video-title">{camera}</div>
              <video id="video-{camera}" controls preload="metadata" src="{rel_path}"></video>
            </section>
            """
        )

    metric_cards = []
    for key, rel_path in metrics_rel_paths.items():
        if rel_path.endswith(".png"):
            metric_cards.append(
                f"""
                <div class="metric-card">
                  <div class="metric-title">{key}</div>
                  <img src="{rel_path}" alt="{key}" />
                </div>
                """
            )
        else:
            metric_cards.append(
                f"""
                <div class="metric-card">
                  <div class="metric-title">{key}</div>
                  <a href="{rel_path}" target="_blank" rel="noreferrer">{rel_path}</a>
                </div>
                """
            )

    readme_block = ""
    if readme_rel_path is not None:
        readme_block = f'<p><a href="{readme_rel_path}" target="_blank" rel="noreferrer">打开原始 README</a></p>'

    payload_json = json.dumps(sequence_payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1115;
      color: #e7eaf0;
    }}
    .page {{
      max-width: 1880px;
      margin: 0 auto;
      padding: 20px 24px 40px;
    }}
    .subtle {{
      color: #9ba6b8;
    }}
    .header {{
      margin-bottom: 18px;
    }}
    .pill-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0 18px;
    }}
    .pill {{
      background: #1b2030;
      border: 1px solid #2c3447;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 13px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
    }}
    .controls input[type="range"] {{
      width: 100%;
    }}
    button {{
      background: #1f6feb;
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 10px 14px;
      cursor: pointer;
    }}
    button.secondary {{
      background: #2a3244;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.7fr 1fr;
      gap: 18px;
    }}
    .left-column {{
      display: grid;
      gap: 16px;
    }}
    .videos-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}
    .panel, .video-card {{
      background: #151922;
      border: 1px solid #293141;
      border-radius: 14px;
      padding: 14px;
    }}
    .video-title, .metric-title {{
      font-weight: 600;
      margin-bottom: 10px;
    }}
    video {{
      width: 100%;
      max-height: 520px;
      background: black;
      border-radius: 10px;
    }}
    .curve-grid {{
      display: grid;
      gap: 14px;
    }}
    .metric-card {{
      margin-bottom: 12px;
    }}
    .metric-card img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #2a3244;
    }}
    .stat-block {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .small {{
      font-size: 13px;
      color: #9ba6b8;
    }}
    code {{
      background: #202635;
      padding: 1px 6px;
      border-radius: 6px;
    }}
    ul {{
      margin-top: 8px;
      padding-left: 20px;
    }}
    canvas {{
      width: 100% !important;
      height: 260px !important;
    }}
    @media (max-width: 1280px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>{title}</h1>
      <div class="subtle">Episode <code>{episode_index:08d}</code> | RGB + action/state curves | fps={fps}</div>
      <div class="pill-row">
        <div class="pill">focus range: <code>{frame_start} - {frame_end if frame_end is not None else "N/A"}</code></div>
        <div class="pill">total frames: <code id="totalFrames"></code></div>
        <div class="pill">current frame: <code id="currentFrameLabel"></code></div>
        <div class="pill">current time: <code id="currentTimeLabel"></code></div>
      </div>
    </div>

    <div class="controls panel">
      <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" />
      <button onclick="jumpToFocus()">跳到可疑区间</button>
      <button class="secondary" onclick="playAll()">播放</button>
      <button class="secondary" onclick="pauseAll()">暂停</button>
    </div>

    <div class="layout">
      <div class="left-column">
        <div class="videos-grid">
          {''.join(video_cards)}
        </div>
        <div class="panel">
          <div class="video-title">Action Curves</div>
          <div class="small">默认展示方差最大的若干 action 维度，拖动 frame 时会联动竖线。</div>
          <canvas id="actionChart"></canvas>
        </div>
        <div class="panel">
          <div class="video-title">State Curves</div>
          <div class="small">默认展示方差最大的若干 state 维度，便于看遥操轨迹变化。</div>
          <canvas id="stateChart"></canvas>
        </div>
      </div>
      <div class="panel">
        <h3>样本信息</h3>
        <div class="stat-block">
          <div>skill index: <code>{meta.get("skill_index", "N/A")}</code></div>
          <div>skill type: <code>{meta.get("skill_type", "N/A")}</code></div>
          <div>suspicion score: <code>{meta.get("suspicion_score", "N/A")}</code></div>
          <div>anomaly fraction: <code>{meta.get("anomaly_step_fraction", "N/A")}</code></div>
        </div>
        {readme_block}
        <h3>Prefilter Reasons</h3>
        <ul>{reason_items}</ul>
        <h3>质量辅助图</h3>
        {''.join(metric_cards)}
      </div>
    </div>
  </div>

  <script>
    const payload = {payload_json};
    const videoIds = payload.cameras.map((camera) => `video-${{camera}}`);
    const focusStart = payload.focus.start_frame;
    const focusEnd = payload.focus.end_frame;
    const fps = payload.fps;
    let currentFrame = focusStart || 0;

    function allVideos() {{
      return videoIds.map((id) => document.getElementById(id)).filter(Boolean);
    }}

    function frameToSeconds(frame) {{
      return frame / fps;
    }}

    function jumpAll(frame) {{
      const seconds = frameToSeconds(frame);
      for (const video of allVideos()) {{
        const handler = () => {{
          video.currentTime = seconds;
          video.removeEventListener("loadedmetadata", handler);
        }};
        if (video.readyState >= 1) {{
          video.currentTime = seconds;
        }} else {{
          video.addEventListener("loadedmetadata", handler);
        }}
      }}
    }}

    function playAll() {{
      for (const video of allVideos()) {{
        video.play();
      }}
    }}

    function pauseAll() {{
      for (const video of allVideos()) {{
        video.pause();
      }}
    }}

    function jumpToFocus() {{
      setFrame(focusStart || 0);
    }}

    function setFrame(frame) {{
      const clamped = Math.max(0, Math.min(payload.total_frames - 1, Math.round(frame)));
      currentFrame = clamped;
      document.getElementById("frameSlider").value = clamped;
      document.getElementById("currentFrameLabel").textContent = String(clamped);
      document.getElementById("currentTimeLabel").textContent = frameToSeconds(clamped).toFixed(3) + "s";
      jumpAll(clamped);
      if (window.actionGuide) {{
        window.actionGuide.setFrame(clamped);
      }}
      if (window.stateGuide) {{
        window.stateGuide.setFrame(clamped);
      }}
    }}

    function buildChart(canvasId, title, labels, seriesMap, highlightRange) {{
      const ctx = document.getElementById(canvasId);
      const colors = ["#58a6ff", "#f2cc60", "#7ee787", "#ff7b72", "#d2a8ff", "#ffa657", "#79c0ff", "#3fb950"];
      const datasets = Object.entries(seriesMap).map(([dim, values], idx) => {{
        return {{
          label: `${{title}}[${{dim}}]`,
          data: values,
          borderColor: colors[idx % colors.length],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        }};
      }});

      const guideDataset = {{
        label: "current_frame",
        data: [null, null],
        borderColor: "#ffffff",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
      }};
      datasets.push(guideDataset);

      const chart = new Chart(ctx, {{
        type: "line",
        data: {{
          labels,
          datasets,
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: {{
            mode: "nearest",
            intersect: false,
          }},
          plugins: {{
            legend: {{
              labels: {{
                color: "#dce4f2",
              }},
            }},
            tooltip: {{
              callbacks: {{
                title(items) {{
                  const idx = items[0].dataIndex;
                  return `frame=${{labels[idx]}}`;
                }},
              }},
            }},
          }},
          scales: {{
            x: {{
              ticks: {{
                color: "#95a0b2",
              }},
              grid: {{
                color: "rgba(255,255,255,0.08)",
              }},
            }},
            y: {{
              ticks: {{
                color: "#95a0b2",
              }},
              grid: {{
                color: "rgba(255,255,255,0.08)",
              }},
            }},
          }},
          onClick: (_, elements) => {{
            if (!elements.length) return;
            const idx = elements[0].index;
            setFrame(labels[idx]);
          }},
        }},
        plugins: [{{
          id: "focusBand",
          beforeDatasetsDraw(chart) {{
            if (highlightRange.end_frame === null || highlightRange.start_frame === null) {{
              return;
            }}
            const xScale = chart.scales.x;
            const area = chart.chartArea;
            const xStart = xScale.getPixelForValue(highlightRange.start_frame);
            const xEnd = xScale.getPixelForValue(highlightRange.end_frame);
            const ctx = chart.ctx;
            ctx.save();
            ctx.fillStyle = "rgba(255, 120, 120, 0.10)";
            ctx.fillRect(xStart, area.top, Math.max(2, xEnd - xStart), area.bottom - area.top);
            ctx.restore();
          }},
        }}],
      }});

      return {{
        setFrame(frame) {{
          const values = Object.values(seriesMap).flat();
          const minY = Math.min(...values);
          const maxY = Math.max(...values);
          guideDataset.data = [
            {{ x: frame, y: minY }},
            {{ x: frame, y: maxY }},
          ];
          chart.update("none");
        }},
      }};
    }}

    window.addEventListener("load", () => {{
      document.getElementById("totalFrames").textContent = String(payload.total_frames);
      const slider = document.getElementById("frameSlider");
      slider.max = String(payload.total_frames - 1);
      slider.value = String(focusStart || 0);
      slider.addEventListener("input", (event) => {{
        setFrame(Number(event.target.value));
      }});

      window.actionGuide = buildChart(
        "actionChart",
        "action",
        payload.downsampled.frames,
        payload.downsampled.action_series,
        payload.focus,
      );
      window.stateGuide = buildChart(
        "stateChart",
        "state",
        payload.downsampled.frames,
        payload.downsampled.state_series,
        payload.focus,
      );
      setFrame(focusStart || 0);
    }});
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def build_view(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    sample_meta = {}
    episode_index = args.episode_index
    frame_start = args.start_frame
    frame_end = args.end_frame

    if args.sample_readme is not None:
        sample_meta = _parse_sample_readme(Path(args.sample_readme).expanduser().resolve())
        episode_index = sample_meta["episode_index"]
        if args.start_frame is None:
            frame_start = sample_meta["frame_start"]
        if args.end_frame is None:
            frame_end = sample_meta["frame_end"]
    elif episode_index is None:
        episode_index = _resolve_episode_index(dataset_root, args.task, args.episode_local_index)

    if episode_index is None:
        raise ValueError("无法确定 episode_index")

    frame_start = 0 if frame_start is None else int(frame_start)
    frame_end = None if frame_end is None else int(frame_end)

    if out_dir.exists() and args.force_override:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_payload = _load_episode_frame(dataset_root, episode_index)
    df = episode_payload["df"]
    action_matrix = _to_matrix(df["action"])
    state_matrix = _to_matrix(df["observation.state"])
    timestamps = [float(x) for x in df["timestamp"].tolist()]
    total_frames = len(df)

    action_dims = _select_dims_by_variance(action_matrix, args.action_dims)
    state_dims = _select_dims_by_variance(state_matrix, args.state_dims)
    sample_indices = _downsample_indices(total_frames, args.max_points)

    cameras = _parse_csv_list(args.cameras, ["head", "left_wrist", "right_wrist"])
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_rel_paths = {}
    for camera in cameras:
        src = _video_path(dataset_root, episode_index, camera)
        if not src.exists():
            raise FileNotFoundError(f"找不到视频文件: {src}")
        dst = videos_dir / f"{camera}.mp4"
        _symlink(src, dst)
        video_rel_paths[camera] = _relative_path(dst, out_dir)

    readme_rel_path = None
    if sample_meta.get("readme_path") is not None:
        dst = out_dir / "artifacts" / "README.md"
        _symlink(sample_meta["readme_path"], dst)
        readme_rel_path = _relative_path(dst, out_dir)

    metrics_rel_paths = {}
    for key, src in sample_meta.get("metrics", {}).items():
        if src.exists():
            dst = out_dir / "artifacts" / src.name
            _symlink(src, dst)
            metrics_rel_paths[key] = _relative_path(dst, out_dir)

    title = args.title
    if title is None:
        if sample_meta.get("title"):
            title = sample_meta["title"]
        else:
            title = f"Behavior-1K Temporal Viewer - episode_{episode_index:08d}"

    sequence_payload = {
        "episode_index": episode_index,
        "fps": args.fps,
        "cameras": cameras,
        "total_frames": total_frames,
        "timestamps": {
            "start": timestamps[0] if timestamps else 0.0,
            "end": timestamps[-1] if timestamps else 0.0,
        },
        "focus": {
            "start_frame": frame_start,
            "end_frame": frame_end,
        },
        "downsampled": {
            "frames": sample_indices,
            "action_dims": action_dims,
            "state_dims": state_dims,
            "action_series": _project_series(action_matrix, action_dims, sample_indices),
            "state_series": _project_series(state_matrix, state_dims, sample_indices),
        },
    }

    _write_index_html(
        out_dir=out_dir,
        title=title,
        episode_index=episode_index,
        cameras=cameras,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=args.fps,
        video_rel_paths=video_rel_paths,
        readme_rel_path=readme_rel_path,
        metrics_rel_paths=metrics_rel_paths,
        meta=sample_meta,
        sequence_payload=sequence_payload,
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为 Behavior-1K episode 生成一个 headless 时序 web viewer，支持 RGB + action/state 曲线联动。"
    )
    parser.add_argument("--dataset-root", required=True, help="2025-challenge-demos 根目录")
    parser.add_argument("--sample-readme", default=None, help="可选：dirty sample 的 README.md，自动解析 demo/frame range/metrics")
    parser.add_argument("--episode-index", type=int, default=None, help="全局 episode id，例如 380010")
    parser.add_argument("--task", default=None, help="task 名称，例如 spraying_for_bugs")
    parser.add_argument("--episode-local-index", type=int, default=None, help="task 内局部序号，从 0 开始")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cameras", default="head,left_wrist,right_wrist")
    parser.add_argument("--action-dims", type=int, default=8, help="按方差排序后展示的 action 维度数")
    parser.add_argument("--state-dims", type=int, default=8, help="按方差排序后展示的 state 维度数")
    parser.add_argument("--max-points", type=int, default=900, help="曲线最大采样点数")
    parser.add_argument("--output-dir", default="outputs/dataset_viewer_web")
    parser.add_argument("--force-override", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9094)
    parser.add_argument("--no-serve", action="store_true", help="只生成页面，不启动 http.server")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = build_view(args)
    url = f"http://{args.host}:{args.port}/"
    print(f"viewer_root={out_dir}")
    print(f"viewer_url={url}")
    if args.no_serve:
        return

    os.chdir(out_dir)
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)
    print("Serving temporal web viewer ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
