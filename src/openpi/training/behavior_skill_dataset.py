from __future__ import annotations

import bisect
import json

from behavior.learning.datas.dataset import BehaviorLeRobotDataset


class BehaviorLeRobotSkillDataset(BehaviorLeRobotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skill_phrase_segments: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
        self._skill_phrase_segment_ends: dict[tuple[int, int], list[int]] = {}
        if (
            getattr(self, "_chunk_streaming_using_keyframe", False)
            and getattr(self, "current_streaming_chunk_idx", None) is not None
            and not hasattr(self, "_active_chunks")
        ):
            self._active_chunks = self.chunks

    def _load_skill_phrase_segments(self, episode_index: int, task_index: int) -> None:
        p = self.root / "subtasks" / f"task-{task_index:04d}" / f"episode_{episode_index:08d}.json"
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merged = (payload.get("skill") or {}).get("merged_segments") or []
        segs: list[tuple[int, int, str]] = []
        ends: list[int] = []
        for seg in merged:
            if not isinstance(seg, dict):
                continue
            s = seg.get("start_frame")
            e = seg.get("end_frame")
            t = seg.get("phrase")
            if s is None or e is None or t is None:
                continue
            t = str(t).strip()
            if not t:
                continue
            segs.append((int(s), int(e), t))
            ends.append(int(e))
        key = (task_index, episode_index)
        self._skill_phrase_segments[key] = segs
        self._skill_phrase_segment_ends[key] = ends

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        try:
            ep_idx = int(item["episode_index"].item())
            task_idx = int(item["task_index"].item())
            frame_index = round(item["timestamp"].item() * self.fps)
        except Exception:
            return item

        key = (task_idx, ep_idx)
        if key not in self._skill_phrase_segments:
            try:
                self._load_skill_phrase_segments(ep_idx, task_idx)
            except Exception:
                return item

        segs = self._skill_phrase_segments.get(key) or []
        ends = self._skill_phrase_segment_ends.get(key) or []
        if not segs:
            return item
        i = bisect.bisect_left(ends, frame_index)
        if 0 <= i < len(segs):
            s, e, phrase = segs[i]
            if s <= frame_index <= e and phrase:
                item["task"] = phrase
        return item
