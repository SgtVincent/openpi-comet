from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import re
from typing import Any


def _subtask_flatten(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(_subtask_flatten(y))
        return out
    return [x]


def _subtask_first_text(x) -> str | None:
    for y in _subtask_flatten(x):
        if y is None:
            continue
        s = str(y).strip()
        if s:
            return s
    return None


def _canonicalize_object_id_fallback(obj_id: str) -> str | None:
    s = str(obj_id).strip()
    if not s:
        return None
    if s.lower() in {"left", "right"}:
        return None
    if s.startswith("[") and s.endswith("]"):
        return None
    s = s.replace("-", "_")
    parts = [p for p in s.split("_") if p]
    if not parts:
        return None
    while parts and re.fullmatch(r"\d+", parts[-1]):
        parts.pop()
    phrase = " ".join(parts).strip().lower()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase if phrase else None


def _subtask_obj_name(raw_id: str | None, object_name_mapping: dict[str, str] | None) -> str | None:
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    if not s:
        return None
    if object_name_mapping is not None and s in object_name_mapping:
        v = object_name_mapping.get(s)
        if v is None:
            return None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _canonicalize_object_id_fallback(s)


def _duration_to_segments(dur):
    if isinstance(dur, list) and len(dur) == 2 and all(isinstance(z, (int,)) for z in dur):
        return [(int(dur[0]), int(dur[1]))]
    if isinstance(dur, list) and dur and all(isinstance(z, list) and len(z) == 2 for z in dur):
        out = []
        for z in dur:
            if all(isinstance(t, (int,)) for t in z):
                out.append((int(z[0]), int(z[1])))
        return out
    ints = [int(z) for z in _subtask_flatten(dur) if isinstance(z, (int,))]
    if len(ints) >= 2:
        return [(min(ints), max(ints))]
    return []


def _extract_main_target(object_id_val):
    if object_id_val is None:
        return None, None
    if isinstance(object_id_val, list):
        args = object_id_val
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        args = [a for a in args if a is not None]
        if len(args) >= 2:
            return _subtask_first_text(args[0]), _subtask_first_text(args[-1])
        if len(args) == 1:
            return _subtask_first_text(args[0]), None
        return None, None
    return _subtask_first_text(object_id_val), None


def _apply_template(template: dict[str, Any], *, obj: str | None, src: str | None, dst: str | None, target: str | None, verb: str) -> str:
    s = template.get("template", "")
    if not s:
        return ""
    for k, v in {"obj": obj, "src": src, "dst": dst, "target": target, "verb": verb}.items():
        s = s.replace("{" + k + "}", v or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


class SubtaskPhraseConverter:
    def __init__(
        self,
        *,
        subtask_source: str,
        subtask_template_path: str | None,
        subtask_object_name_mapping_path: str | None,
        subtask_templates: dict[str, Any] | None = None,
        object_name_mapping: dict[str, str] | None = None,
        subtask_joiner: str = " then ",
    ) -> None:
        self.subtask_source = subtask_source
        self.subtask_joiner = subtask_joiner
        self._subtask_templates: dict[str, Any] | None = None
        self._object_name_mapping: dict[str, str] | None = None

        if self.subtask_source == "orchestrator":
            return

        if subtask_templates is not None:
            self._subtask_templates = subtask_templates
        else:
            if subtask_template_path is None:
                raise ValueError("subtask_template_path is required when subtask_source is not orchestrator")
            with open(subtask_template_path, "r", encoding="utf-8") as f:
                self._subtask_templates = json.load(f)

        if object_name_mapping is not None:
            self._object_name_mapping = object_name_mapping
        else:
            if subtask_object_name_mapping_path is None:
                raise ValueError("subtask_object_name_mapping_path is required when subtask_source is not orchestrator")
            with open(subtask_object_name_mapping_path, "r", encoding="utf-8") as f:
                self._object_name_mapping = json.load(f)

    def phrase_from_skill_ann(self, ann: dict) -> str | None:
        action = _subtask_first_text(ann.get("skill_description"))
        if not action:
            return None
        manip_raw = _subtask_first_text(ann.get("manipulating_object_id"))
        obj_raw, tgt_raw = _extract_main_target(ann.get("object_id"))
        a = action.strip()
        al = a.lower()

        if self._subtask_templates is None:
            tpl = None
        else:
            tpl = self._subtask_templates.get("skill", {}).get(a) or self._subtask_templates.get("primitive", {}).get(a)
        fields = [] if tpl is None else (tpl.get("fields", []) or [])

        if fields == ["obj"] and tgt_raw is not None:
            main_raw = tgt_raw
        else:
            main_raw = obj_raw or manip_raw

        obj = _subtask_obj_name(main_raw, self._object_name_mapping)
        arg_raw = tgt_raw if tgt_raw is not None else obj_raw
        arg = _subtask_obj_name(arg_raw, self._object_name_mapping)
        src = arg if "src" in fields else None
        dst = arg if "dst" in fields else None
        target = arg if "target" in fields else None

        def phrase_for_action(*, obj: str | None, src: str | None, dst: str | None, target: str | None) -> str:
            if tpl is None:
                return ""
            return _apply_template(tpl, obj=obj, src=src, dst=dst, target=target, verb=al)

        if al == "move to":
            tgt = target or dst or src or obj
            phrase = phrase_for_action(obj=obj, src=None, dst=None, target=tgt)
            return phrase or (f"move to the {tgt}" if tgt else a)
        if al in {"pick up from", "take out of"}:
            phrase = phrase_for_action(obj=obj, src=src, dst=None, target=None)
            if phrase:
                return phrase
            if obj and src:
                return f"{al} the {obj} {('from' if al=='pick up from' else 'out of')} the {src}".strip()
            return al
        if al in {"place in", "place on"}:
            phrase = phrase_for_action(obj=obj, src=None, dst=dst, target=None)
            if phrase:
                return phrase
            if obj and dst:
                return f"{al} the {obj} {('in' if al=='place in' else 'on')} the {dst}".strip()
            return al
        if al in {"open door", "close door"}:
            door = target or obj or "door"
            phrase = phrase_for_action(obj=None, src=None, dst=None, target=door)
            return phrase or (f"{al.split()[0]} the {door}".strip())

        phrase = phrase_for_action(obj=obj, src=src, dst=dst, target=target)
        if phrase:
            return phrase
        if obj and arg:
            if obj == arg:
                return f"{al} the {obj}".strip()
            return f"{al} the {obj} to the {arg}".strip()
        if obj:
            return f"{al} the {obj}".strip()
        return al

    def phrase_from_primitive_ann(self, ann: dict) -> str | None:
        atoms = [str(x).strip() for x in _subtask_flatten(ann.get("primitive_description")) if str(x).strip()]
        if not atoms:
            return None

        phrases = []
        obj_id = ann.get("object_id")
        manip = ann.get("manipulating_object_id")
        for i, a in enumerate(atoms):
            args_i = obj_id[i] if isinstance(obj_id, list) and len(obj_id) == len(atoms) else obj_id
            manip_i = manip[i] if isinstance(manip, list) and len(manip) == len(atoms) else manip
            manip_raw = _subtask_first_text(manip_i)
            obj_raw, tgt_raw = _extract_main_target(args_i)

            if self._subtask_templates is None:
                tpl = None
            else:
                tpl = self._subtask_templates.get("skill", {}).get(a) or self._subtask_templates.get("primitive", {}).get(a)

            fields = [] if tpl is None else (tpl.get("fields", []) or [])
            if fields == ["obj"] and tgt_raw is not None:
                main_raw = tgt_raw
            else:
                main_raw = obj_raw or manip_raw

            obj = _subtask_obj_name(main_raw, self._object_name_mapping)
            arg_raw = tgt_raw if tgt_raw is not None else obj_raw
            arg = _subtask_obj_name(arg_raw, self._object_name_mapping)
            src = arg if "src" in fields else None
            dst = arg if "dst" in fields else None
            target = arg if "target" in fields else None

            if tpl is not None:
                tpl_phrase = _apply_template(tpl, obj=obj, src=src, dst=dst, target=target, verb=a.lower())
            else:
                tpl_phrase = ""

            if tpl_phrase:
                phrases.append(tpl_phrase)
            else:
                phrases.append(a.lower())

        phrases = [p for p in phrases if p]
        if not phrases:
            return None
        joined = self.subtask_joiner.join(phrases)
        joined = re.sub(r"\s+", " ", joined).strip()
        m = re.search(r"the ([a-zA-Z0-9 _-]+) (?:onto|to) the \\1\\b", joined)
        if m:
            joined = re.sub(r"(?:onto|to) the \\1\\b", "", joined).strip()
            joined = re.sub(r"\s+", " ", joined).strip()
        return joined

    def build_subtask_segments_for_episode(self, episode_ann: dict) -> tuple[list[tuple[int, int, str]], list[int]]:
        if not isinstance(episode_ann, dict):
            return [], []
        if self.subtask_source == "annotations_skill":
            anns = episode_ann.get("skill_annotation", []) or []
            get_phrase = self.phrase_from_skill_ann
        else:
            anns = episode_ann.get("primitive_annotation", []) or []
            get_phrase = self.phrase_from_primitive_ann

        segs = []
        for a in anns:
            if not isinstance(a, dict):
                continue
            phrase = get_phrase(a)
            if not phrase:
                continue
            for s, e in _duration_to_segments(a.get("frame_duration")):
                segs.append((s, e, phrase))
        segs.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for s, e, t in segs:
            if not merged:
                merged.append([s, e, t])
                continue
            ps, pe, pt = merged[-1]
            if t == pt and s <= pe + 1:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e, t])
        merged = [(int(s), int(e), str(t)) for s, e, t in merged]
        ends = [e for _, e, _ in merged]
        return merged, ends


def _load_episode_annotation_json(*, dataset_root: Path | None, episode_index: int | None, episode_file: Path | None) -> dict[str, Any]:
    if episode_file is not None:
        with open(episode_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if dataset_root is None or episode_index is None:
        raise ValueError("Either episode_file or (dataset_root + episode_index) is required")
    task_id = int(episode_index // 10000)
    p = dataset_root / "annotations" / f"task-{task_id:04d}" / f"episode_{episode_index:08d}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_episode(
    dataset_root: Path | None,
    episode_index: int | None,
    episode_file: Path | None,
    *,
    subtask_template_path: str,
    subtask_object_name_mapping_path: str,
    subtask_joiner: str,
) -> dict[str, Any]:
    ann = _load_episode_annotation_json(dataset_root=dataset_root, episode_index=episode_index, episode_file=episode_file)
    out: dict[str, Any] = {"episode_index": int(episode_index) if episode_index is not None else None}
    if out["episode_index"] is None and episode_file is not None:
        stem = episode_file.stem
        if stem.startswith("episode_") and stem[8:].isdigit():
            out["episode_index"] = int(stem[8:])
    for subtask_source, key in [("annotations_skill", "skill"), ("annotations_primitive", "primitive")]:
        conv = SubtaskPhraseConverter(
            subtask_source=subtask_source,
            subtask_template_path=subtask_template_path,
            subtask_object_name_mapping_path=subtask_object_name_mapping_path,
            subtask_joiner=subtask_joiner,
        )
        items = []
        anns = ann.get("skill_annotation", []) if subtask_source == "annotations_skill" else ann.get("primitive_annotation", [])
        anns = anns or []
        for i, a in enumerate(anns):
            if not isinstance(a, dict):
                continue
            phrase = conv.phrase_from_skill_ann(a) if subtask_source == "annotations_skill" else conv.phrase_from_primitive_ann(a)
            segs = [{"start_frame": int(s), "end_frame": int(e)} for s, e in _duration_to_segments(a.get("frame_duration"))]
            items.append(
                {
                    "index": int(i),
                    "phrase": phrase,
                    "frame_duration": a.get("frame_duration"),
                    "segments": segs,
                    "skill_description": a.get("skill_description"),
                    "primitive_description": a.get("primitive_description"),
                    "object_id": a.get("object_id"),
                    "manipulating_object_id": a.get("manipulating_object_id"),
                }
            )
        merged, _ = conv.build_subtask_segments_for_episode(ann)
        out[key] = {"items": items, "merged_segments": [{"start_frame": s, "end_frame": e, "phrase": t} for s, e, t in merged]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episode-file", type=str, default=None)
    parser.add_argument("--subtask-template-path", type=str, required=True)
    parser.add_argument("--subtask-object-name-mapping-path", type=str, required=True)
    parser.add_argument("--subtask-joiner", type=str, default=" then ")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = _convert_episode(
        Path(args.dataset_root) if args.dataset_root is not None else None,
        int(args.episode_index) if args.episode_index is not None else None,
        Path(args.episode_file) if args.episode_file is not None else None,
        subtask_template_path=args.subtask_template_path,
        subtask_object_name_mapping_path=args.subtask_object_name_mapping_path,
        subtask_joiner=str(args.subtask_joiner),
    )
    s = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output is not None:
        Path(args.output).write_text(s + "\n", encoding="utf-8")
    else:
        print(s)


if __name__ == "__main__":
    main()
