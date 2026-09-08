"""Memory field text for MoMA-VLA (design P1, item 1).

The design's model-facing text is five labelled fields, generated in a fixed
autoregressive order (design section 2.2)::

    Memory: Completed: pick up the radio from the coffee table (1). Active: press the radio (1).
    Primitive: press the radio (1)
    Skill: press the radio
    Next skill: END_OF_PRIMITIVE
    Next primitive: place the radio on the coffee table (1)

Two label schemes, and why both exist
-------------------------------------
The design specifies **plain-text labels** encoded with the existing
SentencePiece vocabulary: no new special tokens, no embedding resize, no
checkpoint break.  That is :data:`LabelScheme.PLAIN_TEXT`, and it is the
default.

:data:`LabelScheme.RESERVED_SLOT` maps each label onto one of PaliGemma's 99
pre-existing reserved pieces (``<unused0>`` .. ``<unused98>``, ids 7..105).  It
satisfies the same "no resize, no checkpoint break" requirement -- reserved
slots are already in the vocabulary -- and additionally makes each label a
single, atomic id.  That matters because plain-text labels are re-segmented by
context: ``"Memory:"`` at the start of a string and after a newline do not
produce the same leading token.

The choice is still open, so **nothing outside this module may hardcode a label
string or assume a token count**.  Go through :class:`MemoryTextCodec`.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Final

import numpy as np


class LabelScheme(enum.Enum):
    """How the five field labels are represented in the model-facing text."""

    #: Ordinary text, encoded with the existing SentencePiece vocabulary.
    #: This is what the design specifies for P1.
    PLAIN_TEXT = "plain_text"
    #: One pre-existing reserved vocabulary slot per label: atomic and
    #: context-invariant, still no resize and no checkpoint break.
    RESERVED_SLOT = "reserved_slot"


DEFAULT_LABEL_SCHEME: Final[LabelScheme] = LabelScheme.PLAIN_TEXT


@dataclasses.dataclass(frozen=True)
class MemoryField:
    """One field of the memory state."""

    name: str
    #: Plain-text label including the colon, exactly as the design writes it.
    label: str
    #: Reserved SentencePiece piece used when the reserved-slot scheme is active.
    slot: str


# Order is the autoregressive generation order of design section 2.2 and is part
# of the on-disk/on-checkpoint contract. Append only; never reorder or renumber
# the slots, because doing so silently changes what a trained model's tokens mean.
MEMORY_FIELDS: Final[tuple[MemoryField, ...]] = (
    MemoryField("memory", "Memory:", "<unused0>"),
    MemoryField("primitive", "Primitive:", "<unused1>"),
    MemoryField("skill", "Skill:", "<unused2>"),
    MemoryField("next_skill", "Next skill:", "<unused3>"),
    MemoryField("next_primitive", "Next primitive:", "<unused4>"),
)

FIELD_NAMES: Final[tuple[str, ...]] = tuple(f.name for f in MEMORY_FIELDS)

#: Section cue that the design appends after Previous Memory so the model knows
#: to start generating. Kept here so callers never spell it themselves.
# Lower-case, matching the design document verbatim.  This is not cosmetic:
# the upper-case form "Previous Memory:" tokenises to [24226, 22021, 235292],
# whose last two ids are exactly "Memory:" in its after-a-space form -- so the
# prefix cue and the target's own first label would share two tokens.  The
# lower-case form [24226, 6884, 235292] shares nothing but the colon, keeping
# "what to copy" and "what to update" distinct at the token level.
CURRENT_MEMORY_CUE: Final[str] = "Current memory:"
PREVIOUS_MEMORY_CUE: Final[str] = "Previous memory:"

#: Trailing sequence marker of ``model_target_text``. A marker for end-to-end
#: checks, never encoded as part of the CE target (design section 3.4.3).
ACTION_QUERY_MARKER: Final[str] = "Action Query:"

#: Canonical previous-memory text at the start of an episode.  Offline data and
#: online rollout must use the same string; otherwise chunk 0 is trained with a
#: Previous-memory cue and served with the legacy Subtask cue.
INITIAL_PREVIOUS_MEMORY: Final[str] = "No task steps have been completed; prepare to begin the task."

# Longest-first, so "Next skill:" can never be matched as "Skill:" and
# "Next primitive:" never as "Primitive:".
_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(f.label) for f in sorted(MEMORY_FIELDS, key=lambda f: len(f.label), reverse=True))
)
_SLOT_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(f.slot) for f in sorted(MEMORY_FIELDS, key=lambda f: len(f.slot), reverse=True))
)
_LABEL_TO_SLOT: Final[dict[str, str]] = {f.label: f.slot for f in MEMORY_FIELDS}
_SLOT_TO_LABEL: Final[dict[str, str]] = {f.slot: f.label for f in MEMORY_FIELDS}


class MemoryTextError(ValueError):
    """Raised when memory text violates the field contract."""


def build_memory_text(**fields: str) -> str:
    """Build the canonical, model-facing memory text.

    Accepts exactly the names in :data:`FIELD_NAMES`.  Emits them in the design's
    autoregressive order, one ``Label: value`` line each.

    Excluded by design section 3.1: ``annotation_index``, ``segments``, raw object
    handles, task id and frame index.  Those stay in the audit JSONL.  Values are
    whitespace-collapsed so that a newline inside an annotation cannot fake a
    field boundary -- newline is the only field separator.
    """
    missing = [n for n in FIELD_NAMES if n not in fields]
    unexpected = [n for n in fields if n not in FIELD_NAMES]
    if missing or unexpected:
        raise MemoryTextError(
            f"build_memory_text expects exactly {list(FIELD_NAMES)}; missing={missing} unexpected={unexpected}"
        )

    lines: list[str] = []
    for field in MEMORY_FIELDS:
        raw = fields[field.name]
        if raw is None:
            raise MemoryTextError(
                f"memory field {field.name!r} is None; pass an explicit empty string or a sentinel instead"
            )
        body = " ".join(str(raw).split())
        if _LABEL_RE.search(body) or _SLOT_RE.search(body):
            raise MemoryTextError(
                f"memory field {field.name!r} contains a field label in its body: {body!r}. "
                "Label text must never come from annotation content."
            )
        lines.append(f"{field.label} {body}" if body else field.label)
    return "\n".join(lines)


def parse_memory_text(text: str, *, strict: bool = True) -> dict[str, str]:
    """Inverse of :func:`build_memory_text`: split generated text back into fields.

    Design section 6.2's six Planner metrics all require the generated text split
    back into fields before anything can be computed, so this is on the critical
    path for evaluation rather than a convenience.

    **Splits on newline, not on the colon.** ``build_memory_text`` collapses
    whitespace inside every value precisely so that newline is the only field
    separator, and the production tokenizer path (``tokenize_memory``) preserves
    newlines -- unlike ``tokenize_subtask``, which flattens them to spaces. A
    colon-based split would be silently wrong on the 2.61% of real rows whose
    ``Memory:`` value contains its own colon (measured: 274 of 10,489 sampled
    rows, e.g. ``..., next: grab chocolate chip cookie.``): those rows would be
    cut in the middle of a value and every downstream metric would be computed on
    mangled fields with nothing raised.

    Label matching is used only to *verify* each line's prefix, longest-first so
    ``Next skill:`` can never be consumed as ``Skill:``, and case-sensitively --
    bare lower-case ``skill:`` does not occur in the corpus (0 of 10,489), so a
    case-insensitive match would only add ways to be wrong.

    ``strict=True`` raises unless exactly the five fields are recovered in order.
    That is deliberate: a partial result would let evaluation proceed on fields it
    silently failed to find, and "the model is bad at next-skill" is what a
    truncated parse looks like. Set ``strict=False`` only to compute the section
    6.2 format-completion diagnostic, which is meant to observe malformed output
    rather than reject it.
    """
    if text is None:
        raise MemoryTextError("parse_memory_text() requires text, got None")
    body = str(text).strip()
    # Tolerate a trailing Action Query marker: it is a sequence marker for
    # end-to-end checks, never part of the CE target (design section 3.4.3).
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if lines and lines[-1].strip() == ACTION_QUERY_MARKER:
        lines = lines[:-1]

    if len(lines) != len(MEMORY_FIELDS):
        if strict:
            raise MemoryTextError(
                f"expected {len(MEMORY_FIELDS)} newline-separated fields, got {len(lines)}. "
                f"Refusing to return a partial parse. Lines were: {lines!r}"
            )
        lines = _split_by_labels(body)

    out: dict[str, str] = {}
    for field, line in zip(MEMORY_FIELDS, lines):
        prefix = field.label + " "
        if line.startswith(prefix):
            value = line[len(prefix):]
        elif line.startswith(field.label):
            value = line[len(field.label):]
        elif strict:
            raise MemoryTextError(
                f"field {field.name!r} should start with {field.label!r}, got {line!r}. "
                "Field order is part of the supervision contract, not a presentation choice."
            )
        else:
            value = line
        out[field.name] = value.strip()

    if strict:
        empty = [n for n, v in out.items() if not v]
        if empty:
            raise MemoryTextError(f"fields parsed but empty: {empty}; text was {text!r}")
    return out


def _split_by_labels(body: str) -> list[str]:
    """Fallback for text that lost its newlines (e.g. a flattened generation).

    Only reachable from ``strict=False``. Longest-first via ``_LABEL_RE`` so
    ``Next primitive:`` is never consumed as ``Primitive:``.
    """
    hits = list(_LABEL_RE.finditer(body))
    if not hits:
        return [body]
    spans = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        spans.append(body[m.start():end].strip())
    return spans


def format_completion(text: str) -> dict[str, object]:
    """Design section 6.2 "Format completion" -- diagnostic only, never raises.

    Reports what a strict parse would have rejected, so a malformed generation is
    counted rather than thrown away. Explicitly not a parser gate: section 6.2
    says this metric is for diagnosis and must not become a hard parser門.
    """
    body = str(text or "").strip()
    lines = [ln for ln in body.split("\n") if ln.strip()]
    had_marker = bool(lines) and lines[-1].strip() == ACTION_QUERY_MARKER
    if had_marker:
        lines = lines[:-1]
    labels_in_order = [f.label for f in MEMORY_FIELDS]
    present = [lab for lab in labels_in_order if lab in body]
    prefixes_ok = sum(
        1 for f, ln in zip(MEMORY_FIELDS, lines) if ln.startswith(f.label)
    )
    try:
        parse_memory_text(body, strict=True)
        strict_ok = True
    except MemoryTextError:
        strict_ok = False
    return {
        "newline_field_count": len(lines),
        "expected_field_count": len(MEMORY_FIELDS),
        "labels_present": len(present),
        "labels_in_expected_order": present == labels_in_order[: len(present)],
        "line_prefixes_matching": prefixes_ok,
        "had_action_query_marker": had_marker,
        "strict_parse_ok": strict_ok,
    }


def validate_field_structure(text: str) -> None:
    """Fail-closed check that ``text`` carries all five labels in the required order.

    Design section 6.2 treats format completion as a *diagnostic* at rollout
    time, not a hard parser gate.  It is enforced here on the training /
    tokenisation path, where a malformed target is a data bug.
    """
    found = _LABEL_RE.findall(text)
    expected = [f.label for f in MEMORY_FIELDS]
    if found != expected:
        raise MemoryTextError(f"memory label sequence {found} does not match required order {expected}")


@dataclasses.dataclass(frozen=True)
class MemoryTextCodec:
    """Single owner of the label representation.

    ``tokenizer`` is any object exposing SentencePiece's
    ``encode``/``decode``/``piece_to_id``/``unk_id``.  Callers must not encode
    labels themselves: the scheme is still an open decision, and this class is
    what keeps that decision from leaking into rollout logic.
    """

    tokenizer: object
    scheme: LabelScheme = DEFAULT_LABEL_SCHEME

    def __post_init__(self) -> None:
        if self.scheme is LabelScheme.RESERVED_SLOT:
            # Fail-closed: a tokenizer without atomic reserved slots would encode
            # them as several ordinary pieces, silently losing the only property
            # this scheme is chosen for.
            broken = []
            unk_id = self.tokenizer.unk_id() if hasattr(self.tokenizer, "unk_id") else -1
            for field in MEMORY_FIELDS:
                slot_id = self.tokenizer.piece_to_id(field.slot)
                if slot_id == unk_id or slot_id < 0:
                    broken.append(field.slot)
                    continue
                if list(self.tokenizer.encode(field.slot)) != [slot_id]:
                    broken.append(f"{field.slot}(splits)")
            if broken:
                raise MemoryTextError(
                    "tokenizer does not provide atomic reserved slots for memory labels: " + ", ".join(broken)
                )

    @property
    def label_token_ids(self) -> dict[str, list[int]]:
        """Field name -> the token ids its label occupies under the active scheme."""
        return {f.name: list(self.tokenizer.encode(self.to_wire(f.label))) for f in MEMORY_FIELDS}

    def to_wire(self, text: str) -> str:
        """Readable text -> the exact string handed to the tokenizer."""
        if self.scheme is LabelScheme.PLAIN_TEXT:
            return text
        return _LABEL_RE.sub(lambda m: _LABEL_TO_SLOT[m.group(0)], text)

    def from_wire(self, text: str) -> str:
        """Tokenizer-facing string -> readable text."""
        if self.scheme is LabelScheme.PLAIN_TEXT:
            return text
        return _SLOT_RE.sub(lambda m: _SLOT_TO_LABEL[m.group(0)], text)

    def encode(self, text: str, *, validate: bool = True) -> list[int]:
        """Encode readable memory text to token ids (no BOS/EOS)."""
        if validate:
            validate_field_structure(text)
        return list(self.tokenizer.encode(self.to_wire(text)))

    def decode(self, ids) -> str:
        """Decode token ids back to readable memory text."""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.from_wire(self.tokenizer.decode([int(i) for i in ids]))

    def label_token_cost(self) -> int:
        """Total token count of the five labels under the active scheme.

        Used by tests and by the token-budget reporting the design asks for; not
        by rollout logic, which must stay independent of token counts.
        """
        return sum(len(ids) for ids in self.label_token_ids.values())
