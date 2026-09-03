"""Hierarchy special tokens for Hierarchical-MoMA-VLA (design P1, item 1).

The design asks for four paired tags around the four hierarchy fields::

    <MEM>finished: ...; active: ...</MEM>
    <PRIM>press the radio</PRIM>
    <SKILL>press the radio</SKILL>
    <NEXT>place the radio on the coffee table</NEXT>

and says the tags should be "registered as tokenizer special tokens".

Why we repurpose reserved slots instead of growing the vocabulary
----------------------------------------------------------------
The PaliGemma SentencePiece model already ships 99 reserved, user-defined
pieces ``<unused0>`` .. ``<unused98>`` at ids 7..105 (measured, not assumed:
``vocab_size() == 257152``).  Reusing eight of them buys the *only* property
that matters here -- one atomic id per tag, with a boundary the tokenizer can
never re-segment -- at zero cost to checkpoint compatibility.

Growing the vocabulary instead would resize ``embed_tokens.weight`` (and its
tied ``lm_head.weight``), which breaks **every** existing checkpoint on
**every** load path.  Note that ``strict=False`` does not help: safetensors
always calls ``load_state_dict(..., strict=False)`` and PyTorch raises the
size-mismatch ``RuntimeError`` outside the ``if strict:`` block.  There is a
second, quieter hazard: ``gemma_pytorch.py:150`` sets
``image_token_index = 257152``, a sentinel exactly one past the last valid id.
Growing the table turns that sentinel into a real, addressable row.

Readable-vs-wire form
---------------------
Data on disk and in logs keeps the readable ``<MEM>`` spelling so that audit
JSONL stays greppable.  Only the string handed to the tokenizer is rewritten
into the reserved-slot spelling.  :class:`HierarchyTagCodec` owns both
directions; nothing else should hardcode the mapping.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Final

import numpy as np

# Readable tag -> reserved SentencePiece piece.
#
# The slot assignment is an on-disk/on-wire contract: once a checkpoint has been
# trained with it, changing it silently changes what every tag means.  Append
# new pairs at the end (slots 8..98 are still free); never renumber.
HIERARCHY_TAG_TO_SLOT: Final[dict[str, str]] = {
    "<MEM>": "<unused0>",
    "</MEM>": "<unused1>",
    "<PRIM>": "<unused2>",
    "</PRIM>": "<unused3>",
    "<SKILL>": "<unused4>",
    "</SKILL>": "<unused5>",
    "<NEXT>": "<unused6>",
    "</NEXT>": "<unused7>",
}

HIERARCHY_SLOT_TO_TAG: Final[dict[str, str]] = {v: k for k, v in HIERARCHY_TAG_TO_SLOT.items()}

# (open, close) pairs in the autoregressive order the design specifies:
# updated_memory -> primitive -> current_skill -> next_skill.
HIERARCHY_FIELD_ORDER: Final[tuple[tuple[str, str, str], ...]] = (
    ("memory", "<MEM>", "</MEM>"),
    ("primitive", "<PRIM>", "</PRIM>"),
    ("skill", "<SKILL>", "</SKILL>"),
    ("next_skill", "<NEXT>", "</NEXT>"),
)

# Longest-first so that "</MEM>" is never matched as "<" + "/MEM>" and, more
# importantly, so an opening tag can never shadow a closing one.
_TAG_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(t) for t in sorted(HIERARCHY_TAG_TO_SLOT, key=len, reverse=True))
)
_SLOT_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(s) for s in sorted(HIERARCHY_SLOT_TO_TAG, key=len, reverse=True))
)


class HierarchyTagError(ValueError):
    """Raised when hierarchy text violates the tag contract."""


def build_hierarchy_text(
    *,
    memory: str,
    primitive: str,
    skill: str,
    next_skill: str,
) -> str:
    """Build the canonical, model-facing hierarchy string.

    This is the "compressed canonical text" of design item 1: no
    ``annotation_index``, no ``segments``, no raw object handle, no task id and
    no frame index -- those stay in the audit JSONL.  Fields are emitted in the
    autoregressive order the factorization requires.

    Newlines are used only *between* fields.  Inner whitespace is collapsed so
    that a stray newline inside an annotation cannot fake a field boundary.
    """
    values = {"memory": memory, "primitive": primitive, "skill": skill, "next_skill": next_skill}
    parts: list[str] = []
    for name, open_tag, close_tag in HIERARCHY_FIELD_ORDER:
        raw = values[name]
        if raw is None:
            raise HierarchyTagError(f"hierarchy field {name!r} is None; pass an explicit empty string instead")
        body = " ".join(str(raw).split())
        if _TAG_RE.search(body) or _SLOT_RE.search(body):
            raise HierarchyTagError(
                f"hierarchy field {name!r} contains a hierarchy tag in its body: {body!r}. "
                "Tag characters must never come from annotation content."
            )
        parts.append(f"{open_tag}{body}{close_tag}")
    return "\n".join(parts)


def validate_tag_structure(text: str) -> None:
    """Fail-closed check that ``text`` carries all four tag pairs, correctly nested.

    Per design section 6.2 this is a *diagnostic*: format completion is reported,
    not used as a hard parser gate at rollout time.  It is used here to guard the
    training/tokenisation path, where a malformed target is a data bug.
    """
    found = _TAG_RE.findall(text)
    expected: list[str] = []
    for _, open_tag, close_tag in HIERARCHY_FIELD_ORDER:
        expected.extend((open_tag, close_tag))
    if found != expected:
        raise HierarchyTagError(f"hierarchy tag sequence {found} does not match required order {expected}")


@dataclasses.dataclass(frozen=True)
class HierarchyTagCodec:
    """Translates between the readable tag form and the reserved-slot wire form.

    ``tokenizer`` is any object exposing SentencePiece's
    ``encode``/``decode``/``piece_to_id``/``id_to_piece``.  It is kept as a plain
    attribute rather than a subclass so that this codec can be unit-tested
    against the real tokenizer without constructing a model.
    """

    tokenizer: object

    def __post_init__(self) -> None:
        # Fail-closed at construction: a tokenizer that does not actually carry
        # the reserved slots would otherwise silently encode them as several
        # ordinary text pieces, which is the exact drift we are avoiding.
        missing = []
        unk_id = self.tokenizer.unk_id() if hasattr(self.tokenizer, "unk_id") else -1
        for slot in HIERARCHY_TAG_TO_SLOT.values():
            slot_id = self.tokenizer.piece_to_id(slot)
            if slot_id == unk_id or slot_id < 0:
                missing.append(slot)
                continue
            ids = self.tokenizer.encode(slot)
            if list(ids) != [slot_id]:
                missing.append(f"{slot}(splits into {len(ids)} pieces)")
        if missing:
            raise HierarchyTagError(
                "tokenizer does not provide atomic reserved slots for hierarchy tags: " + ", ".join(missing)
            )

    @property
    def slot_ids(self) -> dict[str, int]:
        """Readable tag -> token id."""
        return {tag: self.tokenizer.piece_to_id(slot) for tag, slot in HIERARCHY_TAG_TO_SLOT.items()}

    def to_wire(self, text: str) -> str:
        """Readable ``<MEM>`` form -> reserved-slot ``<unused0>`` form."""
        return _TAG_RE.sub(lambda m: HIERARCHY_TAG_TO_SLOT[m.group(0)], text)

    def from_wire(self, text: str) -> str:
        """Reserved-slot form -> readable form."""
        return _SLOT_RE.sub(lambda m: HIERARCHY_SLOT_TO_TAG[m.group(0)], text)

    def encode(self, text: str, *, validate: bool = True) -> list[int]:
        """Encode readable hierarchy text to token ids (no BOS/EOS)."""
        if validate:
            validate_tag_structure(text)
        return list(self.tokenizer.encode(self.to_wire(text)))

    def decode(self, ids) -> str:
        """Decode token ids back to *readable* hierarchy text."""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return self.from_wire(self.tokenizer.decode([int(i) for i in ids]))
