"""Incremental, chunk-boundary-safe parser for ``<think>`` / ``</think>``.

The parser starts in the ``think`` phase (R1-distill chat templates usually
pre-insert ``<think>\\n`` into the prompt, so the completion may or may not
open with the tag). It switches to ``answer`` exactly once, when ``</think>``
closes. Tags may be split arbitrarily across deltas (``"</th" + "ink>"``) and
are never included in the emitted text.

Servers running a reasoning parser (vLLM ``--reasoning-parser``, some Ollama
builds) deliver thinking in a separate ``reasoning_content`` field and the
answer in ``content`` without any tags. :meth:`ThinkParser.feed_reasoning`
handles that path: reasoning text is always ``think``, and the first
``content`` delta after reasoning forces the transition to ``answer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phase = Literal["think", "answer"]

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
_TAGS = (OPEN_TAG, CLOSE_TAG)


@dataclass(frozen=True, slots=True)
class Segment:
    """A parser output.

    ``transition=True`` marks the think -> answer switch (``text`` is empty,
    ``phase`` is ``"answer"``). Otherwise ``text`` is non-empty content that
    belongs to ``phase``.
    """

    phase: Phase
    text: str = ""
    transition: bool = False


def _partial_tag_suffix_len(buf: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of a tag."""
    longest = 0
    max_check = max(len(t) for t in _TAGS) - 1
    for n in range(1, min(len(buf), max_check) + 1):
        suffix = buf[-n:]
        if any(tag.startswith(suffix) for tag in _TAGS):
            longest = n
    return longest


class ThinkParser:
    def __init__(self) -> None:
        self.phase: Phase = "think"
        self._buf = ""
        self._saw_reasoning = False

    @property
    def in_think(self) -> bool:
        return self.phase == "think"

    def _transition(self) -> Segment | None:
        if self.phase == "think":
            self.phase = "answer"
            return Segment(phase="answer", transition=True)
        return None

    def force_answer(self) -> list[Segment]:
        """Switch to ``answer`` if still thinking (flushes held-back text first)."""
        out = self._flush_buffer()
        seg = self._transition()
        if seg is not None:
            out.append(seg)
        return out

    def feed_reasoning(self, text: str) -> list[Segment]:
        """Feed a ``reasoning_content`` delta: always ``think`` phase, tags stripped."""
        if not text:
            return []
        self._saw_reasoning = True
        cleaned = text.replace(OPEN_TAG, "").replace(CLOSE_TAG, "")
        if not cleaned:
            return []
        if self.phase == "answer":
            # Reasoning after the answer started is unusual; keep it labelled think.
            return [Segment(phase="think", text=cleaned)]
        return [Segment(phase="think", text=cleaned)]

    def feed(self, text: str) -> list[Segment]:
        """Feed a ``content`` delta and return the resulting segments."""
        out: list[Segment] = []
        if not text:
            return out
        if self._saw_reasoning and self.phase == "think" and CLOSE_TAG not in self._buf + text:
            # Server split reasoning out; plain content is the answer.
            out.extend(self.force_answer())

        self._buf += text
        while True:
            open_idx = self._buf.find(OPEN_TAG)
            close_idx = self._buf.find(CLOSE_TAG)
            candidates = [(i, t) for i, t in ((open_idx, OPEN_TAG), (close_idx, CLOSE_TAG)) if i >= 0]
            if not candidates:
                break
            idx, tag = min(candidates)
            before = self._buf[:idx]
            if before:
                out.append(Segment(phase=self.phase, text=before))
            self._buf = self._buf[idx + len(tag) :]
            if tag == CLOSE_TAG:
                seg = self._transition()
                if seg is not None:
                    out.append(seg)

        hold = _partial_tag_suffix_len(self._buf)
        emit = self._buf[: len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold :]
        if emit:
            out.append(Segment(phase=self.phase, text=emit))
        return out

    def _flush_buffer(self) -> list[Segment]:
        if not self._buf:
            return []
        seg = Segment(phase=self.phase, text=self._buf)
        self._buf = ""
        return [seg]

    def finish(self) -> list[Segment]:
        """End of stream: emit any held-back partial-tag text verbatim."""
        return self._flush_buffer()


def parse_all(chunks: list[str]) -> list[Segment]:
    """Convenience helper: run a full list of content deltas through a parser."""
    parser = ThinkParser()
    out: list[Segment] = []
    for chunk in chunks:
        out.extend(parser.feed(chunk))
    out.extend(parser.finish())
    return out
