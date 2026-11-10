"""Tests for incremental, chunk-boundary-safe ThinkParser."""

import pytest
from app.think_parser import ThinkParser, parse_all


def test_standard_think_and_answer():
    chunks = ["<think>\n", "Step 1: Compute.\n", "Step 2: Verify.\n", "</think>\n", "The answer is 42."]
    segments = parse_all(chunks)

    # Filter out empty text transition markers
    think_text = "".join(s.text for s in segments if s.phase == "think")
    answer_text = "".join(s.text for s in segments if s.phase == "answer")

    assert "<think>" not in think_text
    assert "</think>" not in think_text
    assert "Step 1: Compute." in think_text
    assert "The answer is 42." in answer_text

    transitions = [s for s in segments if s.transition]
    assert len(transitions) == 1
    assert transitions[0].phase == "answer"


def test_split_tag_across_chunks():
    # Tags split in arbitrary byte/character slices
    chunks = [
        "<th", "ink>\n",
        "Reasoning here.",
        "</th", "ink>\n",
        "Direct answer."
    ]
    segments = parse_all(chunks)

    think_text = "".join(s.text for s in segments if s.phase == "think")
    answer_text = "".join(s.text for s in segments if s.phase == "answer")

    assert "<think>" not in think_text
    assert "</think>" not in think_text
    assert "Reasoning here." in think_text
    assert "Direct answer." in answer_text


def test_implicit_prefill_think():
    # R1-distill prompt templates often append <think>\n, so completion starts directly with reasoning
    chunks = ["Reasoning without opening tag.\n", "</think>\n", "Final answer \\boxed{7}."]
    segments = parse_all(chunks)

    think_text = "".join(s.text for s in segments if s.phase == "think")
    answer_text = "".join(s.text for s in segments if s.phase == "answer")

    assert "Reasoning without opening tag." in think_text
    assert "Final answer \\boxed{7}." in answer_text


def test_reasoning_content_field():
    # Servers with reasoning parser deliver thinking in separate reasoning_content
    parser = ThinkParser()
    s1 = parser.feed_reasoning("Internal thought 1. ")
    s2 = parser.feed_reasoning("Internal thought 2.")
    s3 = parser.feed("Final answer text.")
    s4 = parser.finish()

    all_segs = s1 + s2 + s3 + s4
    think_text = "".join(s.text for s in all_segs if s.phase == "think")
    answer_text = "".join(s.text for s in all_segs if s.phase == "answer")

    assert "Internal thought 1." in think_text
    assert "Final answer text." in answer_text
    assert any(s.transition for s in all_segs)


def test_no_tags_at_all():
    # Model emits plain text without think tags
    chunks = ["Just plain answer without any tags."]
    segments = parse_all(chunks)

    text = "".join(s.text for s in segments)
    assert text == "Just plain answer without any tags."
