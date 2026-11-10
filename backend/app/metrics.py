"""Metrics tracking and summary calculations for the arena stream.

Conforms to CONTRACT.md metrics & summary specification.
"""

from __future__ import annotations

import time
from typing import Literal

from app.schemas import Metrics, Phase, Summary


class MetricsTracker:
    def __init__(self, start_time: float | None = None) -> None:
        self.start_time: float = start_time if start_time is not None else time.perf_counter()
        self.first_token_time: float | None = None
        self.think_end_time: float | None = None
        self.end_time: float | None = None

        self.think_tokens: int = 0
        self.answer_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.think_tokens + self.answer_tokens

    def record_token(self, phase: Phase, count: int = 1) -> None:
        if count <= 0:
            return
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now

        if phase == "think":
            self.think_tokens += count
        else:
            self.answer_tokens += count

    def record_phase_switch(self) -> None:
        if self.think_end_time is None:
            self.think_end_time = time.perf_counter()

    def record_done(self) -> None:
        if self.end_time is None:
            self.end_time = time.perf_counter()

    def snapshot(self, current_time: float | None = None) -> Metrics:
        now = current_time or (self.end_time or time.perf_counter())
        elapsed_ms = max(0.0, (now - self.start_time) * 1000.0)

        ttft_ms: float | None = None
        if self.first_token_time is not None:
            ttft_ms = round(max(0.0, (self.first_token_time - self.start_time) * 1000.0), 1)

        think_end_ms: float | None = None
        if self.think_end_time is not None:
            think_end_ms = round(max(0.0, (self.think_end_time - self.start_time) * 1000.0), 1)

        # Tokens per second measured from the first token until current time/end time
        tok_per_sec = 0.0
        if self.first_token_time is not None and self.total_tokens > 0:
            active_duration = max(0.001, now - self.first_token_time)
            tok_per_sec = round(self.total_tokens / active_duration, 1)

        return Metrics(
            ttft_ms=ttft_ms,
            elapsed_ms=round(elapsed_ms, 1),
            think_tokens=self.think_tokens,
            answer_tokens=self.answer_tokens,
            total_tokens=self.total_tokens,
            tokens_per_sec=tok_per_sec,
            think_end_ms=think_end_ms,
        )


def compute_summary(
    baseline_metrics: Metrics | None,
    loco_metrics: Metrics | None,
) -> Summary:
    """Compute comparison metrics between baseline and loco runs."""
    think_savings: float | None = None
    total_savings: float | None = None
    time_savings: float | None = None
    winner: Literal["loco", "baseline", "tie"] | None = None

    if baseline_metrics and loco_metrics:
        # Think tokens savings
        if baseline_metrics.think_tokens > 0:
            savings = (1.0 - (loco_metrics.think_tokens / baseline_metrics.think_tokens)) * 100.0
            think_savings = round(savings, 1)

        # Total tokens savings
        if baseline_metrics.total_tokens > 0:
            savings = (1.0 - (loco_metrics.total_tokens / baseline_metrics.total_tokens)) * 100.0
            total_savings = round(savings, 1)

        # Elapsed time savings
        if baseline_metrics.elapsed_ms > 0:
            savings = (1.0 - (loco_metrics.elapsed_ms / baseline_metrics.elapsed_ms)) * 100.0
            time_savings = round(savings, 1)

        # Winner based on elapsed time (who finished first)
        time_diff = baseline_metrics.elapsed_ms - loco_metrics.elapsed_ms
        if abs(time_diff) < 50.0:  # within 50ms is a tie
            winner = "tie"
        elif time_diff > 0:
            winner = "loco"
        else:
            winner = "baseline"
    elif loco_metrics and not baseline_metrics:
        winner = "loco"
    elif baseline_metrics and not loco_metrics:
        winner = "baseline"

    return Summary(
        baseline=baseline_metrics,
        loco=loco_metrics,
        think_token_savings_pct=think_savings,
        total_token_savings_pct=total_savings,
        time_savings_pct=time_savings,
        winner=winner,
    )
