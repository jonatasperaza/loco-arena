"""Pydantic models mirroring the wire format defined in CONTRACT.md."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel

ModelRole = Literal["baseline", "loco"]
Phase = Literal["think", "answer"]


class Metrics(BaseModel):
    ttft_ms: float | None = None
    elapsed_ms: float = 0.0
    think_tokens: int = 0
    answer_tokens: int = 0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    think_end_ms: float | None = None


class Summary(BaseModel):
    baseline: Metrics | None = None
    loco: Metrics | None = None
    think_token_savings_pct: float | None = None
    total_token_savings_pct: float | None = None
    time_savings_pct: float | None = None
    winner: Literal["loco", "baseline", "tie"] | None = None


class Preset(BaseModel):
    id: str
    title: str
    category: Literal["math", "logic"]
    prompt: str
    expected_answer: str
    why_overthinks: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    mode: Literal["demo", "live"]
    baseline_model: str
    loco_model: str


class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    model: ModelRole
    model_name: str
    t_ms: float


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    model: ModelRole
    phase: Phase
    text: str
    metrics: Metrics


class PhaseEvent(BaseModel):
    type: Literal["phase"] = "phase"
    model: ModelRole
    phase: Phase
    metrics: Metrics


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    model: ModelRole
    metrics: Metrics


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    model: ModelRole
    message: str


class ArenaDoneEvent(BaseModel):
    type: Literal["arena_done"] = "arena_done"
    summary: Summary


ModelEvent = Union[StartEvent, TokenEvent, PhaseEvent, DoneEvent, ErrorEvent]
ArenaEvent = Union[ModelEvent, ArenaDoneEvent]

__all__ = [
    "ArenaDoneEvent",
    "ArenaEvent",
    "DoneEvent",
    "ErrorEvent",
    "Health",
    "Metrics",
    "ModelEvent",
    "ModelRole",
    "Phase",
    "PhaseEvent",
    "Preset",
    "StartEvent",
    "Summary",
    "TokenEvent",
]
