"""FastAPI entrypoint and SSE streaming orchestrator for loco-arena.

Routes conform to CONTRACT.md.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.inference import run_demo_stream, run_live_stream
from app.metrics import compute_summary
from app.schemas import (
    ArenaDoneEvent,
    DoneEvent,
    ErrorEvent,
    Health,
    Metrics,
    ModelEvent,
    Preset,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify presets file exists
    if not settings.presets_file.is_file():
        print(f"Warning: Presets file not found at {settings.presets_file}")
    yield


app = FastAPI(
    title="loco-arena Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
async def get_health() -> Health:
    return Health(
        status="ok",
        mode=settings.arena_mode,
        baseline_model=settings.baseline_model,
        loco_model=settings.loco_model,
    )


@app.get("/api/presets", response_model=list[Preset])
async def get_presets() -> list[Preset]:
    if not settings.presets_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Presets file not found at {settings.presets_file}",
        )
    try:
        with open(settings.presets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Preset.model_validate(item) for item in data]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load presets: {exc}",
        )


async def _stream_worker(
    gen: AsyncGenerator[ModelEvent, None],
    queue: asyncio.Queue[ModelEvent | None],
) -> None:
    try:
        async for event in gen:
            await queue.put(event)
    except Exception as exc:
        # Wrap unexpected worker crash
        pass
    finally:
        await queue.put(None)  # Sentinel to mark completion of this model


@app.get("/api/arena/stream")
async def stream_arena(
    request: Request,
    prompt: str = Query(..., min_length=1, max_length=8000, description="User prompt"),
    max_tokens: int | None = Query(None, ge=1, le=32768),
    temperature: float | None = Query(None, ge=0.0, le=2.0),
) -> StreamingResponse:
    async def sse_event_publisher() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[ModelEvent | None] = asyncio.Queue()

        if settings.arena_mode == "demo":
            gen_baseline = run_demo_stream(
                model_role="baseline",
                model_name=settings.baseline_model,
                prompt=prompt,
                delay_scale=settings.demo_delay_scale,
            )
            gen_loco = run_demo_stream(
                model_role="loco",
                model_name=settings.loco_model,
                prompt=prompt,
                delay_scale=settings.demo_delay_scale,
            )
        else:
            gen_baseline = run_live_stream(
                model_role="baseline",
                model_name=settings.baseline_model,
                base_url=settings.baseline_base_url,
                api_key=settings.baseline_api_key,
                prompt=prompt,
                settings=settings,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            gen_loco = run_live_stream(
                model_role="loco",
                model_name=settings.loco_model,
                base_url=settings.loco_base_url,
                api_key=settings.loco_api_key,
                prompt=prompt,
                settings=settings,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        task_baseline = asyncio.create_task(_stream_worker(gen_baseline, queue))
        task_loco = asyncio.create_task(_stream_worker(gen_loco, queue))

        active_producers = 2
        last_baseline_metrics: Metrics | None = None
        last_loco_metrics: Metrics | None = None

        try:
            while active_producers > 0:
                if await request.is_disconnected():
                    break

                try:
                    # 15s keepalive ping if no token emitted
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue

                if item is None:
                    active_producers -= 1
                    continue

                # Track metrics for final summary
                if isinstance(item, (DoneEvent, ErrorEvent)):
                    if item.model == "baseline":
                        if isinstance(item, DoneEvent):
                            last_baseline_metrics = item.metrics
                    elif item.model == "loco":
                        if isinstance(item, DoneEvent):
                            last_loco_metrics = item.metrics

                payload = item.model_dump_json()
                yield f"data: {payload}\n\n"

            # Always emit arena_done as the final event
            summary = compute_summary(last_baseline_metrics, last_loco_metrics)
            arena_done = ArenaDoneEvent(summary=summary)
            yield f"data: {arena_done.model_dump_json()}\n\n"

        finally:
            task_baseline.cancel()
            task_loco.cancel()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse_event_publisher(), headers=headers)
