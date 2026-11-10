"""Integration tests for FastAPI endpoints and demo SSE streaming."""

import json
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.main import app

settings = get_settings()


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] in ("demo", "live")
        assert "baseline_model" in data
        assert "loco_model" in data


@pytest.mark.asyncio
async def test_presets_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/presets")
        assert resp.status_code == 200
        presets = resp.json()
        assert isinstance(presets, list)
        assert len(presets) == 5
        for p in presets:
            assert "id" in p
            assert "title" in p
            assert "category" in p
            assert "prompt" in p
            assert "expected_answer" in p
            assert "why_overthinks" in p


@pytest.mark.asyncio
async def test_arena_stream_demo():
    # Force instant demo delay for testing
    settings.demo_delay_scale = 0.0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/arena/stream", params={"prompt": "Which is bigger: 9.11 or 9.9?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.split("\n")
        events = []
        for line in lines:
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[len("data: ") :]
                events.append(json.loads(data_str))

        assert len(events) > 5

        # Check event types present
        event_types = [e["type"] for e in events]
        assert "start" in event_types
        assert "token" in event_types
        assert "phase" in event_types
        assert "done" in event_types
        assert event_types[-1] == "arena_done", "arena_done MUST be the final event"

        summary = events[-1]["summary"]
        assert summary["winner"] in ("loco", "baseline", "tie")
        assert summary["think_token_savings_pct"] is not None
        assert summary["think_token_savings_pct"] > 30.0  # LoCo saves >30% think tokens in demo


@pytest.mark.asyncio
async def test_arena_stream_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Empty prompt
        resp = await client.get("/api/arena/stream", params={"prompt": ""})
        assert resp.status_code == 422

        # Missing prompt
        resp = await client.get("/api/arena/stream")
        assert resp.status_code == 422
