"""Environment-driven settings (see CONTRACT.md, "Environment variables")."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    arena_mode: Literal["demo", "live"] = "demo"

    baseline_base_url: str = "http://localhost:8001/v1"
    baseline_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    baseline_api_key: str = "EMPTY"

    loco_base_url: str = "http://localhost:8002/v1"
    loco_model: str = "MuyaoYuan/LoCo-7B"
    loco_api_key: str = "EMPTY"

    max_tokens: int = Field(default=8192, ge=1)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    request_timeout: float = Field(default=600.0, gt=0.0)

    cors_origins: str = "*"
    presets_path: str = "presets.json"

    # Optional, demo-only: multiplier applied to simulated delays.
    # 1.0 = realistic pacing, 0 = instant (used by tests).
    demo_delay_scale: float = Field(default=1.0, ge=0.0)

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["*"]

    @property
    def presets_file(self) -> Path:
        path = Path(self.presets_path)
        return path if path.is_absolute() else BACKEND_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
