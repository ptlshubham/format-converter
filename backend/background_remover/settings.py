"""
Runtime settings for the image AI service, read once from the environment.

Every value has a default that keeps local development working as before, so
nothing needs to be set on a laptop. Production sets IMAGE_AI_ENV=production
(see deploy/ecosystem.config.js).

- IMAGE_AI_ENV                    development (default) | production
- IMAGE_AI_ALLOWED_ORIGINS        comma list of browser origins allowed by CORS
- IMAGE_AI_RATE_LIMIT_PER_MINUTE  POST /remove + /export-hd per client IP per minute (0 = off)
- IMAGE_AI_MAX_UPLOAD_MB          largest request body for multipart uploads (/remove)
- IMAGE_AI_MAX_JSON_MB            largest request body for JSON (/export-hd)
- IMAGE_AI_PROFILER_LOG           on | off: per-request profiler JSON files (default on in development, off in production)
- IMAGE_AI_TRUSTED_PROXIES        comma list of proxy IPs whose X-Forwarded-For is believed
"""

import os
from dataclasses import dataclass
from typing import Tuple

DEFAULT_ALLOWED_ORIGINS = "http://localhost:4200,http://127.0.0.1:4200"
DEFAULT_TRUSTED_PROXIES = "127.0.0.1,::1"


def _csv(name: str, default: str) -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _number(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw!r}")
    return value


def _switch(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("on", "true", "1", "yes"):
        return True
    if raw in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"{name} must be on or off, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    env: str
    allowed_origins: Tuple[str, ...]
    rate_limit_per_minute: int
    max_upload_bytes: int
    max_json_bytes: int
    profiler_log: bool
    trusted_proxies: Tuple[str, ...]

    @property
    def is_production(self) -> bool:
        return self.env == "production"


def load_settings() -> Settings:
    env = os.getenv("IMAGE_AI_ENV", "development").strip().lower() or "development"
    if env not in ("development", "production"):
        raise ValueError(f"IMAGE_AI_ENV must be development or production, got {env!r}")
    is_production = env == "production"

    origins = _csv("IMAGE_AI_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    if "*" in origins:
        raise ValueError("IMAGE_AI_ALLOWED_ORIGINS must list origins, not '*'")

    return Settings(
        env=env,
        allowed_origins=origins,
        rate_limit_per_minute=int(_number("IMAGE_AI_RATE_LIMIT_PER_MINUTE", 6)),
        max_upload_bytes=int(_number("IMAGE_AI_MAX_UPLOAD_MB", 26) * 1024 * 1024),
        max_json_bytes=int(_number("IMAGE_AI_MAX_JSON_MB", 20) * 1024 * 1024),
        profiler_log=_switch("IMAGE_AI_PROFILER_LOG", not is_production),
        trusted_proxies=_csv("IMAGE_AI_TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES),
    )


settings = load_settings()
