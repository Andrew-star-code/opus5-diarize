"""Асинхронный клиент к backend."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger(__name__)


class BackendClient:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.backend_url.rstrip("/"),
            timeout=httpx.Timeout(15.0, read=60.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def create_session(
        self, title: str | None, language: str | None
    ) -> dict[str, Any]:
        r = await self._http.post(
            "/api/internal/live/sessions",
            json={"title": title, "language": language},
        )
        r.raise_for_status()
        return r.json()

    async def finalize(
        self,
        session_id: str,
        *,
        audio_path: str | None,
        duration_sec: float,
        segments: list[dict[str, Any]],
        model_info: dict[str, Any],
        refine: bool,
    ) -> dict[str, Any]:
        r = await self._http.post(
            f"/api/internal/live/sessions/{session_id}/finalize",
            json={
                "audio_path": audio_path,
                "duration_sec": round(duration_sec, 3),
                "segments": segments,
                "model_info": model_info,
                "refine": refine,
            },
        )
        r.raise_for_status()
        return r.json()
