"""HTTP-клиент к backend.

Воркер намеренно не ходит в БД напрямую: единственный писатель в SQLite —
backend, поэтому блокировок и гонок между контейнерами просто не возникает.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger(__name__)


class BackendClient:
    def __init__(self, base_url: str | None = None) -> None:
        self._http = httpx.Client(
            base_url=(base_url or settings.backend_url).rstrip("/"),
            timeout=httpx.Timeout(30.0, read=120.0),
        )

    def close(self) -> None:
        self._http.close()

    def next_job(self, types: str = "batch,refine") -> dict[str, Any] | None:
        r = self._http.get("/api/internal/jobs/next", params={"types": types})
        r.raise_for_status()
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def progress(self, job_id: str, stage: str, value: float) -> None:
        try:
            self._http.post(
                f"/api/internal/jobs/{job_id}/progress",
                json={"stage": stage, "progress": round(value, 4)},
            )
        except httpx.HTTPError as exc:
            # Прогресс — вещь необязательная: потеря кадра не должна
            # ронять обработку записи, которая считалась полчаса.
            log.warning("Не удалось отправить прогресс: %s", exc)

    def submit_result(self, job_id: str, payload: dict[str, Any]) -> None:
        r = self._http.post(f"/api/internal/jobs/{job_id}/result", json=payload)
        r.raise_for_status()

    def submit_failure(self, job_id: str, error: str) -> None:
        try:
            self._http.post(
                f"/api/internal/jobs/{job_id}/fail", json={"error": error[:2000]}
            )
        except httpx.HTTPError as exc:
            log.error("Не удалось сообщить об ошибке задания: %s", exc)
