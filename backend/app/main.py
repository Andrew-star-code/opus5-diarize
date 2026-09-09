"""Точка входа backend'а."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session as DBSession
from sqlmodel import select

from .api import exports, internal, sessions, uploads
from .config import settings
from .db import engine, init_db
from .models import Session
from .services import jobs, storage

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("scribe")

_HOUSEKEEPING_INTERVAL = 300  # сек


async def _housekeeping() -> None:
    """Возвращает в очередь зависшие задания и чистит старое аудио."""
    while True:
        try:
            with DBSession(engine) as db:
                jobs.requeue_stale(db)
                if settings.retention_days > 0:
                    _apply_retention(db)
        except Exception:
            log.exception("Ошибка в фоновом обслуживании")
        await asyncio.sleep(_HOUSEKEEPING_INTERVAL)


def _apply_retention(db: DBSession) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.retention_days)
    stale = db.exec(select(Session).where(Session.created_at < cutoff)).all()
    for session in stale:
        if not session.audio_path:
            continue
        # Транскрипт оставляем, удаляем только тяжёлое аудио.
        storage.remove_session_files(session.id)
        session.audio_path = None
        db.add(session)
        log.info("Аудио сессии %s удалено по сроку хранения", session.id)
    if stale:
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    log.info("БД: %s, данные: %s", settings.db_path, settings.data_dir)

    task = asyncio.create_task(_housekeeping())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Scribe",
    description="Локальная транскрипция и диаризация аудио",
    version="1.0.0",
    lifespan=lifespan,
)

# В проде фронт и API за одним Caddy, кросс-доменных запросов нет.
# CORS нужен только для `npm run dev` на :5173.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(uploads.router)
app.include_router(exports.router)
app.include_router(internal.router)
