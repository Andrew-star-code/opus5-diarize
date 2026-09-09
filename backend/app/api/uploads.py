"""Загрузка файлов на распознавание."""
from __future__ import annotations

import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session as DBSession

from .. import schemas
from ..config import settings
from ..models import JobType, Session, SourceType, Status
from ..services import events, jobs, storage, transcripts
from .deps import get_db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["uploads"])

_CHUNK = 1024 * 1024


@router.post("/uploads", response_model=schemas.SessionBrief, status_code=201)
async def upload_audio(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    num_speakers: int = Form(0),
    db: DBSession = Depends(get_db),
):
    original = Path(file.filename or "recording")
    suffix = original.suffix.lower()
    if suffix not in storage.ALLOWED_SUFFIXES:
        raise HTTPException(
            415,
            f"Формат {suffix or '?'} не поддерживается. "
            f"Допустимо: {', '.join(sorted(storage.ALLOWED_SUFFIXES))}",
        )

    session = Session(
        title=(title or original.stem)[:200],
        source_type=SourceType.UPLOAD,
        status=Status.QUEUED,
        language=language or (settings.default_language if settings.default_language != "auto" else None),
        original_filename=original.name,
        prompt=(prompt or "").strip()[:1000],
        # Отрицательные и абсурдные значения молча приводим к «определять
        # автоматически»: пользователь мог ошибиться в поле формы.
        num_speakers=num_speakers if 0 <= num_speakers <= 20 else 0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    target = storage.session_dir(session.id) / f"source{suffix}"
    try:
        async with aiofiles.open(target, "wb") as out:
            while chunk := await file.read(_CHUNK):
                await out.write(chunk)
    except Exception:
        db.delete(session)
        db.commit()
        storage.remove_session_files(session.id)
        raise

    if target.stat().st_size == 0:
        db.delete(session)
        db.commit()
        storage.remove_session_files(session.id)
        raise HTTPException(400, "Файл пустой")

    session.audio_path = storage.rel_path(target)
    session.duration_sec = storage.probe_duration(target)
    db.add(session)
    db.commit()
    db.refresh(session)

    job = jobs.enqueue(db, session, JobType.BATCH)
    events.publish(session.id, "queued", {"job_id": job.id})
    log.info("Принят файл %s → сессия %s", original.name, session.id)

    return transcripts.to_brief(db, session)
