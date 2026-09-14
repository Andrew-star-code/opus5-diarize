"""Служебное API для воркеров.

Наружу не выставляется: Caddy отдаёт 403 на /api/internal/*.
Воркеры ходят сюда по внутренней сети docker.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session as DBSession
from sqlmodel import select

from .. import schemas
from ..config import settings
from ..models import (Job, JobState, JobType, Session, SourceType, Speaker, Quality,
                      Status, utcnow)
from ..services import events, jobs, transcripts
from .deps import get_db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/internal", tags=["internal"])


def _get_job(job_id: str, db: DBSession) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Задание не найдено")
    return job


# response_model=None обязателен: из аннотации возврата FastAPI иначе
# попытается собрать модель по объединению с Response и откажется стартовать.
@router.get("/jobs/next", response_model=None)
def next_job(
    types: str = Query("batch,refine"),
    db: DBSession = Depends(get_db),
) -> schemas.JobOut | Response:
    """Выдаёт одно задание или 204, если очередь пуста.

    Возвращаем именно голый Response, а не None с response_model: иначе
    FastAPI приложит к 204 тело "null", а ответ без содержимого тела
    иметь не должен.
    """
    wanted = [t.strip() for t in types.split(",") if t.strip()]
    jobs.requeue_stale(db)
    job = jobs.claim_next(db, wanted)
    if job is None:
        return Response(status_code=204)

    session = db.get(Session, job.session_id)
    if session is None or not session.audio_path:
        job.state = JobState.FAILED
        job.error = "У сессии нет аудио"
        db.add(job)
        db.commit()
        return Response(status_code=204)

    events.publish(session.id, "progress", {"stage": "queued", "progress": 0.0})
    return schemas.JobOut(
        id=job.id,
        session_id=session.id,
        type=job.type,
        audio_path=session.audio_path,
        language=session.language,
        # Настройки записи перекрывают глобальные: у планёрки и интервью
        # разное число участников и разные имена в затравке.
        num_speakers=session.num_speakers or settings.num_speakers,
        max_speakers=settings.max_speakers,
        prompt=session.prompt or settings.initial_prompt,
    )


@router.post("/jobs/{job_id}/progress", status_code=204)
def report_progress(
    job_id: str,
    payload: schemas.JobProgress,
    db: DBSession = Depends(get_db),
):
    job = _get_job(job_id, db)
    job.stage = payload.stage
    job.progress = max(0.0, min(1.0, payload.progress))
    job.heartbeat_at = utcnow()
    db.add(job)
    db.commit()
    events.publish(
        job.session_id, "progress", {"stage": job.stage, "progress": job.progress}
    )


# Название, которое сервис придумал сам: у живой записи — дата, у
# загруженной — имя файла. Только такое заменяет название от модели.
_AUTO_TITLE = re.compile(r"^Запись \d{2}\.\d{2}\.\d{4} \d{2}:\d{2}$")


def _title_untouched(session: Session) -> bool:
    title = (session.title or "").strip()
    if not title or _AUTO_TITLE.match(title):
        return True
    return bool(session.original_filename) and title == Path(session.original_filename).stem[:200]


def _apply_notes(db: DBSession, session: Session, result: schemas.JobResult) -> None:
    """Название, краткое содержание и подсказки имён от языковой модели.

    Всё необязательно. Название, которое пользователь дал сам, не
    трогаем; имя подсказываем только тем говорящим, кого он ещё не назвал.
    """
    if result.summary.strip():
        session.summary = result.summary.strip()[:4000]
    if result.title and result.title.strip() and _title_untouched(session):
        session.title = result.title.strip()[:120]
    if result.speaker_names:
        for sp in db.exec(select(Speaker).where(Speaker.session_id == session.id)):
            name = (result.speaker_names.get(sp.label) or "").strip()[:80]
            if name and sp.display_name == transcripts._generated_name(sp):
                sp.suggested_name = name
                db.add(sp)


@router.post("/jobs/{job_id}/result", status_code=204)
def submit_result(
    job_id: str,
    result: schemas.JobResult,
    db: DBSession = Depends(get_db),
):
    job = _get_job(job_id, db)
    session = db.get(Session, job.session_id)
    if session is None:
        raise HTTPException(404, "Сессия не найдена")

    transcripts.replace_transcript(
        db, session, result.segments,
        quality=Quality.FINAL,
        language=result.language,
        duration_sec=result.duration_sec,
        model_info=result.model_info,
    )
    _apply_notes(db, session, result)
    session.status = Status.READY
    session.error = None

    job.state = JobState.DONE
    job.progress = 1.0
    job.stage = "done"
    job.finished_at = utcnow()

    db.add(session)
    db.add(job)
    db.commit()

    log.info("Задание %s (%s) завершено: %d сегментов",
             job.id, job.type, len(result.segments))
    events.publish(session.id, "done", {"quality": Quality.FINAL})


@router.post("/jobs/{job_id}/fail", status_code=204)
def submit_failure(
    job_id: str,
    payload: schemas.JobFailure,
    db: DBSession = Depends(get_db),
):
    job = _get_job(job_id, db)
    job.state = JobState.FAILED
    job.error = payload.error[:2000]
    job.finished_at = utcnow()
    db.add(job)

    session = db.get(Session, job.session_id)
    if session:
        # Черновик live-сессии остаётся доступным: неудачная доводка
        # не повод терять то, что уже распознали в реальном времени.
        has_draft = session.source_type == SourceType.LIVE and session.quality == Quality.DRAFT
        session.status = Status.READY if has_draft else Status.FAILED
        session.error = payload.error[:2000]
        session.updated_at = utcnow()
        db.add(session)

    db.commit()
    log.error("Задание %s провалено: %s", job_id, payload.error[:500])
    events.publish(job.session_id, "failed", {"error": payload.error[:500]})


# ─── live-сессии ────────────────────────────────────────────────

@router.post("/live/sessions", response_model=schemas.SessionBrief, status_code=201)
def create_live_session(
    payload: schemas.LiveSessionCreate,
    db: DBSession = Depends(get_db),
):
    session = Session(
        title=payload.title or f"Запись {utcnow().strftime('%d.%m.%Y %H:%M')}",
        source_type=SourceType.LIVE,
        status=Status.RECORDING,
        quality=Quality.DRAFT,
        language=payload.language or (
            settings.default_language if settings.default_language != "auto" else None
        ),
        prompt=payload.prompt or "",
        num_speakers=payload.num_speakers or 0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    log.info("Начата live-сессия %s", session.id)
    return transcripts.to_brief(db, session)


@router.post("/live/sessions/{session_id}/finalize", response_model=schemas.SessionBrief)
def finalize_live_session(
    session_id: str,
    payload: schemas.LiveFinalize,
    db: DBSession = Depends(get_db),
):
    """Сохранить черновик и, если просили, поставить доводку в очередь."""
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Сессия не найдена")

    if payload.audio_path:
        session.audio_path = payload.audio_path
    if payload.duration_sec:
        session.duration_sec = payload.duration_sec

    transcripts.replace_transcript(
        db, session, payload.segments,
        quality=Quality.DRAFT,
        duration_sec=payload.duration_sec,
        model_info=payload.model_info,
    )
    session.status = Status.READY
    db.add(session)
    db.commit()
    db.refresh(session)

    should_refine = (
        payload.refine
        and settings.auto_refine_live
        and session.audio_path
        and payload.segments
    )
    if should_refine:
        job = jobs.enqueue(db, session, JobType.REFINE)
        events.publish(session.id, "queued", {"job_id": job.id, "type": JobType.REFINE})
        log.info("Live-сессия %s поставлена на доводку", session.id)
    else:
        events.publish(session.id, "done", {"quality": Quality.DRAFT})

    db.refresh(session)
    return transcripts.to_brief(db, session)


@router.get("/live/capacity")
def live_capacity(db: DBSession = Depends(get_db)):
    """Сколько live-сессий разрешено держать одновременно (упирается в VRAM)."""
    return {"max_live_sessions": settings.max_live_sessions}


@router.get("/paths")
def data_paths():
    return {
        "data_dir": str(settings.data_dir),
        "audio_dir": str(settings.audio_dir),
    }
