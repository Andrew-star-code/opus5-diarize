"""Публичное API сессий."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session as DBSession
from sqlmodel import delete, select

from .. import schemas
from ..config import settings
from ..models import Job, JobType, Segment, Session, Speaker, Status, utcnow
from ..services import events, jobs, storage, transcripts
from .audio import ranged_response
from .deps import get_db, get_session_or_404

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=list[schemas.SessionBrief])
def list_sessions(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: DBSession = Depends(get_db),
):
    rows = db.exec(
        select(Session).order_by(Session.created_at.desc()).offset(offset).limit(limit)
    ).all()
    return transcripts.briefs_for(db, list(rows))


@router.get("/sessions/{session_id}", response_model=schemas.SessionDetail)
def get_session_detail(
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    return transcripts.to_detail(db, session)


@router.patch("/sessions/{session_id}", response_model=schemas.SessionBrief)
def patch_session(
    patch: schemas.SessionPatch,
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    if patch.title is not None:
        session.title = patch.title.strip()[:200]
    if patch.prompt is not None:
        session.prompt = patch.prompt.strip()[:1000]
    if patch.num_speakers is not None:
        session.num_speakers = (
            patch.num_speakers if 0 <= patch.num_speakers <= 20 else 0
        )
    session.updated_at = utcnow()
    db.add(session)
    db.commit()
    db.refresh(session)
    return transcripts.to_brief(db, session)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    sid = session.id
    db.execute(delete(Segment).where(Segment.session_id == sid))
    db.execute(delete(Speaker).where(Speaker.session_id == sid))
    db.execute(delete(Job).where(Job.session_id == sid))
    db.delete(session)
    db.commit()
    storage.remove_session_files(sid)


@router.get("/sessions/{session_id}/stats", response_model=schemas.SessionStats)
def get_stats(
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    return transcripts.compute_stats(db, session)


@router.get("/sessions/{session_id}/audio")
def get_audio(
    request: Request,
    session: Session = Depends(get_session_or_404),
):
    if not session.audio_path:
        raise HTTPException(404, "У сессии нет аудио")
    return ranged_response(storage.abs_path(session.audio_path), request)


@router.post("/sessions/{session_id}/refine", response_model=schemas.SessionBrief)
def refine_session(
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    """Переобработать запись полным batch-пайплайном.

    Нужно и для live-черновиков, и когда захотелось прогнать уже готовую
    запись более крупной моделью.
    """
    if not session.audio_path:
        raise HTTPException(400, "У сессии нет аудио для переобработки")
    if session.status in (Status.QUEUED, Status.PROCESSING):
        raise HTTPException(409, "Сессия уже в обработке")
    job = jobs.enqueue(db, session, JobType.REFINE)
    events.publish(session.id, "queued", {"job_id": job.id})
    return transcripts.to_brief(db, session)


@router.patch("/segments/{segment_id}", response_model=schemas.SegmentOut)
def patch_segment(
    segment_id: int,
    patch: schemas.SegmentPatch,
    db: DBSession = Depends(get_db),
):
    seg = db.get(Segment, segment_id)
    if seg is None:
        raise HTTPException(404, "Сегмент не найден")

    if patch.text is not None and patch.text != seg.text:
        seg.text = patch.text
        seg.edited = True
        # Пословные тайминги после ручной правки врут. Чинить их без
        # повторного выравнивания нельзя, поэтому честно выбрасываем —
        # экспортёры умеют работать по границам сегмента.
        seg.words = "[]"

    if patch.speaker_id is not None:
        speaker = db.get(Speaker, patch.speaker_id)
        if speaker is None or speaker.session_id != seg.session_id:
            raise HTTPException(400, "Спикер не принадлежит этой сессии")
        seg.speaker_id = patch.speaker_id
        seg.edited = True

    db.add(seg)
    db.commit()
    db.refresh(seg)
    return transcripts._to_segment_out(seg)


@router.patch("/speakers/{speaker_id}", response_model=schemas.SpeakerOut)
def patch_speaker(
    speaker_id: int,
    patch: schemas.SpeakerPatch,
    db: DBSession = Depends(get_db),
):
    speaker = db.get(Speaker, speaker_id)
    if speaker is None:
        raise HTTPException(404, "Спикер не найден")
    if patch.display_name is not None:
        speaker.display_name = patch.display_name.strip()[:80] or speaker.display_name
        # Имя вписано — подсказка модели больше не нужна, даже если это
        # и было её имя.
        speaker.suggested_name = ""
    if patch.color is not None:
        speaker.color = patch.color
    db.add(speaker)
    db.commit()
    db.refresh(speaker)
    return transcripts._to_speaker_out(speaker)


@router.get("/sessions/{session_id}/events")
async def session_events(
    request: Request,
    session: Session = Depends(get_session_or_404),
):
    """SSE-поток прогресса обработки."""
    queue = events.subscribe(session.id)

    async def generator():
        # Первым кадром — текущее состояние, чтобы клиент, подключившийся
        # в середине, не сидел с пустым прогрессом до следующего события.
        yield events.format_sse(
            {"event": "state", "data": {"status": session.status, "quality": session.quality}}
        )
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # держим соединение живым через прокси
                    continue
                yield events.format_sse(payload)
        finally:
            events.unsubscribe(session.id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/config")
def get_config():
    """Что показать в «Настройках». Только чтение: менять модели на лету
    нельзя — они загружены в VRAM воркеров."""
    return {
        "batch_asr_model": settings.batch_asr_model,
        "live_asr_model": settings.live_asr_model,
        "compute_type": settings.compute_type,
        "default_language": settings.default_language,
        "diarization_model": settings.diarization_model,
        "live_diarization": settings.live_diarization,
        "num_speakers": settings.num_speakers,
        "max_speakers": settings.max_speakers,
        "max_live_sessions": settings.max_live_sessions,
        "mic_processing": settings.mic_processing,
        "auto_refine_live": settings.auto_refine_live,
        "formats": _format_list(),
    }


def _format_list() -> list[dict]:
    from ..exporters import FORMATS

    return [{"id": k, "label": f.label, "ext": f.ext} for k, f in FORMATS.items()]


@router.get("/health")
def health(db: DBSession = Depends(get_db)):
    pending = db.exec(select(Job).where(Job.state.in_(["queued", "running"]))).all()
    return {"ok": True, "pending_jobs": len(pending)}
