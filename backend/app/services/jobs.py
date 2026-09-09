"""Очередь заданий поверх SQLite.

Redis/Celery тут были бы лишней движущейся деталью: один GPU,
последовательная обработка, единственный писатель в БД.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlmodel import Session as DBSession
from sqlalchemy import update
from sqlmodel import select

from ..config import settings
from ..models import Job, JobState, JobType, Session, Status, utcnow

log = logging.getLogger(__name__)


def enqueue(db: DBSession, session: Session, job_type: str = JobType.BATCH) -> Job:
    job = Job(session_id=session.id, type=job_type)
    session.status = Status.QUEUED
    session.error = None
    session.updated_at = utcnow()
    db.add(job)
    db.add(session)
    db.commit()
    db.refresh(job)
    return job


def claim_next(db: DBSession, types: list[str]) -> Job | None:
    """Забирает одно задание из очереди.

    Гонки нет: воркер один, а даже если запустить второй — переход
    queued → running делается условным UPDATE и проигравший получит 0 строк.
    """
    candidate = db.exec(
        select(Job)
        .where(Job.state == JobState.QUEUED, Job.type.in_(types))
        .order_by(Job.created_at)
        .limit(1)
    ).first()
    if candidate is None:
        return None

    now = utcnow()
    updated = db.execute(
        update(Job)
        .where(Job.id == candidate.id, Job.state == JobState.QUEUED)
        .values(state=JobState.RUNNING, started_at=now, heartbeat_at=now, progress=0.0)
    )
    if updated.rowcount != 1:
        db.rollback()
        return None
    db.commit()
    db.refresh(candidate)

    session = db.get(Session, candidate.session_id)
    if session:
        session.status = Status.PROCESSING
        session.updated_at = now
        db.add(session)
        db.commit()
    return candidate


def requeue_stale(db: DBSession) -> int:
    """Возвращает в очередь задания, чей воркер умер посреди работы."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.job_stale_after)
    stale = db.exec(
        select(Job).where(Job.state == JobState.RUNNING)
    ).all()
    count = 0
    for job in stale:
        beat = job.heartbeat_at or job.started_at or job.created_at
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=timezone.utc)
        if beat < cutoff:
            job.state = JobState.QUEUED
            job.progress = 0.0
            job.stage = ""
            db.add(job)
            count += 1
    if count:
        db.commit()
        log.warning("Возвращено в очередь зависших заданий: %d", count)
    return count
