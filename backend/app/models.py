"""Схема БД. Backend — единственный процесс, который в неё пишет."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class SourceType:
    LIVE = "live"
    UPLOAD = "upload"


class Status:
    RECORDING = "recording"
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Quality:
    DRAFT = "draft"    # черновик из live-стрима
    FINAL = "final"    # результат полного batch-пайплайна


class JobType:
    BATCH = "batch"      # загруженный файл
    REFINE = "refine"    # переобработка live-записи


class JobState:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: str = Field(default_factory=new_id, primary_key=True)
    title: str = ""
    source_type: str = SourceType.UPLOAD
    status: str = Status.QUEUED
    quality: str = Quality.FINAL
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)
    duration_sec: float = 0.0
    language: Optional[str] = None
    # путь относительно DATA_DIR, чтобы БД не зависела от точки монтирования
    audio_path: Optional[str] = None
    original_filename: Optional[str] = None
    model_info: str = "{}"          # JSON: чем считали
    error: Optional[str] = None

    # Затравка для модели: имена участников, термины, названия. Контекст
    # у каждой записи свой, поэтому поле здесь, а не только в настройках.
    prompt: str = ""
    # Сколько человек говорит. 0 — определять автоматически. Точное число
    # заметно улучшает разделение: иначе модель гадает в широком диапазоне.
    num_speakers: int = 0


class Speaker(SQLModel, table=True):
    __tablename__ = "speakers"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessions.id", index=True)
    label: str = ""                 # как назвала модель: SPEAKER_00
    display_name: str = ""          # что видит пользователь
    color: str = "#64748b"
    order: int = 0


class Segment(SQLModel, table=True):
    __tablename__ = "segments"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessions.id", index=True)
    idx: int = 0
    start: float = 0.0
    end: float = 0.0
    speaker_id: Optional[int] = Field(default=None, foreign_key="speakers.id")
    text: str = ""
    # [{"w": "слово", "s": 1.23, "e": 1.44, "p": 0.98}] — JSON-колонкой,
    # потому что по отдельным словам мы никогда не делаем SQL-запросов,
    # а строк было бы в 10-15 раз больше.
    words: str = "[]"
    edited: bool = False


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="sessions.id", index=True)
    type: str = JobType.BATCH
    state: str = Field(default=JobState.QUEUED, index=True)
    stage: str = ""
    progress: float = 0.0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
