"""DTO для REST API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class WordOut(BaseModel):
    w: str
    s: float
    e: float
    p: float = 1.0


class SpeakerOut(BaseModel):
    id: int
    label: str
    display_name: str
    color: str
    order: int
    # Имя, которое языковая модель нашла в разговоре («Андрей, скажи…»).
    # Только подсказка: пусто, если имя уже вписано пользователем.
    suggested_name: str = ""


class SegmentOut(BaseModel):
    id: int
    idx: int
    start: float
    end: float
    speaker_id: Optional[int]
    text: str
    words: list[WordOut]
    edited: bool


class SessionBrief(BaseModel):
    id: str
    title: str
    source_type: str
    status: str
    quality: str
    created_at: datetime
    duration_sec: float
    language: Optional[str]
    speaker_count: int = 0
    error: Optional[str] = None
    # Лента разговора в сжатом виде: запись нарезана на равные доли,
    # в каждой — порядковый номер доминирующего говорящего (-1 = тишина).
    # Так карточка списка показывает форму разговора, не таща за собой
    # весь транскрипт.
    ribbon: list[int] = Field(default_factory=list)
    speaker_colors: list[str] = Field(default_factory=list)


class SessionDetail(SessionBrief):
    audio_url: Optional[str] = None
    prompt: str = ""
    num_speakers: int = 0
    original_filename: Optional[str] = None
    model_info: dict[str, Any] = Field(default_factory=dict)
    # Краткое содержание от языковой модели: пункты через перевод строки.
    summary: str = ""
    speakers: list[SpeakerOut] = Field(default_factory=list)
    segments: list[SegmentOut] = Field(default_factory=list)


class SessionPatch(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None
    num_speakers: Optional[int] = None


class SegmentPatch(BaseModel):
    text: Optional[str] = None
    speaker_id: Optional[int] = None


class SpeakerPatch(BaseModel):
    display_name: Optional[str] = None
    color: Optional[str] = None


class SpeakerStats(BaseModel):
    speaker_id: Optional[int]
    display_name: str
    color: str
    talk_time_sec: float
    share: float
    word_count: int
    words_per_min: float
    turns: int


class SessionStats(BaseModel):
    duration_sec: float
    speech_time_sec: float
    word_count: int
    speakers: list[SpeakerStats]


# ─── обмен с воркерами ──────────────────────────────────────────

class JobOut(BaseModel):
    id: str
    session_id: str
    type: str
    audio_path: str
    language: Optional[str]
    num_speakers: int
    max_speakers: int
    prompt: str = ""


class JobProgress(BaseModel):
    stage: str = ""
    progress: float = 0.0


class ResultSegment(BaseModel):
    start: float
    end: float
    speaker: Optional[str] = None       # метка модели, напр. SPEAKER_00
    text: str = ""
    words: list[WordOut] = Field(default_factory=list)


class JobResult(BaseModel):
    language: Optional[str] = None
    duration_sec: float = 0.0
    model_info: dict[str, Any] = Field(default_factory=dict)
    segments: list[ResultSegment] = Field(default_factory=list)
    # Полировка языковой моделью — всё необязательно: модель могла быть
    # выключена или недоступна.
    title: Optional[str] = None
    summary: str = ""
    speaker_names: dict[str, str] = Field(default_factory=dict)  # метка → имя


class JobFailure(BaseModel):
    error: str


class LiveSessionCreate(BaseModel):
    title: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    num_speakers: Optional[int] = None


class LiveFinalize(BaseModel):
    audio_path: Optional[str] = None
    duration_sec: float = 0.0
    model_info: dict[str, Any] = Field(default_factory=dict)
    segments: list[ResultSegment] = Field(default_factory=list)
    refine: bool = True
