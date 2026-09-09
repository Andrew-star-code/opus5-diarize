"""Запись результата распознавания в БД и сборка ответов API."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlmodel import Session as DBSession
from sqlmodel import delete, select

from .. import schemas
from ..models import Segment, Session, Speaker, utcnow
from . import palette


def replace_transcript(
    db: DBSession,
    session: Session,
    segments: list[schemas.ResultSegment],
    *,
    quality: str,
    language: str | None = None,
    duration_sec: float | None = None,
    model_info: dict | None = None,
) -> None:
    """Полностью заменяет транскрипт сессии.

    Вызывается и для загруженного файла, и для «доводки» live-записи —
    во втором случае черновые сегменты просто затираются чистовыми.
    Имена спикеров, если пользователь их уже задал, переносятся по метке
    модели: SPEAKER_00 в черновике и в чистовике — это, как правило,
    один и тот же человек.
    """
    previous_names = {
        sp.label: sp.display_name
        for sp in db.exec(select(Speaker).where(Speaker.session_id == session.id))
        if sp.display_name and sp.display_name != _generated_name(sp)
    }

    db.execute(delete(Segment).where(Segment.session_id == session.id))
    db.execute(delete(Speaker).where(Speaker.session_id == session.id))
    db.flush()

    # Спикеры в порядке первого появления — так «Спикер 1» это тот,
    # кто заговорил первым, а не тот, кому модель дала меньший индекс.
    labels: list[str] = []
    for seg in segments:
        label = seg.speaker or "SPEAKER_00"
        if label not in labels:
            labels.append(label)

    speakers: dict[str, Speaker] = {}
    for i, label in enumerate(labels):
        sp = Speaker(
            session_id=session.id,
            label=label,
            display_name=previous_names.get(label, palette.default_name(i)),
            color=palette.color_for(i),
            order=i,
        )
        db.add(sp)
        speakers[label] = sp
    db.flush()

    for idx, seg in enumerate(segments):
        label = seg.speaker or "SPEAKER_00"
        db.add(
            Segment(
                session_id=session.id,
                idx=idx,
                start=seg.start,
                end=seg.end,
                speaker_id=speakers[label].id,
                text=seg.text.strip(),
                words=json.dumps([w.model_dump() for w in seg.words], ensure_ascii=False),
            )
        )

    session.quality = quality
    if language:
        session.language = language
    if duration_sec:
        session.duration_sec = duration_sec
    elif segments:
        session.duration_sec = max(session.duration_sec, segments[-1].end)
    if model_info:
        session.model_info = json.dumps(model_info, ensure_ascii=False)
    session.updated_at = utcnow()
    db.add(session)


def _generated_name(sp: Speaker) -> str:
    return palette.default_name(sp.order)


def _to_speaker_out(sp: Speaker) -> schemas.SpeakerOut:
    return schemas.SpeakerOut(
        id=sp.id, label=sp.label, display_name=sp.display_name,
        color=sp.color, order=sp.order,
    )


def _to_segment_out(seg: Segment) -> schemas.SegmentOut:
    return schemas.SegmentOut(
        id=seg.id, idx=seg.idx, start=seg.start, end=seg.end,
        speaker_id=seg.speaker_id, text=seg.text,
        words=[schemas.WordOut(**w) for w in json.loads(seg.words or "[]")],
        edited=seg.edited,
    )


# Разрешение ленты разговора в карточке списка. 240 долей хватает,
# чтобы разглядеть перебивания, и это всего ~1 КБ на запись.
RIBBON_SLOTS = 240


def build_ribbon(
    segments: list[Segment], speakers: list[Speaker], duration: float
) -> list[int]:
    """Нарезает запись на равные доли и в каждую пишет номер говорящего."""
    if duration <= 0 or not segments:
        return []

    order_of = {sp.id: sp.order for sp in speakers}
    slots = [-1] * RIBBON_SLOTS

    for seg in segments:
        order = order_of.get(seg.speaker_id)
        if order is None:
            continue
        first = int(seg.start / duration * RIBBON_SLOTS)
        last = int(seg.end / duration * RIBBON_SLOTS)
        first = max(0, min(RIBBON_SLOTS - 1, first))
        # Реплика в одно слово короче доли — но именно такие вставки и
        # делают ленту читаемой, поэтому даём каждой хотя бы одну долю.
        last = max(first + 1, min(RIBBON_SLOTS, last))
        for i in range(first, last):
            slots[i] = order

    return slots


def _brief(
    s: Session, speakers: list[Speaker], segments: list[Segment]
) -> schemas.SessionBrief:
    ordered = sorted(speakers, key=lambda sp: sp.order)
    return schemas.SessionBrief(
        id=s.id, title=s.title or "Без названия", source_type=s.source_type,
        status=s.status, quality=s.quality, created_at=_aware(s.created_at),
        duration_sec=s.duration_sec, language=s.language,
        speaker_count=len(speakers), error=s.error,
        ribbon=build_ribbon(segments, ordered, s.duration_sec),
        speaker_colors=[sp.color for sp in ordered],
    )


def to_brief(db: DBSession, s: Session) -> schemas.SessionBrief:
    speakers = db.exec(select(Speaker).where(Speaker.session_id == s.id)).all()
    segments = db.exec(
        select(Segment).where(Segment.session_id == s.id).order_by(Segment.idx)
    ).all()
    return _brief(s, list(speakers), list(segments))


def briefs_for(db: DBSession, sessions: list[Session]) -> list[schemas.SessionBrief]:
    """Карточки для списка — двумя запросами вместо двух на каждую запись."""
    if not sessions:
        return []

    ids = [s.id for s in sessions]
    speakers_by_session: dict[str, list[Speaker]] = {sid: [] for sid in ids}
    segments_by_session: dict[str, list[Segment]] = {sid: [] for sid in ids}

    for sp in db.exec(select(Speaker).where(Speaker.session_id.in_(ids))):
        speakers_by_session[sp.session_id].append(sp)
    for seg in db.exec(
        select(Segment).where(Segment.session_id.in_(ids)).order_by(Segment.idx)
    ):
        segments_by_session[seg.session_id].append(seg)

    return [
        _brief(s, speakers_by_session[s.id], segments_by_session[s.id])
        for s in sessions
    ]


def to_detail(db: DBSession, s: Session) -> schemas.SessionDetail:
    speakers = db.exec(
        select(Speaker).where(Speaker.session_id == s.id).order_by(Speaker.order)
    ).all()
    segments = db.exec(
        select(Segment).where(Segment.session_id == s.id).order_by(Segment.idx)
    ).all()
    try:
        model_info = json.loads(s.model_info or "{}")
    except json.JSONDecodeError:
        model_info = {}
    return schemas.SessionDetail(
        # Карточку собираем из уже загруженных списков, чтобы не ходить
        # в БД второй раз за тем же самым.
        **_brief(s, list(speakers), list(segments)).model_dump(),
        audio_url=f"/api/sessions/{s.id}/audio" if s.audio_path else None,
        original_filename=s.original_filename,
        prompt=s.prompt,
        num_speakers=s.num_speakers,
        model_info=model_info,
        speakers=[_to_speaker_out(sp) for sp in speakers],
        segments=[_to_segment_out(seg) for seg in segments],
    )


def compute_stats(db: DBSession, s: Session) -> schemas.SessionStats:
    speakers = {
        sp.id: sp
        for sp in db.exec(select(Speaker).where(Speaker.session_id == s.id))
    }
    segments = db.exec(
        select(Segment).where(Segment.session_id == s.id).order_by(Segment.idx)
    ).all()

    talk: dict[int | None, float] = {}
    words: dict[int | None, int] = {}
    turns: dict[int | None, int] = {}
    _NOBODY = -1  # чтобы первый сегмент всегда засчитался как новая реплика
    prev_speaker: int | None = _NOBODY

    for seg in segments:
        key = seg.speaker_id
        talk[key] = talk.get(key, 0.0) + max(0.0, seg.end - seg.start)
        words[key] = words.get(key, 0) + len(seg.text.split())
        if key != prev_speaker:
            turns[key] = turns.get(key, 0) + 1
            prev_speaker = key

    speech_time = sum(talk.values())
    rows: list[schemas.SpeakerStats] = []
    for key, seconds in sorted(talk.items(), key=lambda kv: -kv[1]):
        sp = speakers.get(key)
        wc = words.get(key, 0)
        rows.append(
            schemas.SpeakerStats(
                speaker_id=key,
                display_name=sp.display_name if sp else "Неизвестный",
                color=sp.color if sp else "#94a3b8",
                talk_time_sec=round(seconds, 2),
                share=round(seconds / speech_time, 4) if speech_time else 0.0,
                word_count=wc,
                words_per_min=round(wc / (seconds / 60), 1) if seconds > 1 else 0.0,
                turns=turns.get(key, 0),
            )
        )

    return schemas.SessionStats(
        duration_sec=s.duration_sec,
        speech_time_sec=round(speech_time, 2),
        word_count=sum(words.values()),
        speakers=rows,
    )


def _aware(dt: datetime) -> datetime:
    """SQLite отдаёт naive datetime — возвращаем UTC-осознанный."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
