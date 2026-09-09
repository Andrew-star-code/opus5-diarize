"""Общее для всех экспортёров: таймкоды, имена спикеров, нарезка на реплики."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import SegmentOut, SessionDetail

# Пределы читаемых субтитров: две строки по ~42 символа и не длиннее 7 секунд.
MAX_CUE_CHARS = 84
MAX_CUE_SECONDS = 7.0


def hhmmss(seconds: float, *, sep: str = ",", hours: bool = True) -> str:
    seconds = max(0.0, seconds)
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    if ms == 1000:  # округление вверх не должно давать «.1000»
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if hours:
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
    return f"{m:02d}:{s:02d}{sep}{ms:03d}"


def clock(seconds: float) -> str:
    """Компактный таймкод для текстовых форматов: 1:02:03 или 02:03."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def speaker_names(detail: SessionDetail) -> dict[int, str]:
    return {sp.id: sp.display_name for sp in detail.speakers}


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def split_into_cues(segment: SegmentOut, speaker: str | None) -> list[Cue]:
    """Режет длинный сегмент на субтитровые реплики по словам.

    Один сегмент диалога легко тянется на полминуты — целиком в субтитр
    он не влезает. Если пословных таймингов нет (например, текст правили
    вручную), отдаём сегмент как есть: лучше длинный субтитр, чем неверные
    тайминги.
    """
    if not segment.words:
        return [Cue(segment.start, segment.end, speaker, segment.text.strip())]

    cues: list[Cue] = []
    buf: list[str] = []
    start = segment.words[0].s
    end = start

    for word in segment.words:
        candidate_len = len(" ".join(buf + [word.w]))
        too_long = candidate_len > MAX_CUE_CHARS
        too_slow = (word.e - start) > MAX_CUE_SECONDS
        if buf and (too_long or too_slow):
            cues.append(Cue(start, end, speaker, " ".join(buf)))
            buf, start = [], word.s
        buf.append(word.w)
        end = word.e

    if buf:
        cues.append(Cue(start, end, speaker, " ".join(buf)))
    return cues


def all_cues(detail: SessionDetail) -> list[Cue]:
    names = speaker_names(detail)
    cues: list[Cue] = []
    for seg in detail.segments:
        if not seg.text.strip():
            continue
        cues.extend(split_into_cues(seg, names.get(seg.speaker_id)))
    return cues
