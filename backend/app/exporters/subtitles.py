"""SRT и WebVTT."""
from __future__ import annotations

from ..schemas import SessionDetail
from .common import all_cues, hhmmss


def to_srt(detail: SessionDetail, *, with_speakers: bool = True) -> str:
    lines: list[str] = []
    for i, cue in enumerate(all_cues(detail), start=1):
        text = f"{cue.speaker}: {cue.text}" if with_speakers and cue.speaker else cue.text
        lines.append(str(i))
        lines.append(f"{hhmmss(cue.start)} --> {hhmmss(cue.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(detail: SessionDetail, *, with_speakers: bool = True) -> str:
    lines = ["WEBVTT", ""]
    for cue in all_cues(detail):
        # <v Имя> — родная разметка говорящего в WebVTT, плееры её понимают
        text = f"<v {cue.speaker}>{cue.text}" if with_speakers and cue.speaker else cue.text
        lines.append(f"{hhmmss(cue.start, sep='.')} --> {hhmmss(cue.end, sep='.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
