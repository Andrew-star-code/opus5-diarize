"""Простой текст и Markdown."""
from __future__ import annotations

from ..schemas import SessionDetail
from .common import clock, speaker_names


def to_txt(detail: SessionDetail, *, timestamps: bool = True) -> str:
    names = speaker_names(detail)
    out: list[str] = [detail.title, "=" * len(detail.title), ""]
    last_speaker: int | None = None

    for seg in detail.segments:
        if not seg.text.strip():
            continue
        who = names.get(seg.speaker_id)
        # Заголовок реплики печатаем только при смене говорящего —
        # иначе стена из повторяющихся имён.
        if seg.speaker_id != last_speaker:
            head = f"[{clock(seg.start)}] {who}:" if timestamps else f"{who}:"
            out.append("")
            out.append(head)
            last_speaker = seg.speaker_id
        out.append(seg.text.strip())

    return "\n".join(out).strip() + "\n"


def to_markdown(detail: SessionDetail) -> str:
    names = speaker_names(detail)
    out = [f"# {detail.title}", ""]
    last_speaker: int | None = None
    for seg in detail.segments:
        if not seg.text.strip():
            continue
        if seg.speaker_id != last_speaker:
            out.append("")
            out.append(f"**{names.get(seg.speaker_id)}** `{clock(seg.start)}`")
            out.append("")
            last_speaker = seg.speaker_id
        out.append(seg.text.strip())
    return "\n".join(out).strip() + "\n"
