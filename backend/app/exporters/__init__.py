"""Реестр форматов экспорта."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..schemas import SessionDetail, SessionStats
from .docx_export import to_docx
from .json_export import to_json
from .plain import to_markdown, to_txt
from .subtitles import to_srt, to_vtt

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass(frozen=True)
class Format:
    ext: str
    mime: str
    label: str
    render: Callable[[SessionDetail, SessionStats], bytes]


def _text(fn: Callable[[SessionDetail], str]):
    return lambda detail, stats: fn(detail).encode("utf-8")


FORMATS: dict[str, Format] = {
    "txt": Format("txt", "text/plain; charset=utf-8", "Текст", _text(to_txt)),
    "md": Format("md", "text/markdown; charset=utf-8", "Markdown", _text(to_markdown)),
    "srt": Format("srt", "application/x-subrip; charset=utf-8", "Субтитры SRT", _text(to_srt)),
    "vtt": Format("vtt", "text/vtt; charset=utf-8", "Субтитры WebVTT", _text(to_vtt)),
    "json": Format("json", "application/json; charset=utf-8", "JSON",
                   lambda d, s: to_json(d, s).encode("utf-8")),
    "docx": Format("docx", DOCX_MIME, "Word", lambda d, s: to_docx(d, s)),
}
