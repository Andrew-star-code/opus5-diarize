"""Отдача аудиофайла с поддержкой Range.

Без Range браузер не умеет перематывать длинную запись: <audio> тянет
файл целиком, а wavesurfer не может позиционироваться.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".oga": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
    ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
}
_CHUNK = 1024 * 512
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def media_type(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "application/octet-stream")


def ranged_response(path: Path, request: Request) -> FileResponse | StreamingResponse:
    if not path.is_file():
        raise HTTPException(404, "Аудиофайл не найден")

    size = path.stat().st_size
    mime = media_type(path)
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path, media_type=mime, headers={"Accept-Ranges": "bytes"}
        )

    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match:
        raise HTTPException(416, "Некорректный Range")

    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    else:  # суффиксный запрос вида "bytes=-500"
        length = int(raw_end or 0)
        start = max(0, size - length)
        end = size - 1

    if start >= size or start > end:
        raise HTTPException(416, "Range за пределами файла",
                            headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)

    def stream():
        remaining = end - start + 1
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
        },
    )
