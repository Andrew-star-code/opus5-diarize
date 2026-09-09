"""Выгрузка транскрипта в файл."""
from __future__ import annotations

import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlmodel import Session as DBSession

from ..exporters import FORMATS
from ..models import Session
from ..services import transcripts
from .deps import get_db, get_session_or_404

router = APIRouter(prefix="/api", tags=["exports"])

_UNSAFE = re.compile(r"[^\w\-. ]+", re.UNICODE)


def _filename(title: str, ext: str) -> str:
    stem = _UNSAFE.sub("", title).strip() or "transcript"
    return f"{stem[:80]}.{ext}"


@router.get("/sessions/{session_id}/export")
def export_session(
    format: str = Query("txt", pattern="^[a-z]+$"),
    session: Session = Depends(get_session_or_404),
    db: DBSession = Depends(get_db),
):
    fmt = FORMATS.get(format)
    if fmt is None:
        raise HTTPException(
            400, f"Неизвестный формат. Доступны: {', '.join(FORMATS)}"
        )

    detail = transcripts.to_detail(db, session)
    if not detail.segments:
        raise HTTPException(409, "Транскрипт ещё не готов")

    body = fmt.render(detail, transcripts.compute_stats(db, session))
    name = _filename(session.title, fmt.ext)

    return Response(
        content=body,
        media_type=fmt.mime,
        headers={
            # filename* с UTF-8 — иначе кириллица в имени файла превращается
            # в мусор в Chrome под Windows.
            "Content-Disposition":
                f"attachment; filename=\"transcript.{fmt.ext}\"; "
                f"filename*=UTF-8''{quote(name)}"
        },
    )
