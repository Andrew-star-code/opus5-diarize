"""Общие зависимости роутеров."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Path
from sqlmodel import Session as DBSession

from ..db import get_session
from ..models import Session


def get_db() -> DBSession:  # обёртка ради читаемости сигнатур
    yield from get_session()


def get_session_or_404(
    session_id: str = Path(..., min_length=4, max_length=64),
    db: DBSession = Depends(get_db),
) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, "Сессия не найдена")
    return session
