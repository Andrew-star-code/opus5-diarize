"""Мини-шина событий для SSE.

Воркеры шлют прогресс в backend по HTTP, backend раздаёт его браузерам
через Server-Sent Events. Всё в памяти одного процесса — переживать
перезапуск тут нечему.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
_QUEUE_SIZE = 64


def subscribe(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
    _subscribers[session_id].add(q)
    return q


def unsubscribe(session_id: str, q: asyncio.Queue) -> None:
    subs = _subscribers.get(session_id)
    if not subs:
        return
    subs.discard(q)
    if not subs:
        _subscribers.pop(session_id, None)


def publish(session_id: str, event: str, data: dict[str, Any]) -> None:
    """Не блокируется: если подписчик не успевает читать, событие теряется.

    Прогресс — это поток снимков состояния, потеря промежуточного кадра
    ни на что не влияет; терминальные события клиент всё равно
    перепроверяет запросом сессии.
    """
    payload = {"event": event, "data": data}
    for q in list(_subscribers.get(session_id, ())):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def format_sse(payload: dict[str, Any]) -> str:
    return f"event: {payload['event']}\ndata: {json.dumps(payload['data'], ensure_ascii=False)}\n\n"
