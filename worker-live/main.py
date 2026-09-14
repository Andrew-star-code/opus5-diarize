"""Live-режим: WebSocket, стриминговое распознавание, черновой транскрипт.

Поток данных:
    браузер --webm/opus, куски по 200 мс--> WebSocket
                                             |
                          +------------------+------------------+
                          |                                     |
                  WhisperLiveKit                          ffmpeg -> WAV
                (текст + говорящие)                  (для последующей доводки)
                          |
                     WebSocket --> браузер (подтверждённое + гипотеза)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import engine
from client import BackendClient
from config import settings
from recorder import LiveRecorder

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("worker-live")

# Больше одновременных сессий 12 ГБ VRAM не потянут: на каждую нужен
# свой буфер и своё состояние диаризации.
_slots = asyncio.Semaphore(settings.max_live_sessions)
_backend = BackendClient()

# Сколько ждём остаточный текст после команды остановки
FLUSH_TIMEOUT = 20.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Прогреваем модель на старте, а не на первом подключении: иначе первый
    # пользователь минуту смотрит на пустой экран.
    try:
        await asyncio.get_running_loop().run_in_executor(None, engine.get_engine)
    except Exception:
        log.exception("Не удалось прогреть движок, попробую при подключении")
    yield
    await _backend.aclose()


app = FastAPI(title="Scribe Live", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "free_slots": _slots._value,
        "max_live_sessions": settings.max_live_sessions,
        "diarization": settings.live_diarization if settings.diarization_enabled else "off",
        "model": settings.live_asr_model,
    }


async def _send(ws: WebSocket, payload: dict) -> None:
    if ws.client_state is WebSocketState.CONNECTED:
        with suppress(Exception):
            await ws.send_text(json.dumps(payload, ensure_ascii=False))


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()

    if _slots.locked():
        await _send(ws, {
            "type": "error",
            "code": "busy",
            "message": (
                f"Уже идёт записей: {settings.max_live_sessions}. "
                "Больше видеопамять не позволяет, попробуйте позже."
            ),
        })
        await ws.close(code=1013)
        return

    async with _slots:
        await _run_session(ws)


async def _run_session(ws: WebSocket) -> None:
    params = ws.query_params
    language = params.get("language") or settings.default_language
    title = params.get("title") or None

    try:
        session = await _backend.create_session(title, language)
    except Exception as exc:
        log.exception("Не удалось создать сессию")
        await _send(ws, {"type": "error", "code": "backend",
                         "message": f"Backend недоступен: {exc}"})
        await ws.close(code=1011)
        return

    session_id = session["id"]
    audio_target = settings.audio_dir / session_id / "live.wav"
    recorder = LiveRecorder(audio_target)
    started = time.monotonic()
    latest_lines: list[dict] = []

    await _send(ws, {
        "type": "session",
        "session_id": session_id,
        "title": session.get("title", ""),
        "language": language,
        "diarization": settings.diarization_enabled,
        "model": settings.live_asr_model,
    })

    processor = None
    pump: asyncio.Task | None = None

    try:
        await recorder.start()
        processor = await engine.new_processor()
        results = await processor.create_tasks()

        async def pump_results() -> None:
            nonlocal latest_lines
            async for response in results:
                state = engine.normalize(response)
                if state["lines"]:
                    latest_lines = state["lines"]
                await _send(ws, {
                    "type": "transcript",
                    "lines": state["lines"],
                    "buffer": state["buffer"],
                    "lag": state["lag"],
                    "elapsed": round(time.monotonic() - started, 2),
                })

        pump = asyncio.create_task(pump_results())

        stopped_by_client = await _receive_loop(ws, processor, recorder)

        # Пустой кусок означает конец потока: движок дораспознаёт хвост.
        with suppress(Exception):
            await processor.process_audio(b"")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pump), timeout=FLUSH_TIMEOUT)

        log.info("Сессия %s завершена (%s)", session_id,
                 "по команде" if stopped_by_client else "обрыв соединения")

    except Exception:
        log.exception("Ошибка live-сессии %s", session_id)
        await _send(ws, {"type": "error", "code": "internal",
                         "message": "Внутренняя ошибка распознавания"})
    finally:
        if pump is not None and not pump.done():
            pump.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pump
        if processor is not None:
            with suppress(Exception):
                await processor.cleanup()

        saved = await recorder.stop()
        duration = time.monotonic() - started
        await _finalize(ws, session_id, saved, duration, latest_lines)


async def _receive_loop(ws: WebSocket, processor, recorder: LiveRecorder) -> bool:
    """Читает сообщения клиента. True, если остановились по его команде."""
    while True:
        try:
            message = await ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return False

        if message["type"] == "websocket.disconnect":
            return False

        chunk = message.get("bytes")
        if chunk:
            # Один и тот же кусок уходит и в распознавание, и в файл.
            await processor.process_audio(chunk)
            await recorder.feed(chunk)
            continue

        text = message.get("text")
        if not text:
            continue
        try:
            command = json.loads(text)
        except json.JSONDecodeError:
            continue
        if command.get("type") == "stop":
            return True


async def _finalize(
    ws: WebSocket,
    session_id: str,
    saved_audio,
    duration: float,
    lines: list[dict],
) -> None:
    audio_path = None
    if saved_audio is not None:
        audio_path = saved_audio.relative_to(settings.data_dir).as_posix()

    segments = engine.to_result_segments(lines)
    refine = bool(settings.auto_refine_live and audio_path and segments)

    try:
        await _backend.finalize(
            session_id,
            audio_path=audio_path,
            duration_sec=duration,
            segments=segments,
            model_info={
                "asr": settings.live_asr_model,
                "compute_type": settings.compute_type,
                "diarization": settings.live_diarization,
                "pipeline": "live",
            },
            refine=refine,
        )
        await _send(ws, {
            "type": "finalized",
            "session_id": session_id,
            "segments": len(segments),
            # По этому флагу клиент решает, показывать ли плашку
            # об уточнении и подписываться ли на SSE прогресса.
            "refine": refine,
        })
    except Exception as exc:
        log.exception("Не удалось сохранить сессию %s", session_id)
        await _send(ws, {"type": "error", "code": "save",
                         "message": f"Не удалось сохранить запись: {exc}"})

    if ws.client_state is WebSocketState.CONNECTED:
        with suppress(Exception):
            await ws.close()
