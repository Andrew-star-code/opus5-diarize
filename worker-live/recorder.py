"""Параллельная запись живого потока в WAV.

Стриминговый транскрипт — это черновик. Чтобы потом прогнать запись
полноценным batch-пайплайном и получить чистовик, нужно сохранить исходное
аудио. Пишем те же байты, что уходят в распознавание, во второй ffmpeg.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class LiveRecorder:
    """Принимает куски webm/opus от браузера и складывает их в WAV 16 кГц."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self._proc: asyncio.subprocess.Process | None = None
        self._bytes = 0
        self._failed = False

    async def start(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", "pipe:0",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(self.target),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info("Пишу живое аудио в %s", self.target)

    async def feed(self, chunk: bytes) -> None:
        if self._proc is None or self._proc.stdin is None or self._failed:
            return
        try:
            self._proc.stdin.write(chunk)
            await self._proc.stdin.drain()
            self._bytes += len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Запись сорвалась — но живой транскрипт продолжаем отдавать:
            # потерять черновик из-за проблемы с файлом было бы обидно.
            self._failed = True
            log.warning("Поток записи оборвался, продолжаю без сохранения аудио")

    async def stop(self) -> Path | None:
        if self._proc is None:
            return None
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
                await self._proc.wait()
            # ffmpeg дописывает размер в заголовок WAV при закрытии,
            # поэтому дожидаемся выхода, а не убиваем процесс.
            await asyncio.wait_for(self._proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            log.warning("ffmpeg не завершился за 30 с, убиваю")
            self._proc.kill()
        except Exception as exc:
            log.warning("Ошибка при закрытии записи: %s", exc)

        if self._failed or not self.target.exists() or self.target.stat().st_size < 1024:
            log.warning("Аудио записи пустое или повреждено: %s", self.target)
            return None
        log.info("Записано %.1f МБ в %s", self.target.stat().st_size / 1e6, self.target)
        return self.target
