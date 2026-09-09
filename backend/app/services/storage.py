"""Работа с файлами аудио: пути, длительность, удаление."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

# Что принимаем на загрузку. Всё равно всё декодируется ffmpeg'ом,
# список нужен только чтобы отсечь очевидный мусор.
ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".wma", ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".amr", ".3gp",
}


def session_dir(session_id: str) -> Path:
    d = settings.audio_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel_path(path: Path) -> str:
    """Путь относительно DATA_DIR — воркеры монтируют его в ту же точку."""
    return path.relative_to(settings.data_dir).as_posix()


def abs_path(rel: str) -> Path:
    return settings.data_dir / rel


def remove_session_files(session_id: str) -> None:
    d = settings.audio_dir / session_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def probe_duration(path: Path) -> float:
    """Длительность в секундах через ffprobe. 0.0, если определить не вышло."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception as exc:  # ffprobe не смог — не повод падать
        log.warning("ffprobe не смог прочитать %s: %s", path, exc)
        return 0.0
