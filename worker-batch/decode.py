"""Приведение любого входного файла к 16 кГц моно WAV.

И Whisper, и pyannote работают именно с этим форматом. Декодируем один раз
и явно, вместо того чтобы каждая библиотека дёргала свой декодер.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class DecodeError(RuntimeError):
    pass


def to_wav16k(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(src),
        "-vn",                  # видеодорожку выбрасываем
        "-ac", "1",             # моно
        "-ar", "16000",         # 16 кГц
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise DecodeError(
            f"ffmpeg не смог декодировать {src.name}: " + " | ".join(tail)
        )
    log.info("Декодировано: %s → %s (%.1f МБ)", src.name, dst.name,
             dst.stat().st_size / 1e6)
    return dst


def duration_of(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0
