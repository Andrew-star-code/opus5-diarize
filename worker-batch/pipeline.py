"""Полный цикл обработки одного задания."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import asr
import decode
import diarize
import merge
from config import settings

log = logging.getLogger(__name__)

# Вес каждого этапа в общем прогрессе. ASR и диаризация — самое долгое.
STAGES = {
    "decode": (0.00, 0.05),
    "asr": (0.05, 0.65),
    "diarize": (0.65, 0.95),
    "merge": (0.95, 1.00),
}


def _scaled(report: Callable[[str, float], None], stage: str) -> Callable[[float], None]:
    """Переводит прогресс этапа (0..1) в общий прогресс задания."""
    lo, hi = STAGES[stage]
    return lambda fraction: report(stage, lo + (hi - lo) * max(0.0, min(1.0, fraction)))


def process(job: dict[str, Any], report: Callable[[str, float], None]) -> dict[str, Any]:
    job_id = job["id"]
    source = settings.data_dir / job["audio_path"]
    if not source.is_file():
        raise FileNotFoundError(f"Аудио не найдено: {source}")

    workdir = Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
    try:
        # 1. декодирование в 16 кГц моно
        decode_progress = _scaled(report, "decode")
        decode_progress(0.0)
        wav = decode.to_wav16k(source, workdir / "audio.wav")
        duration = decode.duration_of(wav)
        decode_progress(1.0)

        # 2. распознавание
        asr_progress = _scaled(report, "asr")
        words, asr_segments, language = asr.transcribe(
            wav,
            language=job.get("language"),
            duration=duration,
            prompt=job.get("prompt") or "",
            on_progress=asr_progress,
        )
        asr_progress(1.0)

        # 3. диаризация
        turns = diarize.diarize(
            wav,
            num_speakers=job.get("num_speakers") or settings.num_speakers,
            max_speakers=job.get("max_speakers") or settings.max_speakers,
            on_progress=_scaled(report, "diarize"),
        )

        # 4. слияние
        merge_progress = _scaled(report, "merge")
        merge_progress(0.0)
        if words:
            labels = merge.assign_speakers(words, turns)
            segments = merge.build_segments(words, labels)
        else:
            # Пословных таймингов нет — размечаем целыми фразами.
            log.warning("Пословные тайминги отсутствуют, размечаю по фразам")
            segments = merge.segments_from_asr_only(asr_segments, turns)

        payload = {
            "language": language,
            "duration_sec": duration,
            "model_info": {
                "asr": settings.batch_asr_model,
                "compute_type": settings.compute_type,
                "diarization": settings.diarization_model,
                "pipeline": "batch",
            },
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "speaker": s.speaker,
                    "text": s.text,
                    "words": [
                        {"w": w.w, "s": round(w.s, 3), "e": round(w.e, 3), "p": w.p}
                        for w in s.words
                    ],
                }
                for s in segments
            ],
        }
        merge_progress(1.0)
        log.info("Задание %s: %d реплик, %.1f сек аудио",
                 job_id, len(segments), duration)
        return payload
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
