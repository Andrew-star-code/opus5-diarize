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
import polish
from config import settings

log = logging.getLogger(__name__)

# Вес каждого этапа в общем прогрессе. ASR и диаризация — самое долгое.
STAGES = {
    "decode": (0.00, 0.05),
    "asr": (0.05, 0.65),
    "diarize": (0.65, 0.95),
    "merge": (0.95, 0.96),
    "polish": (0.96, 1.00),
}


def _scaled(report: Callable[[str, float], None], stage: str) -> Callable[[float], None]:
    """Переводит прогресс этапа (0..1) в общий прогресс задания."""
    lo, hi = STAGES[stage]
    return lambda fraction: report(stage, lo + (hi - lo) * max(0.0, min(1.0, fraction)))


def _polish_labels(
    words: list[merge.Word], labels: list[str | None], progress: Callable[[float], None]
) -> tuple[list[str | None], str | None]:
    """Поправка стыков реплик языковой моделью — надстройка, может не случиться.

    Возвращает метки и имя модели, если она поработала. Выключена,
    недоступна или сбоит — метки остаются как были.
    """
    if not settings.llm_fix_speakers:
        return labels, None
    from llm import LLM

    progress(0.0)
    client = LLM()
    try:
        if not client.available():
            return labels, None
        fixed, stats = polish.adjust_boundaries(
            words, labels, client.ask, max_shift=settings.llm_max_shift
        )
        log.info("Стыки реплик: границ %d, спрошено %d, сдвинуто %d (%d слов)",
                 stats["boundaries"], stats["asked"], stats["moved"], stats["words"])
        return fixed, (client.model if stats["asked"] else None)
    finally:
        client.close()
        progress(0.5)


def _describe(segments: list[merge.Segment], progress: Callable[[float], None]) -> dict[str, Any]:
    """Название, краткое содержание, подсказки имён — языковой моделью, если она есть."""
    if not settings.llm_notes or not segments:
        return {}
    from llm import LLM

    client = LLM()
    try:
        if not client.available():
            return {}
        notes = polish.describe(segments, client.ask)
        log.info("Полировка: название %s, пунктов краткого содержания %d, подсказано имён %d",
                 "есть" if notes.get("title") else "нет",
                 len(notes.get("summary", "").splitlines()), len(notes.get("speaker_names", {})))
        return notes
    finally:
        client.close()
        progress(1.0)


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
        polished_by = None
        if words:
            labels = merge.assign_speakers(
                words, turns, smooth=settings.smooth_speaker_turns
            )
            merge_progress(1.0)
            # 5. стыки реплик по смыслу — языковой моделью, если она есть
            labels, polished_by = _polish_labels(words, labels, _scaled(report, "polish"))
            segments = merge.build_segments(words, labels)
        else:
            # Пословных таймингов нет — размечаем целыми фразами.
            log.warning("Пословные тайминги отсутствуют, размечаю по фразам")
            segments = merge.segments_from_asr_only(asr_segments, turns)

        # 6. название, краткое содержание, имена — тоже языковой моделью
        notes = _describe(segments, _scaled(report, "polish"))
        if notes and not polished_by:
            polished_by = settings.llm_model

        payload = {
            "language": language,
            "duration_sec": duration,
            "model_info": {
                "asr": settings.batch_asr_model,
                "compute_type": settings.compute_type,
                "diarization": settings.diarization_model,
                "pipeline": "batch",
                **({"llm": polished_by} if polished_by else {}),
            },
            "title": notes.get("title"),
            "summary": notes.get("summary", ""),
            "speaker_names": notes.get("speaker_names", {}),
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
        log.info("Задание %s: %d реплик, %.1f сек аудио",
                 job_id, len(segments), duration)
        return payload
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
