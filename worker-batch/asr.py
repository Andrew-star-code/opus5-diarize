"""Распознавание речи через faster-whisper."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from config import settings
from hallucinations import drop_hallucinations, hallucinated_words
from merge import Segment, Word

log = logging.getLogger(__name__)

_model = None

# Предел длины словаря. Он подмешивается в промпт КАЖДОГО окна, поэтому
# длинный вытесняет оттуда накопленный контекст речи. Замерено на живой
# записи: описание встречи на 160 символов стоило 30% распознанных слов —
# из транскрипта выпали 25 секунд подряд. Короткий список терминов такого
# эффекта не даёт.
MAX_HOTWORDS_CHARS = 200


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True")


def get_model():
    """Модель грузится один раз на весь процесс и живёт в VRAM.

    Перезагружать её на каждое задание — это +20-30 секунд впустую.
    """
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        log.info("Загружаю ASR-модель %s (%s)",
                 settings.batch_asr_model, settings.compute_type)
        _model = WhisperModel(
            settings.batch_asr_model,
            device="cuda",
            # ВНИМАНИЕ: на RTX 50xx (sm_120) int8 в CTranslate2 падает
            # с CUBLAS_STATUS_NOT_SUPPORTED. Держим float16.
            compute_type=settings.compute_type,
            # download_root не задаём: faster-whisper возьмёт стандартный
            # $HF_HOME/hub, туда же кладёт веса скрипт предзагрузки.
            local_files_only=_offline(),
        )
    return _model


def transcribe(
    wav: Path,
    *,
    language: str | None,
    duration: float,
    prompt: str = "",
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[Word], list[Segment], str]:
    """Возвращает (слова, сегменты Whisper, определённый язык)."""
    model = get_model()
    lang = None if (language in (None, "", "auto")) else language
    prompt = (prompt or settings.initial_prompt or "").strip()
    if len(prompt) > MAX_HOTWORDS_CHARS:
        log.warning(
            "Словарь обрезан до %d символов: длинный подмешивается в каждое "
            "окно и вытесняет контекст речи", MAX_HOTWORDS_CHARS,
        )
        prompt = prompt[:MAX_HOTWORDS_CHARS].rsplit(",", 1)[0]

    segments_iter, info = model.transcribe(
        str(wav),
        language=lang,
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        # VAD по умолчанию ВЫКЛЮЧЕН, и это осознанно.
        #
        # Он задумывался как защита от галлюцинаций Whisper на тишине, но
        # на реальном материале вредит куда сильнее, чем помогает: под
        # речью в фильме, записи из комнаты или созвоне почти всегда есть
        # музыка, шум и эффекты, и Silero VAD такие куски за речь не
        # считает. На проверочной записи фильтр срезал больше половины
        # слов — 28 против 58 — и заодно лишил текст пунктуации, потому
        # что уцелевшие обрывки распознаются вне контекста фразы.
        #
        # Терять речь молча хуже, чем изредка получить лишнюю строку на
        # тишине: галлюцинацию человек видит и удалит, а пропавшего текста
        # он не заметит. Включайте VAD_FILTER=true только для заведомо
        # чистых записей с длинными паузами.
        vad_filter=settings.vad_filter,
        vad_parameters={"min_silence_duration_ms": settings.vad_min_silence_ms},
        # Контекст предыдущих фрагментов включён: без него Whisper теряет
        # ход фразы на границах окон, и это видно по пропадающей пунктуации
        # и разнобою в написании имён.
        #
        # Риск здесь в том, что одна галлюцинация протащится в контекст и
        # испортит всё дальше. Раньше я просто выключал контекст целиком —
        # это лечило симптом ценой качества. Правильный инструмент другой:
        # пороги ниже заставляют faster-whisper распознать сорвавшийся
        # фрагмент и сбросить контекст, не теряя его на нормальных.
        condition_on_previous_text=settings.use_previous_context,
        # Зацикливание («трон трон трон») даёт аномально хорошо сжимаемый
        # текст — на этом его и ловим.
        compression_ratio_threshold=2.4,
        # Фрагмент, который модель сама считает не-речью, выбрасываем.
        no_speech_threshold=0.6,
        # Текст, «произнесённый» в тишине, отбрасывается по пословным
        # таймингам. Работает только вместе с word_timestamps=True.
        hallucination_silence_threshold=2.0,
        # Словарь имён, терминов и названий.
        #
        # Именно hotwords, а не initial_prompt, хотя по названию логичнее
        # выглядит второй. Разница в том, как faster-whisper собирает
        # подсказку для каждого окна: initial_prompt лишь засеивает начало
        # записи, и через пару минут его вытесняет накопленный текст, а
        # hotwords подмешивается в промпт каждого окна и работает до
        # конца записи. Для словаря нужно именно второе.
        hotwords=prompt or None,
    )

    words: list[Word] = []
    asr_segments: list[Segment] = []

    dropped = 0
    for seg in segments_iter:
        seg_words = [
            Word(w=w.word.strip(), s=w.start, e=w.end, p=round(float(w.probability or 1.0), 3))
            for w in (seg.words or []) if w.word.strip()
        ]
        text = seg.text.strip()
        if settings.drop_hallucinations:
            # Строчки из титров, которые Whisper выдумывает на музыке и
            # тишине («Субтитры сделал DimaTorzok»), — см. hallucinations.py.
            # Слова уходят вместе с таймингами, до разметки говорящих.
            marks = hallucinated_words([w.w for w in seg_words])
            dropped += sum(marks)
            seg_words = [w for w, bad in zip(seg_words, marks) if not bad]
            text = drop_hallucinations(text).strip()
        if text:
            asr_segments.append(Segment(start=seg.start, end=seg.end, speaker=None, text=text))
        words.extend(seg_words)
        if on_progress and duration > 0:
            on_progress(min(1.0, seg.end / duration))
    if dropped:
        log.info("ASR: выброшено %d слов из строчек титров", dropped)

    detected = getattr(info, "language", None) or lang or "unknown"
    log.info("ASR: %d слов, %d фраз, язык=%s", len(words), len(asr_segments), detected)
    return words, asr_segments, detected
