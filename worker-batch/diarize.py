"""Диаризация через pyannote.audio.

community-1 заметно устойчивее 3.1 на шумных реальных записях и не привязан
к языку — важно, потому что стриминговый Sortformer от NVIDIA обучен
преимущественно на английском.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from config import settings
from merge import Turn, tiny_speaker_renames

log = logging.getLogger(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        log.info("Загружаю пайплайн диаризации %s", settings.diarization_model)
        token = settings.hf_token or None
        try:
            # pyannote.audio 4.x
            _pipeline = Pipeline.from_pretrained(settings.diarization_model, token=token)
        except TypeError:
            # 3.x называл этот аргумент иначе
            _pipeline = Pipeline.from_pretrained(
                settings.diarization_model, use_auth_token=token
            )

        if _pipeline is None:
            raise RuntimeError(
                f"Не удалось загрузить {settings.diarization_model}. "
                "Проверьте, что веса скачаны в /models (scripts/prefetch_models.py) "
                "и что лицензия модели принята на huggingface.co."
            )

        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
        else:
            log.warning("CUDA недоступна — диаризация пойдёт на CPU и будет медленной")
    return _pipeline


class _ProgressHook:
    """Переводит внутренние шаги pyannote в один общий процент."""

    # Порядок шагов пайплайна и их вес в общем прогрессе
    WEIGHTS = {
        "segmentation": 0.45,
        "embeddings": 0.40,
        "speaker_counting": 0.05,
        "discrete_diarization": 0.10,
    }

    def __init__(self, on_progress: Callable[[float], None] | None):
        self.on_progress = on_progress
        self.done = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if self.on_progress is None:
            return
        weight = self.WEIGHTS.get(step_name, 0.0)
        if total and completed is not None and total > 0:
            fraction = min(1.0, completed / total)
            self.on_progress(min(1.0, self.done + weight * fraction))
            if completed >= total:
                self.done = min(1.0, self.done + weight)
        else:
            self.on_progress(min(1.0, self.done))


def _load_waveform(wav: Path) -> dict:
    """Читает WAV в память и отдаёт в том виде, какой ждёт pyannote.

    Файл пайплайну не передаём сознательно: pyannote 4.x декодирует
    через torchcodec, а он в этом образе не собирается (несовпадение
    версий ffmpeg) и валится прямо в момент чтения. Аудио к этому шагу
    уже приведено к 16 кГц моно, так что читаем сами — заодно исчезает
    лишний декодер в цепочке.
    """
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    # soundfile отдаёт (отсчёты, каналы), pyannote ждёт (каналы, отсчёты)
    return {"waveform": torch.from_numpy(data.T.copy()), "sample_rate": sample_rate}


def _annotation(output):
    """Достаёт разметку из результата пайплайна.

    pyannote 4.x возвращает DiarizeOutput с двумя вариантами разметки,
    3.x — сразу Annotation. Берём exclusive-вариант: в нём на каждый
    момент времени приходится ровно один говорящий, а нам и нужно
    приписать каждому слову одного автора.

    Обычная разметка с наложениями проверялась как альтернатива на трёх
    встречах AMI: в среднем путаница на словах чуть ниже (2,16% против
    2,20%), но на двух записях из трёх — выше. Это шум, а не выигрыш.
    """
    for field in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, field, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation
    if hasattr(output, "itertracks"):
        return output
    raise TypeError(
        f"Не понимаю результат диаризации: {type(output).__name__}. "
        "Проверьте совместимость версии pyannote.audio."
    )


def _tiny_speaker_renames(output) -> dict[str, str]:
    """Кого из найденных говорящих слить с другим — см. merge.tiny_speaker_renames.

    Эмбеддинги идут в порядке меток обычной, а не эксклюзивной разметки:
    в эксклюзивной может не оказаться говорящего, который звучал только
    в наложениях, и тогда индексы разъехались бы.
    """
    regular = getattr(output, "speaker_diarization", None)
    embeddings = getattr(output, "speaker_embeddings", None)
    if regular is None or embeddings is None:
        return {}  # pyannote 3.x эмбеддингов не отдаёт
    labels = regular.labels()
    if len(labels) != len(embeddings):
        return {}
    renames = tiny_speaker_renames(
        [str(label) for label in labels],
        [regular.label_duration(label) for label in labels],
        [[float(x) for x in row] for row in embeddings],
        max_share=settings.tiny_speaker_max_share,
        min_similarity=settings.tiny_speaker_min_similarity,
    )
    if renames:
        log.info("Слиты лишние говорящие: %s", renames)
    return renames


def diarize(
    wav: Path,
    *,
    num_speakers: int = 0,
    max_speakers: int = 8,
    on_progress: Callable[[float], None] | None = None,
) -> list[Turn]:
    pipeline = get_pipeline()

    kwargs: dict = {}
    if num_speakers and num_speakers > 0:
        # Точное число известно — не даём модели фантазировать.
        kwargs["num_speakers"] = num_speakers
    elif max_speakers and max_speakers > 0:
        kwargs["min_speakers"] = 1
        kwargs["max_speakers"] = max_speakers

    audio = _load_waveform(wav)
    hook = _ProgressHook(on_progress)
    with hook:
        output = pipeline(audio, hook=hook, **kwargs)

    # Число задано человеком — модель уже в него уложилась, и сливать
    # кого-то поверх значило бы спорить с его указанием.
    renames = {} if kwargs.get("num_speakers") else _tiny_speaker_renames(output)

    turns = [
        Turn(
            start=float(segment.start),
            end=float(segment.end),
            label=renames.get(str(label), str(label)),
        )
        for segment, _track, label in _annotation(output).itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)

    speakers = {t.label for t in turns}
    log.info("Диаризация: %d турнов, %d говорящих", len(turns), len(speakers))
    return turns


def warm_up() -> None:
    """Прогрев на старте, чтобы первое задание не ждало загрузку весов."""
    if os.environ.get("SKIP_WARMUP"):
        return
    try:
        get_pipeline()
    except Exception:
        log.exception("Прогрев диаризации не удался — попробуем на первом задании")
