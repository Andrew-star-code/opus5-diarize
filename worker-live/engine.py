"""Обёртка над WhisperLiveKit.

Мы используем его именно как библиотеку, а не как готовый сервер: своё
хранилище, свой протокол и свой интерфейс. Отсюда и слой нормализации —
формат ответа у WhisperLiveKit между версиями менялся, а ломать из-за
этого весь фронтенд не хочется.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import Any

from config import settings

log = logging.getLogger(__name__)

_engine = None

# "0:05" или "1:02:03" — в части версий тайминги приходят строкой
_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)$")


def get_engine():
    """Модель загружается один раз на процесс и остаётся в VRAM."""
    global _engine
    if _engine is None:
        from whisperlivekit import TranscriptionEngine

        language = settings.default_language
        kwargs: dict[str, Any] = {
            # Именно model_size, а не model: TranscriptionEngine принимает
            # **kwargs и молча глотает незнакомые имена, поэтому опечатка
            # тут не падает, а тихо оставляет модель по умолчанию (base).
            "model_size": settings.live_asr_model,
            "model_cache_dir": os.environ.get("HF_HOME", "/models"),
            "lan": None if language == "auto" else language,
            "diarization": settings.diarization_enabled,
            # vac — контроллер речевой активности, vad — сам детектор.
            # Выключаются вместе: по отдельности смысла нет.
            "vac": settings.live_vad,
            "vad": settings.live_vad,
        }
        if settings.diarization_enabled:
            kwargs["diarization_backend"] = settings.live_diarization
            # По умолчанию diart берёт pyannote/embedding — репозиторий с
            # отдельной лицензией, которую нужно принимать вручную, и без
            # этого модель молча грузится как None, а падает всё позже и
            # непонятно: "NoneType has no attribute to".
            #
            # wespeaker — более новая модель эмбеддингов от тех же авторов,
            # доступ к ней уже открыт той же лицензией, что и к остальной
            # диаризации. Меньше шагов при установке и лучше качество.
            kwargs["embedding_model"] = settings.live_embedding_model
            kwargs["segmentation_model"] = settings.live_segmentation_model

        # backend оставляем 'auto': на Linux с CUDA он и так выбирает
        # faster-whisper, а список допустимых литералов между версиями
        # меняется, и неверный молча уведёт не туда.
        #
        # Точность вычислений WhisperLiveKit наружу не выставляет вовсе.
        # Модели Systran для CTranslate2 хранятся в float16, поэтому на
        # sm_120 (RTX 50xx) проблемы с int8-ядрами не возникает. Если
        # всё же увидите CUBLAS_STATUS_NOT_SUPPORTED в живом режиме —
        # причина будет здесь.

        _check_names(kwargs)
        if settings.diarization_enabled and settings.live_diarization == "diart":
            _patch_diart_kwargs()
            _patch_diart_speaker_labels()
        log.info("Загружаю live-движок: %s", kwargs)
        with _trusted_checkpoints():
            _engine = TranscriptionEngine(**kwargs)
        log.info("Live-движок готов")
    return _engine


@contextmanager
def _trusted_checkpoints():
    """Разрешает распаковку чекпоинтов pyannote на PyTorch 2.6+.

    В 2.6 у torch.load сменилось значение по умолчанию: weights_only
    стало True, и загрузка отклоняет файлы, где кроме тензоров лежат
    служебные объекты. Чекпоинты pyannote 3.x именно такие, поэтому
    diart падает на первой же модели, перечисляя классы по одному —
    TorchVersion, потом Specifications, и так далее.

    Перечислять их бессмысленно: список зависит от версии весов.
    Возвращаем прежнее поведение, но с двумя ограничениями.

    Первое: только на время загрузки движка. Распаковка с
    weights_only=False исполняет код из файла, и держать это включённым
    постоянно нельзя.

    Второе: файлы наши. Они скачаны scripts/prefetch_models.py с
    Hugging Face по принятой лицензии и лежат в локальном каталоге
    моделей — тот же уровень доверия, что и к коду в образе.
    """
    import torch

    original = torch.load

    def patched(*args, **kwargs):
        # Именно присваивание, а не setdefault: lightning, через который
        # pyannote грузит веса, передаёт weights_only=True явным
        # аргументом, и умолчание его не перебивает.
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _patch_diart_kwargs() -> None:
    """Чинит рассинхрон внутри самого WhisperLiveKit 0.2.26.

    Его core.py вызывает собственный diart-бэкенд с именами
    segmentation_model / embedding_model, а DiartDiarization ждёт те же
    аргументы с суффиксом _name. Из-за этого живая диаризация падает на
    TypeError ещё до первого куска аудио, и настройкой это не обойти:
    вызов зашит внутри пакета.

    Переименовываем аргументы на лету. Патч самоотключается: если в
    установленной версии имена уже совпадают, мы ничего не трогаем.
    """
    try:
        import inspect

        from whisperlivekit.diarization import diart_backend
    except Exception:
        return

    cls = diart_backend.DiartDiarization
    if getattr(cls, "_scribe_patched", False):
        return

    params = inspect.signature(cls.__init__).parameters
    renames = {
        wrong: right
        for wrong, right in (
            ("segmentation_model", "segmentation_model_name"),
            ("embedding_model", "embedding_model_name"),
        )
        if wrong not in params and right in params
    }
    if not renames:
        return  # апстрим починили — переходник не нужен

    original = cls.__init__

    def patched(self, *args, **kwargs):
        for wrong, right in renames.items():
            if wrong in kwargs:
                kwargs[right] = kwargs.pop(wrong)
        return original(self, *args, **kwargs)

    cls.__init__ = patched
    cls._scribe_patched = True
    log.warning(
        "WhisperLiveKit зовёт diart с устаревшими именами аргументов, "
        "переименовываю: %s", renames,
    )


def _patch_diart_speaker_labels() -> None:
    """Приводит метку говорящего от diart к числу.

    diart называет говорящих строками: speaker0, speaker1. В самом
    WhisperLiveKit это учтено только наполовину: diart_backend зовёт
    extract_number, а tokens_alignment делает speaker + 1 напрямую и
    падает на TypeError — метки в живом транскрипте не проставляются
    вовсе, хотя диаризация при этом честно работает.

    Чиним в источнике: сегмент сразу хранит число. Заодно делаем
    extract_number терпимым к числу — иначе сломается второй путь,
    который его вызывает.
    """
    try:
        from whisperlivekit.diarization import diart_backend
        from whisperlivekit.diarization import utils as diar_utils
    except Exception:
        return

    if getattr(diart_backend, "_scribe_labels_patched", False):
        return

    original_extract = diar_utils.extract_number

    def tolerant_extract(value):
        return value if isinstance(value, int) else original_extract(value)

    diar_utils.extract_number = tolerant_extract
    if hasattr(diart_backend, "extract_number"):
        diart_backend.extract_number = tolerant_extract

    segment_cls = getattr(diart_backend, "SpeakerSegment", None)
    if segment_cls is not None:
        original_init = segment_cls.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if isinstance(getattr(self, "speaker", None), str):
                self.speaker = tolerant_extract(self.speaker)

        segment_cls.__init__ = init

    diart_backend._scribe_labels_patched = True
    log.warning("Метки говорящих от diart приводятся к числу: апстрим "
                "ожидает их в разном виде в разных местах")


def _check_names(kwargs: dict[str, Any]) -> None:
    """Сверяет имена параметров с конфигурацией установленной версии.

    Движок принимает **kwargs и на неизвестное имя не ругается — молча
    берёт значение по умолчанию. Один раз это уже стоило нам модели
    base вместо large-v3-turbo, поэтому расхождение должно быть громким.
    """
    try:
        import dataclasses

        from whisperlivekit.config import WhisperLiveKitConfig
    except Exception:
        return  # структура пакета изменилась — не повод не стартовать

    known = {f.name for f in dataclasses.fields(WhisperLiveKitConfig)}
    unknown = sorted(set(kwargs) - known)
    if unknown:
        log.error(
            "WhisperLiveKit не знает параметры %s — они будут проигнорированы, "
            "и движок возьмёт значения по умолчанию. Сверьтесь с полями "
            "WhisperLiveKitConfig в установленной версии пакета.",
            unknown,
        )


async def new_processor():
    """Отдельный обработчик на каждое соединение (движок при этом общий)."""
    from whisperlivekit import AudioProcessor

    engine = get_engine()
    try:
        return AudioProcessor(transcription_engine=engine)
    except TypeError:
        return AudioProcessor(engine)


# ─── нормализация ответов ───────────────────────────────────────

def _seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = _CLOCK_RE.match(str(value).strip())
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)


def _speaker_label(raw: Any) -> str | None:
    """WhisperLiveKit нумерует говорящих целыми, отрицательные значения
    означают тишину или ещё не определённого говорящего."""
    if raw is None:
        return None
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return str(raw)
    if number < 0:
        return None
    return f"SPEAKER_{number:02d}"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize(response: Any) -> dict[str, Any]:
    """Приводит ответ движка к формату, который понимает наш фронтенд."""
    raw_lines = _get(response, "lines") or []
    lines = []
    for line in raw_lines:
        text = (_get(line, "text") or "").strip()
        if not text:
            continue
        lines.append(
            {
                "start": _seconds(_get(line, "beg", _get(line, "start"))),
                "end": _seconds(_get(line, "end")),
                "speaker": _speaker_label(_get(line, "speaker")),
                "text": text,
            }
        )

    return {
        "lines": lines,
        # Гипотеза, которую движок ещё может переписать. Показываем её
        # серым: честнее, чем выдавать неустоявшийся текст за готовый.
        "buffer": (_get(response, "buffer_transcription") or "").strip(),
        "buffer_speaker": (_get(response, "buffer_diarization") or "").strip(),
        "status": _get(response, "status") or "active",
    }


def to_result_segments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Черновой транскрипт в том же виде, в каком его отдаёт batch-воркер.

    Пословных таймингов у стрима нет — отдаём пустой список слов;
    экспортёры на этот случай умеют работать по границам реплики.
    """
    return [
        {
            "start": round(line["start"], 3),
            "end": round(max(line["end"], line["start"]), 3),
            "speaker": line["speaker"] or "SPEAKER_00",
            "text": line["text"],
            "words": [],
        }
        for line in lines
        if line["text"]
    ]
