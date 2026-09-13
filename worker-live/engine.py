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
import threading
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
            # Своя диаризация на каждую запись. Прежний переходник имён
            # аргументов (_patch_diart_kwargs) больше не нужен: наш класс
            # принимает оба варианта имён сам.
            _install_session_diart()
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


class _SharedDiartModels:
    """То, что TranscriptionEngine хранит как diarization_model: только веса.

    Ни конвейера, ни потока, ни цикла событий — поэтому создаётся где
    угодно, в том числе при прогреве в пуле потоков. Имена аргументов те,
    с которыми его зовёт core.py WhisperLiveKit; варианты с суффиксом _name
    приняты на случай, если апстрим приведёт вызов к сигнатуре
    DiartDiarization.
    """

    def __init__(
        self,
        block_duration: float = 1.5,
        sample_rate: int = 16000,
        segmentation_model: str | None = None,
        embedding_model: str | None = None,
        segmentation_model_name: str | None = None,
        embedding_model_name: str | None = None,
        **_ignored: Any,
    ) -> None:
        import diart.models as models
        import torch

        self.block_duration = block_duration
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation = models.SegmentationModel.from_pretrained(
            segmentation_model or segmentation_model_name or settings.live_segmentation_model
        )
        self.embedding = models.EmbeddingModel.from_pretrained(
            embedding_model or embedding_model_name or settings.live_embedding_model
        )
        # Модели diart ленивые. Без явной загрузки веса читались бы с диска
        # на первой записи — уже вне _trusted_checkpoints(), где torch.load
        # их отвергнет.
        for model in (self.segmentation, self.embedding):
            model.to(self.device)  # LazyModel.to() сам вызывает load()
            if hasattr(model, "eval"):
                model.eval()

    def new_config(self):
        from diart import SpeakerDiarizationConfig

        return SpeakerDiarizationConfig(
            segmentation=self.segmentation,
            embedding=self.embedding,
            # У diart по умолчанию 20 — берём тот же предел, что у batch.
            max_speakers=settings.max_speakers,
            device=self.device,
            sample_rate=self.sample_rate,
        )


class _SessionDiart:
    """Диаризация одной живой записи.

    Интерфейс ровно тот, что ждёт AudioProcessor от diart-бэкенда. Буфера
    buffer_audio у него нет намеренно: по этому признаку AudioProcessor
    понимает, что сегменты накопительные и их надо заменять, а не дописывать.
    """

    def __init__(self, shared: _SharedDiartModels) -> None:
        from diart import SpeakerDiarization
        from diart.inference import StreamingInference
        from whisperlivekit.diarization.diart_backend import (
            DiarizationObserver,
            WebSocketAudioSource,
        )

        self.observer = DiarizationObserver()
        self.source = WebSocketAudioSource(
            uri="scribe-live",
            sample_rate=shared.sample_rate,
            block_duration=shared.block_duration,
        )
        self.inference = StreamingInference(
            pipeline=SpeakerDiarization(config=shared.new_config()),
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        # Свой поток, а не run_in_executor: разбор живёт всю запись, и держать
        # ради него слот общего пула незачем. Цикл событий ему не нужен.
        self._thread = threading.Thread(target=self._run, name="diart-session", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.inference()
        except Exception:
            log.exception("Поток diart остановился с ошибкой")
        finally:
            # По этой строке в журнале видно, что запись за собой убрала.
            # Ждать потока в close() нельзя: его зовут из цикла событий,
            # и блокировка остановила бы все живые записи разом.
            log.info("Поток diart завершён")

    def insert_silence(self, duration: float) -> None:
        self.observer.global_time_offset += duration

    def insert_audio_chunk(self, pcm_array) -> None:
        self.source.push_audio(pcm_array)

    async def diarize(self):
        return self.observer.get_segments()

    def close(self) -> None:
        self.source.close()


def _install_session_diart() -> None:
    """Даёт каждой живой записи собственную диаризацию diart.

    В WhisperLiveKit 0.2.26 diart — одиночка на весь процесс:
    TranscriptionEngine создаёт один DiartDiarization, а
    online_diarization_factory отдаёт этот же объект каждой записи. Внутри
    у него один источник звука, один поток разбора и одна история сегментов.

    - Конец первой записи закрывал общий источник: AudioProcessor.cleanup()
      зовёт close() у «своей» диаризации, а она общая. Дальше источник молча
      выбрасывал весь звук, сегментов не было, и каждое слово получало
      говорящего по умолчанию — «Спикер 1». По логам: в первой записи после
      старта воркера diart нашёл двоих, во второй и третьей — ни одного
      сегмента.
    - Вторая запись и без того попадала бы в чужую историю: время diart
      отсчитывается от первого звука процесса, а слова — от начала записи.
    - Две записи одновременно смешивали бы звук в одном потоке.

    Разводим так: веса грузятся один раз на процесс и остаются общими — это
    дорого, а состояния в них нет. Конвейер со своими кластерами говорящих,
    источник и наблюдатель создаются на каждую запись и умирают вместе с ней.

    Заодно проходит прогрев. Старый DiartDiarization запускал разбор через
    asyncio.get_event_loop() и в пуле потоков, где цикла нет, падал: прогрев
    не удавался ни разу, и модель грузилась уже на первом подключении.

    При обновлении WhisperLiveKit загляните в online_diarization_factory в
    core.py: если diart там начнут создавать на каждую запись, переходник
    станет лишним.
    """
    try:
        from whisperlivekit import audio_processor
        from whisperlivekit.diarization import diart_backend
    except Exception:
        # Молча отступать нельзя: без подмены живая диаризация снова
        # работает только в первой записи после старта воркера.
        log.exception("Не удалось подменить diart — живая диаризация будет сломана")
        return

    if getattr(diart_backend, "_scribe_session_diart", False):
        return

    # core.py импортирует класс в момент вызова, так что подмена в модуле
    # действует и на него.
    diart_backend.DiartDiarization = _SharedDiartModels

    original_factory = audio_processor.online_diarization_factory

    def factory(args, diarization_backend):
        if isinstance(diarization_backend, _SharedDiartModels):
            return _SessionDiart(diarization_backend)
        return original_factory(args, diarization_backend)

    # audio_processor забрал функцию к себе при импорте — подменять надо там.
    audio_processor.online_diarization_factory = factory
    diart_backend._scribe_session_diart = True
    log.warning(
        "diart в WhisperLiveKit общий на весь процесс — даю каждой записи свой конвейер"
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
