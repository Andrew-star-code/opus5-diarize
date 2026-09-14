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
from dataclasses import replace
from typing import Any

from config import settings
from hallucinations import drop_hallucinations

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
        if settings.diarization_enabled and settings.live_diarization == "sortformer":
            # Своя модель порций и свой поток на запись — см. _SessionSortformer.
            _install_session_sortformer()
        if settings.diarization_enabled:
            _patch_word_continuations()
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
            # Подобраны замером, см. комментарии в config.py.
            latency=settings.live_diar_latency,
            tau_active=settings.live_diar_tau_active,
            rho_update=settings.live_diar_rho_update,
            delta_new=settings.live_diar_delta_new,
            device=self.device,
            sample_rate=self.sample_rate,
        )


class _SegmentsObserver:
    """Собирает разметку diart в сегменты для выравнивания WhisperLiveKit.

    Замена DiarizationObserver из WhisperLiveKit, у которого два изъяна:

    - границы реплик он берёт из плоского списка попарно, так что пауза
      между двумя кусками речи одного говорящего тоже засчитывается ему;
    - сегменты складываются по говорящим, а не по времени, а выравнивание
      идёт по ним курсором, который считает список упорядоченным, — и на
      смене говорящих может проскочить нужный сегмент.

    Здесь сегменты идут строго по времени, а соседние куски одного
    говорящего склеиваются. Заодно список остаётся коротким — по сегменту
    на реплику, а не на каждые полсекунды, — и его копирование на каждом
    обновлении не дорожает к концу долгой записи.
    """

    _GAP = 0.05  # щель между шагами diart, которую не считаем паузой

    def __init__(self) -> None:
        from whisperlivekit.timed_objects import SpeakerSegment

        self._segment_cls = SpeakerSegment
        self._segments: list = []
        self._lock = threading.Lock()
        self.global_time_offset = 0.0
        self.processed_time = 0.0

    def on_next(self, value) -> None:
        annotation, audio = value
        tracks = sorted(annotation.itertracks(yield_label=True), key=lambda item: item[0].start)
        with self._lock:
            self.processed_time = max(self.processed_time, audio.extent.end)
            for segment, _track, label in tracks:
                speaker = _speaker_index(label)
                start = segment.start + self.global_time_offset
                end = segment.end + self.global_time_offset
                last = self._segments[-1] if self._segments else None
                if last is not None and last.speaker == speaker and start <= last.end + self._GAP:
                    last.end = max(last.end, end)
                else:
                    self._segments.append(self._segment_cls(start=start, end=end, speaker=speaker))

    def on_error(self, error) -> None:
        log.error("Ошибка в потоке diart: %s", error)

    def on_completed(self) -> None:
        pass

    def get_segments(self) -> list:
        # Копии: последний сегмент ещё растёт в потоке diart.
        with self._lock:
            return [replace(segment) for segment in self._segments]


def _speaker_index(label: Any) -> int:
    """«speaker3» от diart -> 3; выравнивание WhisperLiveKit ждёт число."""
    if isinstance(label, int):
        return label
    digits = re.sub(r"\D", "", str(label))
    return int(digits) if digits else 0


_REALTIME_SOURCE = None


def _realtime_source_class():
    """Источник звука для diart без искусственного торможения.

    WebSocketAudioSource из WhisperLiveKit перед каждым блоком досыпает до
    длительности блока, отсчитывая время от конца обработки предыдущего.
    Выходит, что на каждый шаг diart уходит время звука плюс время работы
    моделей, и разметка отстаёт всё сильнее: по замеру на 5,5 с к полутора
    минутам записи, и отставание не рассасывается. Пока разметка не
    покрыла слово, WhisperLiveKit держит его серой гипотезой — текст
    «застывал» тем сильнее, чем дольше шла запись. Живой звук и так
    приходит с реальной скоростью, тормозить его незачем.

    Второе отличие: неполный блок не добивается нулями посреди записи.
    Порции от WhisperLiveKit произвольной длины, и на каждой паузе между
    ними исходный источник вставлял тишину, которой не было, — время diart
    уходило вперёд времени слов. Остаток дожидается следующей порции, а
    нулями дополняется только в самом конце записи.
    """
    global _REALTIME_SOURCE
    if _REALTIME_SOURCE is not None:
        return _REALTIME_SOURCE

    from queue import Empty

    import numpy as np
    from whisperlivekit.diarization.diart_backend import WebSocketAudioSource

    class RealtimeAudioSource(WebSocketAudioSource):
        def _process_chunks(self) -> None:
            try:
                while not self._closed:
                    try:
                        chunk = self._queue.get(timeout=0.1)
                    except Empty:
                        continue
                    self._emit(chunk)
                # Хвост: всё, что успело прийти до закрытия, и неполный блок.
                while True:
                    try:
                        self._emit(self._queue.get_nowait())
                    except Empty:
                        break
                with self._buffer_lock:
                    if len(self._buffer):
                        padded = np.zeros(self.block_size, dtype=np.float32)
                        padded[:len(self._buffer)] = self._buffer
                        self._buffer = np.array([], dtype=np.float32)
                        self.stream.on_next(padded.reshape(1, -1))
            except Exception as exc:
                log.exception("Источник звука diart остановился с ошибкой")
                self.stream.on_error(exc)
                return
            self.stream.on_completed()

        def _emit(self, chunk) -> None:
            with self._buffer_lock:
                self._buffer = np.concatenate([self._buffer, chunk])
                while len(self._buffer) >= self.block_size:
                    block = self._buffer[:self.block_size]
                    self._buffer = self._buffer[self.block_size:]
                    self.stream.on_next(block.reshape(1, -1))

    _REALTIME_SOURCE = RealtimeAudioSource
    return _REALTIME_SOURCE


class _SessionDiart:
    """Диаризация одной живой записи.

    Интерфейс ровно тот, что ждёт AudioProcessor от diart-бэкенда. Буфера
    buffer_audio у него нет намеренно: по этому признаку AudioProcessor
    понимает, что сегменты накопительные и их надо заменять, а не дописывать.
    """

    def __init__(self, shared: _SharedDiartModels) -> None:
        from diart import SpeakerDiarization
        from diart.inference import StreamingInference

        self.observer = _SegmentsObserver()
        self.source = _realtime_source_class()(
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


def _patch_word_continuations() -> None:
    """Не даёт говорящему смениться посреди слова в живом черновике.

    SimulStreaming режет речь на порции, и слово на стыке порций приходит
    двумя токенами: начало с ведущим пробелом (« Ск»), продолжение без
    него («анируем»). Выравнивание WhisperLiveKit назначает говорящего
    каждому токену отдельно, и если граница diart легла между кусками,
    слово разрывается между двумя репликами: «Ск» у одного говорящего,
    «анируем» у другого.

    В апстриме уже есть ровно такой случай: одиночный знак препинания
    всегда уходит к текущему говорящему, чтобы не плодить реплики из
    одной точки. Эта проверка вызывается только в том месте обхода, и
    мы расширяем её на продолжения слов. Новое слово Whisper всегда
    выдаёт с пробелом, поэтому целые слова не прилипают к чужой реплике.
    """
    try:
        from whisperlivekit.tokens_alignment import TokensAlignment
    except Exception:
        log.exception("Не удалось подключить склейку слов — в живом "
                      "черновике слова могут рваться между говорящими")
        return

    original = getattr(TokensAlignment, "_is_punctuation_only", None)
    if original is None:
        log.error("В WhisperLiveKit нет _is_punctuation_only — склейка слов "
                  "не подключена, проверьте версию")
        return
    if getattr(original, "_scribe_patched", False):
        return

    def stays_with_previous(token) -> bool:
        text = getattr(token, "text", "") or ""
        return original(token) or (bool(text) and not text[0].isspace())

    stays_with_previous._scribe_patched = True
    TokensAlignment._is_punctuation_only = staticmethod(stays_with_previous)
    log.info("Говорящий в живом черновике меняется только на начале слова")


class _SharedSortformer:
    """Streaming Sortformer, одна модель на процесс.

    Встаёт на место SortformerDiarization из WhisperLiveKit. Тот грузит
    модель по имени через сеть и ставит свои настройки порций: порция в
    секунду без заглядывания вперёд и subsampling 10 при кадре модели 8.
    Здесь — локальный .nemo из кеша (сервис живёт без сети) и настройки
    из статьи, те же, что на стенде (bench/eval_sortformer.py).
    """

    def __init__(self, model_name: str | None = None, model_path: str | None = None,
                 **_ignored: Any) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import SortformerEncLabelModel
        from sortformer_stream import configure

        repo = settings.live_sortformer_model
        # С HF_HUB_OFFLINE=1 hf_hub_download берёт файл из кеша и в сеть не
        # ходит; сам .nemo кладёт туда scripts/prefetch_models.py.
        path = model_path or hf_hub_download(repo, f"{repo.rsplit('/', 1)[-1]}.nemo")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SortformerEncLabelModel.restore_from(path, map_location=device)
        self.model.eval()
        configure(self.model, settings.live_sortformer_latency)
        log.info("Sortformer готов: %s, задержка %.2f с", repo, settings.live_sortformer_latency)


class _SessionSortformer:
    """Диаризация одной живой записи на общей модели Sortformer.

    Интерфейс — тот, что AudioProcessor ждёт от буферного бэкенда: по
    атрибуту buffer_audio он понимает, что diarize() надо звать, пока она
    не вернёт пусто, а сегменты — дописывать к прежним. Звук вставляется и
    разбирается по очереди в одной сопрограмме, так что блокировка не
    нужна; сам шаг модели уходит в поток, чтобы не держать цикл событий.
    """

    def __init__(self, shared_model: _SharedSortformer, max_speakers: int | None = None,
                 **_ignored: Any) -> None:
        from sortformer_stream import PRESETS, SortformerStream

        self.stream = SortformerStream(shared_model.model, latency=settings.live_sortformer_latency)
        # Второй проход: та же модель с запасом по времени идёт следом и
        # перемечает уже выданные отрезки. Кто говорит, она видит заметно
        # точнее: сквозной замер — 13,5% слов под чужим именем против 15,2%,
        # начало реплики уходит предыдущему в 32% смен против 43%.
        refine = settings.live_sortformer_refine_latency
        if refine and refine not in PRESETS:
            log.error("LIVE_SORTFORMER_REFINE_LATENCY=%s нет среди %s — второй проход выключен",
                      refine, sorted(PRESETS))
            refine = 0
        self.slow = SortformerStream(shared_model.model, latency=refine) if refine else None
        # Выданные отрезки быстрого прохода: время потока, метка и куски,
        # ушедшие в WhisperLiveKit. Он хранит их по ссылке и при каждом
        # обновлении заново раздаёт слова говорящим, поэтому поправка метки
        # на месте доходит до реплик сама.
        self._fast: list[tuple[float, float, int, list]] = []
        self._fixed = 0                    # сколько из них уже сверено
        self._slow: list[tuple[float, float, int]] = []
        self._slow_from = 0                # первый ещё нужный медленный отрезок
        self._overlap: dict[tuple[int, int], float] = {}  # (медленная, быстрая) → секунды
        self.relabeled = 0
        self.buffer_audio = None  # признак буферного бэкенда для AudioProcessor
        # Вырезанные паузы. Фильтр тишины не пускает их в разметку, а время
        # слов их включает, так что время потока надо переводить в настоящее.
        # Сдвиг привязан к месту в потоке, где была пауза, а не к моменту
        # выдачи сегмента: о паузе сообщают, когда хвост реплики перед ней
        # ещё не разобран, и со сдвигом «на момент выдачи» он переезжал за
        # паузу — на первые слова следующего говорящего.
        self._pause_at: list[float] = []     # время потока, где была пауза
        self._shift_after: list[float] = []  # суммарный сдвиг после неё

    def insert_silence(self, duration: float | None) -> None:
        from sortformer_stream import SAMPLE_RATE

        if not duration:
            return
        total = (self._shift_after[-1] if self._shift_after else 0.0) + duration
        self._pause_at.append(self.stream.samples / SAMPLE_RATE)
        self._shift_after.append(total)

    def _real_time(self, start: float, end: float) -> list[tuple[float, float]]:
        """Отрезок во времени потока → куски в настоящем времени, с разрывами на паузах."""
        import bisect

        i = bisect.bisect_right(self._pause_at, start)
        shift = self._shift_after[i - 1] if i else 0.0
        pieces = []
        while i < len(self._pause_at) and self._pause_at[i] < end:
            cut = self._pause_at[i]
            pieces.append((start + shift, cut + shift))
            start, shift = cut, self._shift_after[i]
            i += 1
        pieces.append((start + shift, end + shift))
        return [(a, b) for a, b in pieces if b > a]

    def insert_audio_chunk(self, pcm_array) -> None:
        self.stream.push(pcm_array)
        if self.slow is not None:
            self.slow.push(pcm_array)

    async def diarize(self):
        import asyncio

        from whisperlivekit.timed_objects import SpeakerSegment

        fast, slow = await asyncio.to_thread(self._step)
        out = []
        for start, end, speaker in fast:
            pieces = [SpeakerSegment(speaker=speaker, start=a, end=b)
                      for a, b in self._real_time(start, end)]
            self._fast.append((start, end, speaker, pieces))
            out += pieces
        if slow:
            # Перемечаем здесь, в цикле событий, а не в потоке: выравнивание
            # WhisperLiveKit читает эти объекты тоже из цикла событий.
            self._slow += slow
            self._reconcile()
        return out

    def _step(self):
        fast = self.stream.step()
        slow = self.slow.step() if self.slow is not None else []
        return fast, slow

    def _reconcile(self) -> None:
        """Отрезки быстрого прохода, которые уже покрыл медленный, получают его метку."""
        slow, slow_end = self._slow, self.slow.frames_out * self.slow.frame_sec
        batch = []
        while self._fixed < len(self._fast) and self._fast[self._fixed][1] <= slow_end:
            start, end, fast_label, pieces = self._fast[self._fixed]
            self._fixed += 1
            # Отрезки обоих проходов идут по времени начала — медленные,
            # закончившиеся до начала этого быстрого, больше не понадобятся.
            while self._slow_from < len(slow) and slow[self._slow_from][1] <= start:
                self._slow_from += 1
            votes: dict[int, float] = {}
            k = self._slow_from
            while k < len(slow) and slow[k][0] < end:
                overlap = min(end, slow[k][1]) - max(start, slow[k][0])
                if overlap > 0:
                    votes[slow[k][2]] = votes.get(slow[k][2], 0.0) + overlap
                k += 1
            if votes:
                best = max(votes, key=votes.get)
                key = (best, fast_label)
                self._overlap[key] = self._overlap.get(key, 0.0) + votes[best]
                batch.append((best, fast_label, pieces))
        if not batch:
            return
        mapping = self._mapping()
        for best, fast_label, pieces in batch:
            target = mapping.get(best, fast_label)
            if target != fast_label:
                for piece in pieces:
                    piece.speaker = target
                self.relabeled += 1
        if self._slow_from > 1000:
            del slow[: self._slow_from]
            self._slow_from = 0

    def _mapping(self) -> dict[int, int]:
        """Метка медленного прохода → метка быстрого, по накопленным перекрытиям.

        Проходы нумеруют говорящих каждый по-своему, в порядке появления.
        Показываем нумерацию быстрого: её пользователь уже видел. Говорящий,
        которого быстрый проход слил с другим, получает свой номер после
        четырёх каналов модели.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        slow_labels = sorted({s for s, _ in self._overlap})
        fast_labels = sorted({f for _, f in self._overlap})
        matrix = np.zeros((len(slow_labels), len(fast_labels)))
        for (s, f), seconds in self._overlap.items():
            matrix[slow_labels.index(s), fast_labels.index(f)] = seconds
        rows, cols = linear_sum_assignment(-matrix)
        mapping = {slow_labels[r]: fast_labels[c] for r, c in zip(rows, cols)}
        for s in slow_labels:
            mapping.setdefault(s, 4 + s)
        return mapping

    def close(self) -> None:
        # Хвост короче порции не дожимаем: черновик после остановки
        # всё равно заменяет чистовик.
        self.stream = None
        self.slow = None


def _install_session_sortformer() -> None:
    """Подменяет Sortformer из WhisperLiveKit нашим — см. _SharedSortformer.

    core.py и online_diarization_factory импортируют оба класса из модуля
    в момент вызова, так что подмены в модуле достаточно.
    """
    try:
        from whisperlivekit.diarization import sortformer_backend
    except (Exception, SystemExit):
        # Без NeMo модуль делает SystemExit — это тоже надо поймать.
        log.exception("Не удалось подключить Sortformer — живая диаризация не заработает")
        return
    if getattr(sortformer_backend, "_scribe_session_sortformer", False):
        return
    sortformer_backend.SortformerDiarization = _SharedSortformer
    sortformer_backend.SortformerDiarizationOnline = _SessionSortformer
    sortformer_backend._scribe_session_sortformer = True
    log.info("Sortformer: локальная модель, порции из статьи, свой поток на запись")


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


_SENTENCE_END = (".", "!", "?", "…")


def _ends_sentence(word: str) -> bool:
    return word.rstrip("»\"')").endswith(_SENTENCE_END)


def _snap_turns(lines: list[dict[str, Any]], max_shift: int) -> list[dict[str, Any]]:
    """Смена говорящего посреди предложения переносится к его концу.

    diart ставит границу по звуку и часто промахивается на слово: «…Шептать.
    Надо» у одного, «всегда находить…» у другого. Если в пределах max_shift
    слов от границы предложение кончается, граница переезжает туда —
    сначала ищем назад, потом вперёд, как в стенде (bench/eval_live.py).
    Время реплик не трогаем: пословного времени в репликах нет, а сдвиг
    на слово черновику не важен.

    Замер на семи записях вместе с правилом четырёх слов: 16,0% слов под
    чужим именем вместо 16,3%, и ни одна запись не стала хуже. При сдвиге
    на два слова в среднем чуть лучше, но две записи ухудшаются.
    """
    if max_shift < 1 or len(lines) < 2:
        return lines
    out = [dict(line) for line in lines]
    for prev, cur in zip(out, out[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        before, after = prev["text"].split(), cur["text"].split()
        if not before or not after or _ends_sentence(before[-1]):
            continue
        for d in range(1, max_shift + 1):
            if len(before) > d and _ends_sentence(before[-1 - d]):
                after[:0] = before[-d:]
                del before[-d:]
                break
            if len(after) > d and _ends_sentence(after[d - 1]):
                before.extend(after[:d])
                del after[:d]
                break
        else:
            continue
        prev["text"], cur["text"] = " ".join(before), " ".join(after)
    return out


def _absorb_short_turns(lines: list[dict[str, Any]], min_words: int) -> list[dict[str, Any]]:
    """Обрывок короче min_words слов не может сменить говорящего.

    diart решает «кто говорит» каждые полсекунды по звуку, и на стыке
    реплик метка дрожит: «Но их» у одного, «нельзя» у другого, дальше
    снова первый. Коммерческие системы (AssemblyAI) решают по реплике
    целиком, а обрывку короче секунды своего говорящего не дают. Здесь
    так же: подряд идущие реплики одного говорящего — одна серия, и если
    в серии меньше min_words слов, она остаётся у предыдущего говорящего
    и приклеивается к его реплике.

    Смотрим только назад, поэтому годится для живого режима: пока новый
    человек сказал меньше min_words слов, они идут под предыдущим, а
    когда реплика выросла — становятся отдельной. Замер (bench/eval_live.py,
    семь записей): обрывков короче трёх слов 2 вместо 140, слов под чужим
    именем 16,3% вместо 18,1%. Цена — настоящий ответ в пару слов («Да,
    конечно») попадает к собеседнику.
    """
    if min_words <= 1 or len(lines) < 2:
        return lines

    series: list[list[dict[str, Any]]] = []
    for line in lines:
        if series and series[-1][0]["speaker"] == line["speaker"]:
            series[-1].append(line)
        else:
            series.append([line])

    out: list[dict[str, Any]] = []
    glued = False  # к последней реплике только что приклеен чужой обрывок
    for run in series:
        words = sum(len(line["text"].split()) for line in run)
        if out and words < min_words:
            for line in run:
                _glue(out[-1], line)
            glued = True
            continue
        run = [dict(line) for line in run]
        if glued and out[-1]["speaker"] == run[0]["speaker"]:
            # «А, обрывок Б, снова А» — это одна реплика А.
            _glue(out[-1], run.pop(0))
        out.extend(run)
        glued = False
    return out


def _glue(target: dict[str, Any], line: dict[str, Any]) -> None:
    target["text"] = f"{target['text']} {line['text']}"
    target["end"] = max(target["end"], line["end"])


def normalize(response: Any) -> dict[str, Any]:
    """Приводит ответ движка к формату, который понимает наш фронтенд."""
    raw_lines = _get(response, "lines") or []
    lines = []
    for line in raw_lines:
        text = drop_hallucinations(_get(line, "text") or "").strip()
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
        "lines": _absorb_short_turns(
            _snap_turns(lines, settings.live_snap_words), settings.live_min_turn_words
        ),
        # Гипотеза, которую движок ещё может переписать. Показываем её
        # серым: честнее, чем выдавать неустоявшийся текст за готовый.
        "buffer": drop_hallucinations(_get(response, "buffer_transcription") or "").strip(),
        "buffer_speaker": (_get(response, "buffer_diarization") or "").strip(),
        "status": _get(response, "status") or "active",
        # Отставание в секундах звука: распознавания — от поступившего
        # звука, разметки говорящих — от распознанного текста. Пока
        # разметка не догнала слово, оно висит серой гипотезой.
        "lag": {
            "asr": _seconds(_get(response, "remaining_time_transcription")),
            "diarization": _seconds(_get(response, "remaining_time_diarization")),
        },
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
