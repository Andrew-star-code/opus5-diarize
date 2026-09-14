"""Проверка потока живой диаризации: источник звука и сбор сегментов.

Источник. Звук, пришедший разом, должен уходить в diart сразу, а не с
реальной скоростью: исходный источник WhisperLiveKit досыпал перед каждым
блоком, и разметка отставала всё сильнее — к полутора минутам записи на
5–6 с. Посреди записи неполный блок не добивается нулями — иначе время
diart уходит вперёд времени слов. В конце записи хвост дожимается.

Сегменты. Идут строго по времени, пауза между кусками речи одного
говорящего не засчитывается ему, соседние куски склеиваются, метка —
число, время сдвигается на вырезанную тишину.

Сначала гоняется исходный источник WhisperLiveKit — отставание должно
воспроизвестись, иначе проверка ничего не ловит. Звук и видеокарта не
нужны, только контейнер worker-live:

    docker compose exec -T worker-live python - < tests/live_diart_stream_check.py
"""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/srv")  # здесь лежит код worker-live

import engine  # noqa: E402

try:
    from pyannote.core import Annotation, Segment  # noqa: E402
    from whisperlivekit.diarization.diart_backend import WebSocketAudioSource  # noqa: E402
except (ImportError, SystemExit):
    # Живая диаризация теперь на Sortformer, и diart в образе нет. Проверка
    # относится только к образу, собранному с diart.
    print("ПРОПУСК: в образе нет diart — проверка относится только к нему")
    raise SystemExit(0)

SR = 16000
BLOCK = 0.1
AUDIO_SECONDS = 3.0

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ок     " if ok else "  ПРОВАЛ ") + message)
    if not ok:
        failures.append(message)


def run_source(source_cls) -> tuple[float, np.ndarray, np.ndarray]:
    """Подаёт 3 с звука порциями произвольной длины разом.

    Возвращает: за сколько секунд ушли все полные блоки, что ушло до
    закрытия и что ушло всего.
    """
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(int(AUDIO_SECONDS * SR)).astype(np.float32)
    source = source_cls(uri="check", sample_rate=SR, block_duration=BLOCK)
    blocks: list[np.ndarray] = []
    source.stream.subscribe(on_next=lambda block: blocks.append(block[0].copy()))
    reader = threading.Thread(target=source.read, daemon=True)
    reader.start()

    started = time.monotonic()
    pos = 0
    while pos < len(audio):
        size = int(rng.uniform(0.15, 0.35) * SR)
        source.push_audio(audio[pos:pos + size])
        pos += size
    full = (len(audio) // source.block_size) * source.block_size
    deadline = started + AUDIO_SECONDS * 2 + 2
    while sum(len(b) for b in blocks) < full and time.monotonic() < deadline:
        time.sleep(0.01)
    elapsed = time.monotonic() - started
    before_close = np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)

    source.close()
    reader.join(timeout=3)
    everything = np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)
    return elapsed, audio, before_close, everything


def check_source() -> None:
    print("источник WhisperLiveKit как есть:")
    elapsed, _, _, _ = run_source(WebSocketAudioSource)
    print(f"  3 с звука ушли в diart за {elapsed:.2f} с")
    if elapsed < AUDIO_SECONDS * 0.8:
        failures.append("исходный источник не тормозит — проверка ничего не ловит")
        print("  ПРОВАЛ отставание не воспроизвелось — проверка ничего не ловит")

    print("наш источник:")
    source_cls = engine._realtime_source_class()
    elapsed, audio, before_close, everything = run_source(source_cls)
    check(elapsed < 0.5, f"3 с звука ушли в diart за {elapsed:.2f} с (надо быстрее 0,5 с)")
    check(np.array_equal(before_close, audio[:len(before_close)]),
          "до закрытия ушёл ровно поданный звук, без нулей посреди записи")
    block = int(BLOCK * SR)
    expected = -(-len(audio) // block) * block
    check(len(everything) == expected and np.array_equal(everything[:len(audio)], audio),
          "после закрытия хвост дожат нулями до целого блока")


def check_observer() -> None:
    print("сбор сегментов:")
    observer = engine._SegmentsObserver()

    class Audio:
        class extent:
            start, end = 0.0, 5.0

    first = Annotation()
    # Нарочно не по времени и с паузой внутри реплики говорящего 1.
    first[Segment(2.0, 2.5), 0] = "speaker1"
    first[Segment(0.0, 1.0), 1] = "speaker0"
    first[Segment(1.0, 1.4), 2] = "speaker1"
    observer.on_next((first, Audio))

    second = Annotation()
    second[Segment(2.52, 3.0), 0] = "speaker1"  # продолжение той же реплики
    observer.global_time_offset = 10.0  # вырезано 10 с тишины
    second[Segment(3.0, 3.5), 1] = "speaker0"
    observer.on_next((second, Audio))

    got = [(round(s.start, 2), round(s.end, 2), s.speaker) for s in observer.get_segments()]
    print("  сегменты:", got)
    starts = [s for s, _, _ in got]
    check(starts == sorted(starts), "сегменты идут по времени")
    check(all(isinstance(spk, int) for _, _, spk in got), "метки говорящих — числа")
    check(got[:3] == [(0.0, 1.0, 0), (1.0, 1.4, 1), (2.0, 2.5, 1)],
          "пауза 1,4–2,0 с не засчитана говорящему 1")
    check((12.52, 13.0, 1) in got and (13.0, 13.5, 0) in got,
          "после вырезанной тишины время сдвинуто на 10 с")

    copy = observer.get_segments()
    copy[0].end = 99.0
    check(observer.get_segments()[0].end == 1.0, "наружу отдаются копии")


def main() -> int:
    check_source()
    check_observer()
    print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
