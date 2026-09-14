"""Проверка живой диаризации на Streaming Sortformer.

Гоняет запись с двумя собеседниками через тот же класс, что стоит в живом
режиме (engine._SessionSortformer), и так же, как это делает
WhisperLiveKit: звук кусками по 0,1 с, после каждого куска diarize()
зовётся, пока не вернёт пусто.

Проверяет, что говорящих нашлось не меньше двух, сегменты идут по
времени, разметка не отстаёт от поданного звука больше чем на порцию с
заглядыванием вперёд и сдвиг на вырезанную тишину доходит до сегментов.
Нужны видеокарта и контейнер worker-live:

    docker compose exec -T worker-live python - < tests/live_sortformer_check.py
"""
import asyncio
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/srv")  # здесь лежит код worker-live

import engine  # noqa: E402

SR = 16000
PIECE = 0.1
AUDIO = "/data/stream_dialog.webm"

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ок     " if ok else "  ПРОВАЛ ") + message)
    if not ok:
        failures.append(message)


async def main() -> int:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", AUDIO, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()

    shared = engine._SharedSortformer()
    session = engine._SessionSortformer(shared)
    stream = session.stream
    budget = (stream.chunk + stream.right) * stream.hop / SR + 0.1  # порция + заглядывание + кусок

    segments, emitted, worst_lag = [], [], 0.0
    piece = int(PIECE * SR)
    for i in range(0, len(audio), piece):
        session.insert_audio_chunk(audio[i:i + piece])
        while True:
            new = await session.diarize()
            if not new:
                break
            segments += new
            emitted += [s.speaker for s in new]
        pushed = min(len(audio), i + piece) / SR
        worst_lag = max(worst_lag, pushed - stream.frames_out * stream.frame_sec)

    speakers = sorted({s.speaker for s in segments})
    print(f"сегментов {len(segments)}, говорящие {speakers}, "
          f"наибольшее отставание разметки {worst_lag:.2f} с (допуск {budget:.2f} с)")
    check(len(speakers) >= 2, "нашлось не меньше двух говорящих")

    # Второй проход перемечает выданные отрезки на месте — но только те,
    # что уже покрыл медленный поток, и только номерами говорящих модели.
    if session.slow is not None:
        fixed = {id(p) for record in session._fast[: session._fixed] for p in record[3]}
        changed = [s for s, before in zip(segments, emitted) if s.speaker != before]
        print(f"второй проход: сверено отрезков {session._fixed} из {len(session._fast)}, "
              f"перемечено {session.relabeled}, кусков с новой меткой {len(changed)}")
        check(all(id(s) in fixed for s in changed), "метки меняются только там, где прошёл медленный проход")
        check(all(0 <= s.speaker < 8 for s in segments), "номера говорящих — в пределах каналов модели")
    check(all(s.end > s.start for s in segments), "у каждого сегмента конец позже начала")
    starts = [s.start for s in segments]
    check(max(np.diff(starts), default=0) < 30 and min(np.diff(starts), default=0) > -1.0,
          "сегменты идут по времени, без скачков назад")
    check(worst_lag <= budget, "разметка не отстаёт больше чем на порцию с заглядыванием")

    # Пауза посреди потока. Хвост звука перед ней ещё не разобран (меньше
    # порции с заглядыванием) — его сегменты должны остаться до паузы, а
    # всё после — сдвинуться на её длину. Раньше сдвигалось всё, что выдано
    # после сообщения о паузе, и конец реплики переезжал на начало следующей.
    pause_at = len(audio) / SR
    session.insert_silence(5.0)
    session.insert_audio_chunk(audio[:SR * 2])
    after = []
    while True:
        new = await session.diarize()
        if not new:
            break
        after += new
    check(all(s.end <= pause_at + 0.01 or s.start >= pause_at + 5.0 - 0.01 for s in after),
          "ни один сегмент не попал внутрь вырезанной паузы")
    check(any(s.start >= pause_at + 5.0 - 0.01 for s in after),
          "звук после паузы сдвинут на её длину")
    session.close()

    print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
