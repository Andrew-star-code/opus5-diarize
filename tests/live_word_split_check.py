"""Проверка живого черновика: слова не рвутся между говорящими.

Гоняет настоящий обход WhisperLiveKit (build_token_speaker_segments) на
токенах, как их выдаёт SimulStreaming: слово на стыке порций приходит
двумя кусками, и граница diart ложится между ними. Сначала без нашей
подмены — баг должен воспроизвестись, иначе проверка ничего не ловит.
Потом с подменой — слова целые, а говорящих по-прежнему двое.

Заодно проверяет фильтр галлюцинаций в normalize().

Запуск внутри контейнера worker-live:

    docker compose exec -T worker-live python - < tests/live_word_split_check.py
"""
import sys

sys.path.insert(0, "/srv")  # здесь лежит код worker-live

from whisperlivekit.timed_objects import ASRToken, SpeakerSegment  # noqa: E402
from whisperlivekit.tokens_alignment import TokensAlignment  # noqa: E402

import engine  # noqa: E402

# Сцена из фильма, на которой баг был замечен: «Таноса нед» / «ели три»,
# «Ск» / «ани» / «руем». Граница говорящих намеренно режет оба слова.
TOKENS = [
    (" Мы", 0.0, 0.3), (" ловим", 0.3, 0.7), (" Таноса", 0.7, 1.2),
    (" нед", 1.2, 1.45), ("ели", 1.45, 1.7), (" три", 1.7, 2.0),
    (",", 2.0, 2.0), (" и", 2.0, 2.1), (" что", 2.1, 2.3), ("?", 2.3, 2.3),
    (" Ск", 2.3, 2.5), ("ани", 2.5, 2.7), ("руем", 2.7, 3.0),
    (" космос", 3.0, 3.6),
]
DIARIZATION = [
    SpeakerSegment(start=0.0, end=1.4, speaker=0),
    SpeakerSegment(start=1.4, end=2.6, speaker=1),
    SpeakerSegment(start=2.6, end=4.0, speaker=0),
]
WORDS = ["Мы", "ловим", "Таноса", "недели", "три,", "и", "что?",
         "Сканируем", "космос"]


def lines() -> list[tuple[int, str]]:
    alignment = TokensAlignment.__new__(TokensAlignment)
    alignment.all_tokens = [ASRToken(start=s, end=e, text=t) for t, s, e in TOKENS]
    segments, _buffer = alignment.build_token_speaker_segments(DIARIZATION)
    return [(seg.speaker, seg.text.strip()) for seg in segments]


def words_of(result: list[tuple[int, str]]) -> list[str]:
    return " ".join(text for _, text in result).split()


def show(title: str, result: list[tuple[int, str]]) -> None:
    print(title)
    for speaker, text in result:
        print(f"  Спикер {speaker}: {text}")


def main() -> int:
    failures = []

    before = lines()
    show("без подмены:", before)
    if words_of(before) == WORDS:
        failures.append("баг не воспроизвёлся без подмены — проверка ничего не ловит")

    engine._patch_word_continuations()
    engine._patch_word_continuations()  # повторный вызов безопасен
    after = lines()
    show("с подменой:", after)
    if words_of(after) != WORDS:
        failures.append(f"слова рвутся: {words_of(after)}")
    if len({speaker for speaker, _ in after}) < 2:
        failures.append("говорящий остался один — подмена съела диаризацию")

    response = {
        "lines": [
            {"start": 0, "end": 2, "speaker": 1,
             "text": "Субтитры создавал DimaTorzok Девушки отдыхают..."},
            {"start": 2, "end": 4, "speaker": 2,
             "text": "Редактор субтитров А.Синецкая Корректор А.Егорова"},
            {"start": 4, "end": 6, "speaker": 2,
             "text": "Подписывайтесь на канал."},
        ],
        "buffer_transcription": "Субтитры сделал DimaTorzok",
        "status": "active_transcription",
    }
    normalized = engine.normalize(response)
    texts = [line["text"] for line in normalized["lines"]]
    print("фильтр галлюцинаций:", texts, "| буфер:", repr(normalized["buffer"]))
    if texts != ["Девушки отдыхают...", "Подписывайтесь на канал."]:
        failures.append(f"фильтр галлюцинаций: {texts}")
    if normalized["buffer"]:
        failures.append(f"галлюцинация осталась в буфере: {normalized['buffer']!r}")

    for failure in failures:
        print("ПРОВАЛ:", failure)
    print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
