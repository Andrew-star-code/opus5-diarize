"""Проверка живого черновика: слова не рвутся между говорящими.

Гоняет настоящий обход WhisperLiveKit (build_token_speaker_segments) на
токенах, как их выдаёт SimulStreaming: слово на стыке порций приходит
двумя кусками, и граница diart ложится между ними. Сначала без нашей
подмены — баг должен воспроизвестись, иначе проверка ничего не ловит.
Потом с подменой — слова целые, а говорящих по-прежнему двое.

Заодно проверяет фильтр галлюцинаций в normalize() и то, что обрывок
в пару слов не заводит нового говорящего.

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
             "text": "Подписывайтесь на наш канал, друзья."},
        ],
        "buffer_transcription": "Субтитры сделал DimaTorzok",
        "status": "active_transcription",
    }
    normalized = engine.normalize(response)
    texts = [line["text"] for line in normalized["lines"]]
    print("фильтр галлюцинаций:", texts, "| буфер:", repr(normalized["buffer"]))
    if texts != ["Девушки отдыхают...", "Подписывайтесь на наш канал, друзья."]:
        failures.append(f"фильтр галлюцинаций: {texts}")
    if normalized["buffer"]:
        failures.append(f"галлюцинация осталась в буфере: {normalized['buffer']!r}")

    # Обрывки на стыке реплик: так выглядел живой черновик на записи
    # с двумя собеседниками — метка дрожала на каждом слове.
    a, b = "SPEAKER_01", "SPEAKER_02"

    def line(speaker, text, start):
        return {"start": start, "end": start + 1, "speaker": speaker, "text": text}

    absorb = engine._absorb_short_turns
    got = absorb([
        line(a, "Но их", 0), line(b, "нельзя", 1),
        line(a, "заменить на слово, в котором есть «е».", 2),
        line(b, "Вот и думаю, сейчас вот первое слово", 11),
    ], 4)
    print("обрывки:", [(l["speaker"], l["text"]) for l in got])
    if [(l["speaker"], l["text"]) for l in got] != [
        (a, "Но их нельзя заменить на слово, в котором есть «е»."),
        (b, "Вот и думаю, сейчас вот первое слово"),
    ]:
        failures.append(f"обрывок не приклеился: {got}")
    kept = absorb([line(a, "Длинная реплика первого участника", 0),
                   line(b, "Нет, я так не думаю", 3), line(a, "Хорошо", 5)], 4)
    # «Хорошо» — обрывок, он уходит к B; сама реплика B остаётся своей.
    if [l["speaker"] for l in kept] != [a, b] or not kept[1]["text"].startswith("Нет, я так не думаю"):
        failures.append(f"реплика из 4+ слов потеряла говорящего: {kept}")
    paused = [line(a, "Первая часть мысли у меня", 0), line(a, "и вторая после паузы тоже", 4)]
    if absorb(paused, 4) != paused:
        failures.append("разрыв на паузе у одного говорящего склеился")
    if absorb(got, 1) != got:
        failures.append("min_words=1 должен выключать правило")

    # Граница на слово мимо конца предложения — так было на той же записи.
    snap = engine._snap_turns
    got = snap([line(a, "Шепот через «е» шептать. Шептать. Надо", 0),
                line(b, "всегда находить вот эти слова.", 3)], 1)
    print("подтяжка:", [l["text"] for l in got])
    if [l["text"] for l in got] != ["Шепот через «е» шептать. Шептать.",
                                     "Надо всегда находить вот эти слова."]:
        failures.append(f"граница не подтянулась назад: {got}")
    got = snap([line(a, "ну вот, да у нас", 0), line(b, "своих-то. Исконно русских слов", 3)], 1)
    if [l["text"] for l in got] != ["ну вот, да у нас своих-то.", "Исконно русских слов"]:
        failures.append(f"граница не подтянулась вперёд: {got}")
    done = [line(a, "Это конец мысли.", 0), line(b, "а это начало", 2)]
    if snap(done, 1) != done:
        failures.append("граница на конце предложения сдвинулась")
    if snap(got, 0) != got:
        failures.append("max_shift=0 должен выключать подтяжку")

    for failure in failures:
        print("ПРОВАЛ:", failure)
    print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
