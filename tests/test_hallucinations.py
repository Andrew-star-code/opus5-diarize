"""Фильтр галлюцинаций Whisper: строчки из титров вырезаются, речь — нет.

Галлюцинации — настоящие ответы Whisper large-v3 на Golos farfield
(bench/eval_asr.py): на месте короткой команды он написал подпись из
титров. Настоящие фразы рядом — те, что похожи на подозрительные, но
сказаны на самом деле. Отдельно проверяется, что копии модуля в двух
воркерах совпадают. Зависимостей нет:

    py -3 tests/test_hallucinations.py
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE = ROOT / "worker-live" / "hallucinations.py"
BATCH = ROOT / "worker-batch" / "hallucinations.py"

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ок     " if ok else "  ПРОВАЛ ") + message)
    if not ok:
        failures.append(message)


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("hallucinations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check(LIVE.read_bytes() == BATCH.read_bytes(), "копии в worker-live и worker-batch совпадают")
h = load(BATCH)

HALLUCINATIONS = [
    "Субтитры сделал DimaTorzok",
    "Субтитры создавал DimaTorzok",
    "Субтитры подогнал «Симон»",
    "Добавил субтитры DimaTorzok",
    "Редактор субтитров А.Семкин Корректор А.Егорова",
    "Субтитры создавались сообществом Amara.org",
    "Продолжение следует...",
]
for text in HALLUCINATIONS:
    check(h.drop_hallucinations(text).strip() == "", f"вырезано: «{text}»")

SPEECH = [
    "Джой, набери сламчика с Симоном.",
    "Включи нежного редактора.",
    "Подписывайтесь на наш канал, друзья.",
    "Спасибо за просмотр!",
    "Продолжение следует в следующем выпуске.",
    "Я смотрел эти субтитры вчера.",
    "Редактор журнала сказал, что всё готово.",
]
for text in SPEECH:
    check(h.drop_hallucinations(text) == text, f"не тронуто: «{text}»")

check(h.drop_hallucinations("Девушки отдыхают... Субтитры создавал DimaTorzok").strip() == "Девушки отдыхают...",
      "подпись внутри текста вырезается, речь рядом остаётся")

words = ["Девушки", "отдыхают.", "Субтитры", "сделал", "DimaTorzok"]
check(h.hallucinated_words(words) == [False, False, True, True, True],
      "в чистовике помечаются ровно слова подписи")
check(h.hallucinated_words(["Продолжение", "следует..."]) == [True, True],
      "«Продолжение следует» целой фразой — помечена вся")
check(h.hallucinated_words(["Продолжение", "следует", "завтра."]) == [False, False, False],
      "«Продолжение следует» внутри фразы не трогается")

print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
sys.exit(1 if failures else 0)
