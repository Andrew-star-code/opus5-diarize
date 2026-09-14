"""Поправка стыков реплик языковой моделью — без самой модели.

Модель подменяется функцией, которая отвечает заранее заданным номером.
Проверяется, что сдвиг применяется в обе стороны, соседняя граница не
пересекается, варианты в схеме ограничены ±max_shift, уверенные стыки
(конец предложения + пауза) модель не спрашивает, слова без говорящего не
трогаются, а сбой модели не ломает метки. Зависимостей нет:

    py -3 tests/test_llm_fix.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "worker-batch"))

from merge import Word  # noqa: E402
from polish import adjust_boundaries  # noqa: E402

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ок     " if ok else "  ПРОВАЛ ") + message)
    if not ok:
        failures.append(message)


def words_of(text: str, gap_after: dict[int, float] | None = None) -> list[Word]:
    """Слова по 0,3 с; gap_after — пауза после слова с таким индексом."""
    out, t = [], 0.0
    for i, w in enumerate(text.split()):
        out.append(Word(w=w, s=t, e=t + 0.3))
        t += 0.3 + (gap_after or {}).get(i, 0.0)
    return out


def answering(position_of_start):
    """Поддельная модель: отвечает номером, вычисленным по показанным ей словам."""
    seen = []

    def ask(system, user, schema):
        seen.append((user, schema))
        return {"start": position_of_start(user, schema)}
    return ask, seen


# «…как ты думаешь? Да» | «я думаю, что…» — «Да» ушло к А, должно быть у Б.
words = words_of("мы долго спорили как ты думаешь? Да я думаю что стоит попробовать")
labels = ["A"] * 7 + ["B"] * 5
ask, seen = answering(lambda user, schema: schema["properties"]["start"]["enum"][0] + 2)
# варианты: от b-3 до b+3; первый вариант — b-3, +2 → b-1, то есть «Да»
fixed, stats = adjust_boundaries(words, labels, ask)
check(fixed == ["A"] * 6 + ["B"] * 6, "граница сдвигается влево: «Да» переходит к Б")
check(stats == {"boundaries": 1, "asked": 1, "moved": 1, "words": 1}, f"счётчики: {stats}")
enum = seen[0][1]["properties"]["start"]["enum"]
check(len(enum) == 7, f"вариантов ровно ±3 слова: {enum}")
check("Сейчас реплика Б начинается со слова [" in seen[0][0], "модели показано, где граница сейчас")

# Сдвиг вправо: первые два слова «Б» на самом деле конец фразы А.
ask, _ = answering(lambda user, schema: schema["properties"]["start"]["enum"][-1] - 1)
fixed, stats = adjust_boundaries(words, labels, ask)
check(fixed == ["A"] * 9 + ["B"] * 3, "граница сдвигается вправо")

# Соседняя граница не пересекается: реплика Б из двух слов между А и А.
words2 = words_of("один два три четыре пять шесть семь восемь")
labels2 = ["A", "A", "A", "B", "B", "A", "A", "A"]
ask, seen = answering(lambda user, schema: max(schema["properties"]["start"]["enum"]))
fixed, _ = adjust_boundaries(words2, labels2, ask)
first_enum = seen[0][1]["properties"]["start"]["enum"]
check(max(first_enum) <= 5, f"первая граница не заходит за вторую: {first_enum}")
check(fixed[0] == "A" and fixed[-1] == "A" and "B" in fixed, f"реплика Б не исчезла и не расползлась: {fixed}")

# Уверенный стык — конец предложения и пауза — модель не спрашиваем.
words3 = words_of("это было давно. Да я помню", gap_after={2: 0.5})
labels3 = ["A"] * 3 + ["B"] * 3
ask, seen = answering(lambda user, schema: 1)
fixed, stats = adjust_boundaries(words3, labels3, ask)
check(not seen and fixed == labels3, "уверенный стык модель не спрашивает")
fixed, stats = adjust_boundaries(words3, labels3, ask, skip_confident=False)
check(len(seen) == 1, "с skip_confident=False спрашивает и уверенные")

# Слова без говорящего — не граница.
labels4 = ["A", "A", None, None, "B", "B"]
ask, seen = answering(lambda user, schema: 1)
fixed, stats = adjust_boundaries(words_of("а б в г д е"), labels4, ask)
check(not seen and fixed == labels4, "переход к словам без говорящего не трогается")


# Сбой модели: метки остаются как были.
def broken(system, user, schema):
    raise RuntimeError("сервис не отвечает")


fixed, stats = adjust_boundaries(words, labels, broken)
check(fixed == labels and stats["asked"] == 0, "сбой модели не меняет метки")

# ─── название, краткое содержание, имена ─────────────────────────
from merge import Segment  # noqa: E402
import polish  # noqa: E402

dialog = [
    Segment(0, 3, "SPEAKER_00", "Андрей, ты посмотрел отчёт за квартал?"),
    Segment(3, 6, "SPEAKER_01", "Да, посмотрел. Продажи выросли на десять процентов."),
    Segment(6, 9, "SPEAKER_00", "Отлично, тогда решаем расширить склад."),
]


def notes_model(names):
    """Поддельная модель для describe: отвечает по тому, какой схемой спросили."""
    calls = []

    def ask(system, user, schema):
        calls.append(user)
        props = schema["properties"]
        if "points" in props:
            return {"points": ["Обсудили отчёт за квартал", "Решили расширить склад"]}
        if "title" in props:
            return {"title": "«Отчёт за квартал и склад.»"}
        return {"names": names}
    return ask, calls


ask, calls = notes_model([{"speaker": "SPEAKER_01", "name": "Андрей", "quote": "Андрей, ты посмотрел отчёт"}])
notes = polish.describe(dialog, ask)
check(notes.get("summary") == "Обсудили отчёт за квартал\nРешили расширить склад", f"краткое содержание: {notes}")
check(notes.get("title") == "Отчёт за квартал и склад", f"название без кавычек и точки: {notes.get('title')!r}")
check(notes.get("speaker_names") == {"SPEAKER_01": "Андрей"}, f"имя с цитатой подсказано: {notes}")

ask, _ = notes_model([{"speaker": "SPEAKER_01", "name": "Андрей", "quote": "Андрей, ты видел бюджет"}])
check("speaker_names" not in polish.describe(dialog, ask), "цитаты нет в тексте — имя не подсказано")

ask, _ = notes_model([{"speaker": "SPEAKER_00", "name": "Андрей", "quote": "Андрей, ты посмотрел отчёт"},
                      {"speaker": "SPEAKER_01", "name": "Андрей", "quote": "Андрей, ты посмотрел отчёт"}])
check("speaker_names" not in polish.describe(dialog, ask), "одно имя двум людям — не подсказано никому")

long_dialog = [Segment(i, i + 1, "SPEAKER_00", "слово " * 300) for i in range(10)]
ask, calls = notes_model([])
polish.describe(long_dialog, ask)
summary_calls = [c for c in calls if "Перескажи" in c or "Сведи" in c]
check(len(summary_calls) == 3 and any("Сведи" in c for c in summary_calls),
      f"длинная запись: две части и свод — {len(summary_calls)} запроса")


def broken_notes(system, user, schema):
    raise RuntimeError("сервис не отвечает")


check(polish.describe(dialog, broken_notes) == {}, "сбой модели — пустые заметки, без исключения")

print("ИТОГ:", "всё верно" if not failures else f"провалов: {len(failures)}")
sys.exit(1 if failures else 0)
