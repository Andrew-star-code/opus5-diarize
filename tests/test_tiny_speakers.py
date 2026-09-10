"""Проверка слияния «крошечных» говорящих (merge.tiny_speaker_renames)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker-batch"))

import merge  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + ("" if cond else f"  <{detail}>"))
    if not cond:
        fails.append(name)


def unit(*values):
    """Эмбеддинг из нескольких чисел — длина значения не имеет."""
    return list(values)


STRICT = dict(max_share=0.02, min_similarity=0.40)

print("слияние крошечных говорящих")

# Случай с ES2004a: пятый говорящий — 6,5 с из ~765, голосом близок к B.
labels = ["A", "B", "TINY", "C", "D"]
durations = [123.8, 76.1, 6.5, 346.9, 211.7]
emb = [
    unit(1, 0, 0, 0),
    unit(0, 1, 0, 0),
    unit(0, 0.6, 0.8, 0),   # сходство с B = 0.6
    unit(0, 0, 0, 1),
    unit(0.2, 0, 0, 0.98),
]
renames = merge.tiny_speaker_renames(labels, durations, emb, **STRICT)
check("обрывок чужого голоса слит с похожим", renames == {"TINY": "B"}, renames)

# Мало говорил, но голос свой — не трогаем.
emb_distinct = [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]
renames = merge.tiny_speaker_renames(
    ["A", "B", "QUIET"], [300.0, 300.0, 5.0], emb_distinct, **STRICT
)
check("тихий участник со своим голосом сохранён", renames == {}, renames)

# Похож голосом, но говорил заметно — это настоящий участник.
renames = merge.tiny_speaker_renames(
    ["A", "B"], [100.0, 10.0], [unit(1, 0.1), unit(1, 0.12)], **STRICT
)
check("доля 9% не считается обрывком", renames == {}, renames)

# Цепочка: X похож на Y, Y — на Z; оба крошечные. Итог должен вести в Z.
renames = merge.tiny_speaker_renames(
    ["X", "Y", "Z", "W"],
    [1.0, 1.5, 500.0, 500.0],
    [unit(1, 0, 0), unit(0.9, 0.44, 0), unit(0.5, 0.87, 0), unit(0, 0, 1)],
    max_share=0.02, min_similarity=0.40,
)
check("цепочка свёрнута, метки Y не осталось",
      set(renames.values()) <= {"Z", "W"} and "Y" in renames, renames)

print("\nграницы и порча данных")
check("один говорящий", merge.tiny_speaker_renames(["A"], [10.0], [unit(1)], **STRICT) == {})
check("пусто", merge.tiny_speaker_renames([], [], [], **STRICT) == {})
check("разная длина списков — ничего не делаем",
      merge.tiny_speaker_renames(["A", "B"], [1.0], [unit(1), unit(1)], **STRICT) == {})
check("нулевой эмбеддинг не роняет",
      merge.tiny_speaker_renames(["A", "B"], [500.0, 1.0], [unit(1, 0), unit(0, 0)], **STRICT) == {})
nan = float("nan")
check("NaN в эмбеддинге не сливается",
      merge.tiny_speaker_renames(["A", "B"], [500.0, 1.0], [unit(1, 0), unit(nan, nan)], **STRICT) == {})
check("max_share=0 выключает правило",
      merge.tiny_speaker_renames(labels, durations, emb, max_share=0, min_similarity=0.4) == {})

print("\nсглаживание выключено по умолчанию")
words = [merge.Word("a", 0.0, 1.0), merge.Word("b", 1.1, 1.3), merge.Word("c", 1.4, 3.0)]
turns = [merge.Turn(0.0, 1.05, "A"), merge.Turn(1.05, 1.35, "B"), merge.Turn(1.35, 3.0, "A")]
check("короткая вставка сохранена без smooth",
      merge.assign_speakers(words, turns) == ["A", "B", "A"], merge.assign_speakers(words, turns))
check("и поглощена со smooth=True",
      merge.assign_speakers(words, turns, smooth=True) == ["A", "A", "A"])

print()
if fails:
    print(f"ПРОВАЛЕНО: {fails}")
    sys.exit(1)
print("Слияние говорящих работает")
