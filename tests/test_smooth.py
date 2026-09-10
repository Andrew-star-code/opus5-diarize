"""Проверка нового сглаживания коротких вставок."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker-batch"))
import merge
from merge import Word

fails = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + ("" if cond else f"  <{detail}>"))
    if not cond:
        fails.append(name)


def w(text, s, e):
    return Word(text, s, e)


print("короткие вставки")

# Реальный случай: фраза разорвана на три части чужими вставками
words = [w("Вы", 40.8, 41.0), w("хотите", 41.0, 41.5),
         w("сберечь", 41.8, 42.3), w("планету", 42.3, 42.9),
         w("но", 43.2, 43.4), w("противитесь", 43.4, 44.2)]
labels = ["B", "B", "A", "A", "B", "B"]
merge._smooth(words, labels)
check("вставка 1.1с между B схлопнулась", labels == ["B"] * 6, labels)

# Длинная реплика не трогается
words2 = [w("раз", 0.0, 0.5), w("два", 0.5, 1.0),
          w("три", 1.2, 2.5), w("четыре", 2.5, 3.8),
          w("пять", 4.0, 4.5), w("шесть", 4.5, 5.0)]
labels2 = ["A", "A", "B", "B", "A", "A"]
merge._smooth(words2, labels2)
check("реплика 2.6с сохранилась", labels2 == ["A", "A", "B", "B", "A", "A"], labels2)

# Соседи разные: вставка уходит к тому, кто говорил дольше
words3 = [w("а", 0.0, 0.2),
          w("б", 0.3, 0.45),
          w("в", 0.5, 3.0), w("г", 3.0, 5.0)]
labels3 = ["A", "B", "C", "C"]
merge._smooth(words3, labels3)
check("вставка ушла к более длинному соседу", labels3[1] == "C", labels3)

# Границы не ломаются
check("пустой вход", merge._smooth([], []) is None)
one = ["A"]
merge._smooth([w("а", 0, 1)], one)
check("одно слово", one == ["A"], one)
two = ["A", "B"]
merge._smooth([w("а", 0, 0.1), w("б", 0.2, 0.3)], two)
check("два слова не зациклились", two == ["A", "B"], two)

print("\nсквозная проверка через assign_speakers")
turns = [merge.Turn(0.0, 2.0, "A"), merge.Turn(2.05, 2.3, "B"), merge.Turn(2.35, 5.0, "A")]
words4 = [w("один", 0.2, 0.8), w("два", 1.0, 1.8),
          w("три", 2.1, 2.25),
          w("четыре", 2.5, 3.2), w("пять", 3.4, 4.2)]
labels4 = merge.assign_speakers(words4, turns, smooth=True)
check("короткий турн B поглощён", set(labels4) == {"A"}, labels4)

segments = merge.build_segments(words4, labels4)
check("получилась одна реплика", len(segments) == 1, [s.speaker for s in segments])

print()
if fails:
    print(f"ПРОВАЛЕНО: {fails}")
    sys.exit(1)
print("Сглаживание работает")
