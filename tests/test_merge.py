"""Проверка слияния ASR + диаризации на синтетических данных."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker-batch"))

import merge
from merge import Turn, Word

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


print("assign_speakers")

# Простой диалог: A говорит 0-2, B говорит 2.5-5
turns = [Turn(0.0, 2.0, "A"), Turn(2.5, 5.0, "B")]
words = [
    Word("привет", 0.1, 0.6),
    Word("как", 0.7, 1.0),
    Word("дела", 1.1, 1.8),
    Word("нормально", 2.6, 3.4),
    Word("спасибо", 3.5, 4.2),
]
labels = merge.assign_speakers(words, turns)
check("реплики разошлись по говорящим", labels == ["A", "A", "A", "B", "B"], labels)

# Слово, попавшее в паузу между турнами
words2 = [Word("угу", 2.15, 2.35)]
labels2 = merge.assign_speakers(words2, turns)
check("слово в паузе отдано ближайшему", labels2[0] in ("A", "B"), labels2)

# Слово перекрывает оба турна — должен победить больший кусок
words3 = [Word("длинное", 1.5, 3.0)]
labels3 = merge.assign_speakers(words3, turns)
check("побеждает максимальное перекрытие", labels3 == ["A"], labels3)

# Диаризации нет вообще
check("без турнов все None", merge.assign_speakers(words, []) == [None] * 5)

# Сглаживание одиночного выброса A A B A A
smoothed = ["A", "A", "B", "A", "A"]
merge._smooth([Word(f"w{i}", i*0.5, i*0.5+0.4) for i in range(5)], smoothed)
check("одиночный выброс сглажен", smoothed == ["A"] * 5, smoothed)

print("\nbuild_segments")

segments = merge.build_segments(words, labels)
check("две реплики", len(segments) == 2, [s.speaker for s in segments])
check("первая реплика от A", segments[0].speaker == "A")
check("текст собран", segments[0].text == "привет как дела", segments[0].text)
check("границы по словам", segments[0].start == 0.1 and segments[0].end == 1.8)
check("слова сохранены", len(segments[0].words) == 3)

# Разрыв по длинной паузе внутри одного говорящего
gap_words = [Word("раз", 0.0, 0.4), Word("два", 3.0, 3.4)]
gap_segments = merge.build_segments(gap_words, ["A", "A"])
check("длинная пауза рвёт реплику", len(gap_segments) == 2, len(gap_segments))

# Пустой вход
check("пустой вход не падает", merge.build_segments([], []) == [])

# Очень длинная реплика режется по MAX_SEGMENT_SEC
long_words = [Word(f"с{i}", i * 0.5, i * 0.5 + 0.4) for i in range(100)]
long_segments = merge.build_segments(long_words, ["A"] * 100)
check("длинная речь нарезана", len(long_segments) > 1, len(long_segments))
check(
    "ни одна реплика не длиннее предела",
    all(s.end - s.start <= merge.MAX_SEGMENT_SEC + 1 for s in long_segments),
    [round(s.end - s.start, 1) for s in long_segments],
)

print("\nsegments_from_asr_only")
asr_only = [
    merge.Segment(0.0, 1.9, None, "привет как дела"),
    merge.Segment(2.6, 4.2, None, "нормально спасибо"),
]
result = merge.segments_from_asr_only(asr_only, turns)
check("фразам назначены говорящие", [s.speaker for s in result] == ["A", "B"],
      [s.speaker for s in result])

print()
if failures:
    print(f"ПРОВАЛЕНО: {len(failures)} -> {failures}")
    sys.exit(1)
print("Все проверки merge пройдены")
