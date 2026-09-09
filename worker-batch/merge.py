"""Слияние результатов ASR и диаризации.

ASR знает, *что* сказано и когда; диаризация — *кто* говорил в каждый
момент. Связываем их по временному перекрытию на уровне отдельных слов,
затем собираем слова обратно в реплики.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Пауза, после которой начинаем новую реплику даже у того же говорящего
GAP_SPLIT_SEC = 0.8
# Жёсткий предел длины реплики — чтобы в интерфейсе не было простыней
MAX_SEGMENT_SEC = 30.0
MAX_SEGMENT_WORDS = 80
# Насколько далеко ищем говорящего, если слово не попало ни в один турн
NEAREST_TURN_SEC = 2.0
# Реплика короче этого — почти наверняка артефакт диаризации на границе
# фразы, а не настоящая смена говорящего.
MIN_TURN_SEC = 0.5
# Вставка внутри непрерывной речи одного человека: длиннее MIN_TURN_SEC,
# но по бокам от неё тот же говорящий и нет пауз. Такое почти всегда
# означает, что диаризация разрезала фразу, а не что кто-то вклинился.
SANDWICH_MAX_SEC = 1.5
SANDWICH_GAP_SEC = 0.4
SENTENCE_END = (".", "!", "?", "…")


@dataclass
class Word:
    w: str
    s: float
    e: float
    p: float = 1.0


@dataclass
class Turn:
    start: float
    end: float
    label: str


@dataclass
class Segment:
    start: float
    end: float
    speaker: str | None
    text: str
    words: list[Word] = field(default_factory=list)


# Знаки, перед которыми пробел не ставится
NO_SPACE_BEFORE = ",.!?;:%…)»]}"
# Знаки, после которых пробел не ставится
NO_SPACE_AFTER = "(«[{"


def join_words(words: list["Word"]) -> str:
    """Собирает слова в текст, восстанавливая пунктуацию.

    Наивное " ".join даёт на русском заметный мусор: Whisper режет
    дефисные слова на два токена («как» + «-нибудь», «90» + «-е»), и в
    тексте появляется «как -нибудь». Поэтому токен, начинающийся с дефиса
    или со знака препинания, приклеиваем к предыдущему.
    """
    out: list[str] = []
    for word in words:
        token = word.w.strip()
        if not token:
            continue
        if not out:
            out.append(token)
            continue
        glue_left = token[0] in NO_SPACE_BEFORE or token.startswith("-")
        glue_right = out[-1][-1] in NO_SPACE_AFTER
        if glue_left or glue_right:
            out[-1] += token
        else:
            out.append(token)
    return " ".join(out).strip()


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def assign_speakers(words: list[Word], turns: list[Turn]) -> list[str | None]:
    """Каждому слову — говорящий с максимальным перекрытием по времени."""
    if not turns:
        return [None] * len(words)

    turns = sorted(turns, key=lambda t: t.start)
    result: list[str | None] = []
    lo = 0

    for word in words:
        # Окно кандидатов: турны сортированы по началу, поэтому двигаем
        # нижнюю границу, пока турны заведомо закончились до слова.
        while lo < len(turns) and turns[lo].end < word.s - NEAREST_TURN_SEC:
            lo += 1

        best_label: str | None = None
        best_overlap = 0.0
        nearest_label: str | None = None
        nearest_distance = float("inf")

        i = lo
        while i < len(turns) and turns[i].start <= word.e + NEAREST_TURN_SEC:
            turn = turns[i]
            ov = _overlap(word.s, word.e, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap, best_label = ov, turn.label
            if ov == 0.0:
                # расстояние до турна — на случай, если перекрытия нет вовсе
                distance = turn.start - word.e if turn.start > word.e else word.s - turn.end
                if 0 <= distance < nearest_distance:
                    nearest_distance, nearest_label = distance, turn.label
            i += 1

        if best_label is not None:
            result.append(best_label)
        elif nearest_label is not None and nearest_distance <= NEAREST_TURN_SEC:
            # Слово в паузе между репликами (вздох, «угу») — отдаём ближайшему.
            result.append(nearest_label)
        else:
            result.append(None)

    _smooth(words, result)
    return result


def _smooth(words: list[Word], labels: list[str | None]) -> None:
    """Убирает слишком короткие вставки чужого говорящего.

    Диаризация регулярно рвёт фразу посередине: на реальной записи
    получалось «Вы хотите / сберечь планету, / но противитесь эволюции»,
    где куски одной фразы приписаны разным людям. Настоящая реплика
    короче полусекунды почти не встречается — за такое время не сказать
    и слова, — так что короткие пробежки считаем артефактом.

    Если соседи по обе стороны согласны, отдаём вставку им. Если не
    согласны, отдаём той стороне, которая говорила дольше: она вероятнее
    и есть хозяин фразы.
    """
    if not labels:
        return

    # Границы пробежек одного говорящего
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i))
            start = i

    def duration(run: tuple[int, int]) -> float:
        first, last = run
        return words[last - 1].e - words[first].s

    def is_artifact(index: int) -> bool:
        run = runs[index]
        if duration(run) < MIN_TURN_SEC:
            return True
        # Фраза, разрезанная посередине: слева и справа один и тот же
        # человек, а пауз вокруг вставки нет — значит он и говорил.
        before, after = runs[index - 1], runs[index + 1]
        if labels[before[0]] != labels[after[0]]:
            return False
        gap_before = words[run[0]].s - words[before[1] - 1].e
        gap_after = words[after[0]].s - words[run[1] - 1].e
        return (
            duration(run) < SANDWICH_MAX_SEC
            and gap_before < SANDWICH_GAP_SEC
            and gap_after < SANDWICH_GAP_SEC
        )

    changed = True
    while changed and len(runs) > 2:
        changed = False
        for index in range(1, len(runs) - 1):
            run = runs[index]
            if not is_artifact(index):
                continue
            before, after = runs[index - 1], runs[index + 1]
            winner = (
                labels[before[0]]
                if labels[before[0]] == labels[after[0]]
                or duration(before) >= duration(after)
                else labels[after[0]]
            )
            for i in range(*run):
                labels[i] = winner
            # Пробежки пересобираем: слияние могло соединить соседей.
            runs = []
            start = 0
            for i in range(1, len(labels) + 1):
                if i == len(labels) or labels[i] != labels[start]:
                    runs.append((start, i))
                    start = i
            changed = True
            break


def build_segments(words: list[Word], labels: list[str | None]) -> list[Segment]:
    """Собирает слова в реплики: по смене говорящего, паузе и длине."""
    segments: list[Segment] = []
    buf: list[Word] = []
    buf_label: str | None = None

    def flush() -> None:
        nonlocal buf, buf_label
        if not buf:
            return
        text = join_words(buf)
        if text:
            segments.append(
                Segment(start=buf[0].s, end=buf[-1].e, speaker=buf_label,
                        text=text, words=list(buf))
            )
        buf = []

    for word, label in zip(words, labels):
        if buf:
            gap = word.s - buf[-1].e
            duration = word.e - buf[0].s
            sentence_boundary = (
                buf[-1].w.rstrip().endswith(SENTENCE_END) and len(buf) >= 12
            )
            if (
                label != buf_label
                or gap > GAP_SPLIT_SEC
                or duration > MAX_SEGMENT_SEC
                or len(buf) >= MAX_SEGMENT_WORDS
                or sentence_boundary
            ):
                flush()
        if not buf:
            buf_label = label
        buf.append(word)

    flush()
    return segments


def segments_from_asr_only(
    asr_segments: list[Segment], turns: list[Turn]
) -> list[Segment]:
    """Запасной путь, когда пословных таймингов нет.

    Тогда говорящего назначаем целой фразе — по наибольшему перекрытию.
    """
    for seg in asr_segments:
        best_label, best_overlap = None, 0.0
        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap, best_label = ov, turn.label
        seg.speaker = best_label
    return asr_segments
