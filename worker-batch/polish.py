"""Полировка чистовика языковой моделью: поправка стыков реплик.

Разметка говорящих ставит границу по звуку и на стыке реплик ошибается на
слово-два: начало фразы нового говорящего достаётся предыдущему или
наоборот. Текст при этом часто подсказывает однозначно: «…как ты
думаешь?» — «Да, я думаю…» — ответ начинает другой человек.

Модель здесь ничего не пишет. На каждой смене говорящего она выбирает из
пронумерованных вариантов, с какого слова на самом деле начинается новая
реплика: номер ограничен JSON-схемой (перечисление), так что выдумать
слово или сдвинуть границу дальше чем на max_shift слов она не может, а
соседнюю границу граница не переходит. Идея — DiarizationLM (Google,
2024) в самой узкой форме.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from merge import SENTENCE_END, Word

log = logging.getLogger(__name__)

# Сколько слов по обе стороны границы показываем модели.
CONTEXT_WORDS = 30
# Стык, где прежняя реплика кончается предложением и за ней пауза не
# короче этого, уверенный — модель о нём не спрашиваем.
CONFIDENT_GAP_SEC = 0.3

BOUNDARY_SYSTEM = (
    "Ты выверяешь расшифровку разговора. Тебе показывают кусок текста, где "
    "стоит граница между репликами двух разных людей: до неё говорит А, после "
    "неё — Б. Границу поставили по звуку, и она может ошибаться на пару слов. "
    "По смыслу реши, с какого слова на самом деле начинается реплика Б: "
    "вопрос и ответ, обращение, «да»/«нет» в начале ответа, оборванная мысль. "
    "Слова пронумерованы. Если граница стоит верно, верни номер первого слова Б. "
    "Отвечай только номером."
)

AskFn = Callable[[str, str, dict[str, Any]], Any]


def _ends_sentence(word: Word) -> bool:
    return word.w.rstrip("»\"')").endswith(SENTENCE_END)


def _confident(words: list[Word], i: int) -> bool:
    prev = words[i - 1]
    return _ends_sentence(prev) and words[i].s - prev.e >= CONFIDENT_GAP_SEC


def _boundaries(labels: list[str | None]) -> list[int]:
    """Индексы первых слов новой реплики — только между двумя известными говорящими."""
    return [
        i for i in range(1, len(labels))
        if labels[i] != labels[i - 1] and labels[i] is not None and labels[i - 1] is not None
    ]


def adjust_boundaries(
    words: list[Word],
    labels: list[str | None],
    ask: AskFn,
    *,
    max_shift: int = 3,
    skip_confident: bool = True,
    show_pauses: bool = False,
    sentence_end_only: bool = False,
) -> tuple[list[str | None], dict[str, int]]:
    """Новые метки слов и счётчики: спрошено, сдвинуто границ, слов сменило говорящего.

    ask(system, user, schema) — вызов модели; бросает исключение при сбое.
    show_pauses — показывать модели паузы между словами: смена говорящего
    почти всегда приходится на паузу. sentence_end_only — принимать сдвиг,
    только если новая граница встаёт сразу после конца предложения.
    Первый же сбой прекращает опрос: уже сделанные сдвиги остаются, прочие
    границы — как были.
    """
    out = list(labels)
    stats = {"boundaries": 0, "asked": 0, "moved": 0, "words": 0}
    if len(words) != len(labels):
        log.warning("Поправка стыков пропущена: слов %d, меток %d", len(words), len(labels))
        return out, stats
    bounds = _boundaries(out)
    stats["boundaries"] = len(bounds)
    prev_b = 0
    for n, b in enumerate(bounds):
        next_b = bounds[n + 1] if n + 1 < len(bounds) else len(out)
        lo, hi = max(prev_b + 1, b - max_shift), min(next_b - 1, b + max_shift)
        if hi <= lo or (skip_confident and _confident(words, b)):
            prev_b = b
            continue

        a_label, b_label = out[b - 1], out[b]
        start, stop = max(prev_b, b - CONTEXT_WORDS), min(next_b, b + CONTEXT_WORDS)
        def numbered(k: int) -> str:
            text = f"[{k - start + 1}]{words[k].w}"
            if show_pauses and k > start:
                gap = words[k].s - words[k - 1].e
                if gap >= 0.3:
                    text = f"(пауза {gap:.1f} с) " + text
            return text
        user = (
            "А: " + " ".join(numbered(k) for k in range(start, b)) + "\n"
            "Б: " + " ".join(numbered(k) for k in range(b, stop)) + "\n\n"
            f"Сейчас реплика Б начинается со слова [{b - start + 1}]. "
            "С какого слова она начинается на самом деле?"
        )
        candidates = range(lo, hi + 1)
        if sentence_end_only:
            # Кроме нынешней границы — только места сразу после конца предложения.
            candidates = [k for k in candidates if k == b or _ends_sentence(words[k - 1])]
            if len(candidates) < 2:
                prev_b = b
                continue
        choices = [k - start + 1 for k in candidates]
        schema = {
            "type": "object",
            "properties": {"start": {"type": "integer", "enum": choices}},
            "required": ["start"],
        }
        try:
            answer = ask(BOUNDARY_SYSTEM, user, schema)
            new_b = start + int(answer["start"]) - 1
        except Exception as exc:  # noqa: BLE001 — модель — надстройка, не повод ронять чистовик
            log.warning("Поправка стыков остановлена на границе %d из %d: %s", n + 1, len(bounds), exc)
            break
        stats["asked"] += 1
        if new_b not in candidates:
            new_b = b  # ответ вне вариантов — схема такого не допускает, но на всякий случай
        if new_b < b:
            out[new_b:b] = [b_label] * (b - new_b)
        elif new_b > b:
            out[b:new_b] = [a_label] * (new_b - b)
        if new_b != b:
            stats["moved"] += 1
            stats["words"] += abs(new_b - b)
        prev_b = new_b
    return out, stats


# ─── название, краткое содержание, имена говорящих ──────────────

NOTES_SYSTEM = (
    "Ты помогаешь с расшифровкой разговора: реплики подписаны метками "
    "говорящих (SPEAKER_00, SPEAKER_01…). Отвечай на языке расшифровки. "
    "Опирайся только на текст расшифровки и ничего не додумывай."
)
# Часть расшифровки на один запрос: помещается в окно модели вместе с
# ответом и не раздувает видеопамять под контекст.
CHUNK_WORDS = 2000

POINTS_SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "items": {"type": "string"}, "maxItems": 7}},
    "required": ["points"],
}
TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "maxLength": 80}},
    "required": ["title"],
}


def _flat(text: str) -> str:
    """Для сверки цитат: нижний регистр, ё → е, без знаков, пробелы одинарные."""
    import re

    text = text.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _chunks(lines: list[str], limit: int = CHUNK_WORDS) -> list[str]:
    chunks, current, count = [], [], 0
    for line in lines:
        words = len(line.split())
        if current and count + words > limit:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += words
    if current:
        chunks.append("\n".join(current))
    return chunks


def _summary(chunks: list[str], ask: AskFn) -> list[str]:
    points: list[str] = []
    for chunk in chunks:
        answer = ask(
            NOTES_SYSTEM,
            "Перескажи этот кусок разговора в 3–7 коротких пунктах: о чём шла "
            "речь, что решили, о чём договорились.\n\n" + chunk,
            POINTS_SCHEMA,
        )
        points += [p.strip() for p in answer.get("points", []) if p.strip()]
    if len(chunks) > 1 and points:
        answer = ask(
            NOTES_SYSTEM,
            "Вот пересказ частей одного разговора по порядку. Сведи его в 3–7 "
            "пунктов о разговоре целиком, без повторов.\n\n"
            + "\n".join(f"- {p}" for p in points),
            POINTS_SCHEMA,
        )
        points = [p.strip() for p in answer.get("points", []) if p.strip()]
    return points[:7]


def _title(points: list[str], ask: AskFn) -> str:
    answer = ask(
        NOTES_SYSTEM,
        "Придумай короткое название для записи этого разговора — 3–7 слов, "
        "без кавычек и точки в конце. О чём разговор:\n"
        + "\n".join(f"- {p}" for p in points),
        TITLE_SCHEMA,
    )
    return answer.get("title", "").strip().strip("«»\"'").rstrip(".").strip()[:80]


def _names(chunks: list[str], labels: list[str], ask: AskFn) -> dict[str, str]:
    """Имена — только с дословной цитатой из текста; иначе это выдумка модели."""
    schema = {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": labels},
                        "name": {"type": "string", "maxLength": 40},
                        "quote": {"type": "string"},
                    },
                    "required": ["speaker", "name", "quote"],
                },
            }
        },
        "required": ["names"],
    }
    prompt = (
        "Узнай имена говорящих — только если к человеку обращаются по имени "
        "или он сам представился. Имя, с которым к кому-то обращаются, "
        "принадлежит тому, к кому обращаются: обычно это тот, кто отвечает "
        "следующим, а не тот, кто имя произнёс. Для каждого имени приведи "
        "дословную цитату из текста, где оно звучит. Не уверен — не пиши.\n\n"
    )
    votes: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        answer = ask(NOTES_SYSTEM, prompt + chunk, schema)
        flat = _flat(chunk)
        for item in answer.get("names", []):
            label = item.get("speaker")
            name = " ".join(str(item.get("name", "")).split())
            quote = _flat(str(item.get("quote", "")))
            if label not in labels or not name or not quote:
                continue
            # Цитата — дословно из текста, и имя в ней (с точностью до
            # падежного окончания) действительно звучит.
            if quote not in flat or _flat(name)[:4] not in quote:
                continue
            votes.setdefault(label, {}).setdefault(name, 0)
            votes[label][name] += 1
    names = {label: max(counts, key=counts.get) for label, counts in votes.items()}
    # Одно имя двум людям — значит, модель запуталась: не подсказываем никому.
    taken: dict[str, int] = {}
    for name in names.values():
        taken[name] = taken.get(name, 0) + 1
    return {label: name for label, name in names.items() if taken[name] == 1}


def describe(segments: list, ask: AskFn) -> dict[str, Any]:
    """Краткое содержание, название и подсказки имён по готовому чистовику.

    Всё или часть может не получиться — тогда соответствующего ключа нет,
    чистовик уходит без этого.
    """
    lines = [f"{seg.speaker or 'SPEAKER_00'}: {seg.text}" for seg in segments if seg.text.strip()]
    if not lines:
        return {}
    chunks = _chunks(lines)
    labels = sorted({seg.speaker for seg in segments if seg.speaker})
    notes: dict[str, Any] = {}
    try:
        points = _summary(chunks, ask)
        if points:
            notes["summary"] = "\n".join(points)
            title = _title(points, ask)
            if title:
                notes["title"] = title
        if labels:
            names = _names(chunks, labels, ask)
            if names:
                notes["speaker_names"] = names
    except Exception as exc:  # noqa: BLE001 — надстройка, не повод ронять чистовик
        log.warning("Название и краткое содержание — не всё получилось: %s", exc)
    return notes
