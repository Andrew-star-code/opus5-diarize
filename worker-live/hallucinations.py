"""Галлюцинации Whisper: строчки из титров, которых в речи не бывает.

Whisper учился на фильмах с субтитрами и на музыке, тишине и коротких
шумных фразах выдаёт подписи их авторов: «Субтитры сделал DimaTorzok»,
«Редактор субтитров А.Семкин Корректор А.Егорова». На замере
(bench/eval_asr.py) так он «расшифровал» 34 короткие команды из 1916 на
Golos farfield, а на чистом чтении (FLEURS) дважды дописал подпись в
конец настоящей фразы — фильтр убирает только её.

Файл один и тот же в worker-live и worker-batch: воркеры собираются из
разных каталогов и общий модуль делить не могут. Что копии совпадают,
проверяет tests/test_hallucinations.py.
"""
from __future__ import annotations

import re

_AUTHOR = r"(?:DimaTorzok|Dima\s*Torzok|Amara\.org|«?Симон»?)"

# Вырезаются где угодно в тексте: в речи такие строчки не встречаются.
TITLES = [
    # «Субтитры сделал DimaTorzok», «Субтитры создавались сообществом
    # Amara.org», «Субтитры подогнал «Симон»»
    re.compile(rf"Субтитры\s+(?:\w+\s+){{1,2}}{_AUTHOR}[.!]*", re.I),
    # «Добавил субтитры DimaTorzok»
    re.compile(r"\w+\s+субтитры\s+(?:DimaTorzok|Dima\s*Torzok)[.!]*", re.I),
    # «Редактор субтитров А.Семкин Корректор А.Егорова»
    re.compile(r"Редактор\s+субтитров\s+[А-ЯЁA-Z]\.\s*\w+(?:\s+Корректор\s+[А-ЯЁA-Z]\.\s*\w+)?[.!]*", re.I),
]

# Вырезается, только если это вся фраза целиком. «Продолжение следует...»
# Whisper пишет на тишине (12 раз на Golos), но так говорят и
# по-настоящему — внутри живой фразы её не трогаем.
STANDALONE = re.compile(r"\s*Продолжение\s+следует\s*(?:\.{1,3}|…)?\s*", re.I)

# Сознательно не входят «Подписывайтесь на канал» и «Спасибо за
# просмотр»: в подкастах это обычная концовка, а не галлюцинация.


def drop_hallucinations(text: str) -> str:
    """Текст без строчек из титров; пустая строка, если он весь — галлюцинация."""
    text = text or ""
    if STANDALONE.fullmatch(text):
        return ""
    for pattern in TITLES:
        text = pattern.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text)


def hallucinated_words(words: list[str]) -> list[bool]:
    """Для каждого слова фразы — входит ли оно в строчку из титров.

    Слова склеиваются через пробел, шаблоны ищутся по склейке, и слово
    помечается, если задевает найденный кусок. Так в чистовике слова
    выбрасываются вместе со своими таймингами, а не вырезаются из текста.
    """
    text = " ".join(words)
    if STANDALONE.fullmatch(text):
        return [True] * len(words)
    spans = [m.span() for pattern in TITLES for m in pattern.finditer(text)]
    marks, pos = [], 0
    for word in words:
        start, end = pos, pos + len(word)
        marks.append(any(s < end and start < e for s, e in spans))
        pos = end + 1
    return marks
