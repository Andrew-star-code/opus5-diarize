"""Клиент к локальной языковой модели — сервис llm на Ollama.

Модель нужна чистовику только для «полировки»: поправить стыки реплик,
придумать название, краткое содержание, подсказать имена говорящих. Всё
это надстройка: если сервис не отвечает или ответ не по схеме, шаг
пропускается, а чистовик уходит как есть. Поэтому любой сбой здесь —
LLMError, и вызывающий код решает, что пропустить.

Ответ всегда строго по JSON-схеме (поле format в /api/chat): модель
физически не может написать ничего, кроме полей и значений из схемы.
Модель загружается в видеопамять на время доводки и выгружается после
простоя (keep_alive) — живой режим и Whisper память не теряют.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Модель недоступна или ответила не по схеме."""


class LLM:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or settings.llm_url).rstrip("/")
        self.model = model or settings.llm_model
        # Первое обращение после простоя грузит модель в видеопамять —
        # это десятки секунд, отсюда длинное чтение.
        self._http = httpx.Client(timeout=httpx.Timeout(10.0, read=settings.llm_timeout))
        self._think = True  # поле think понимают не все модели — см. ask()

    def close(self) -> None:
        self._http.close()

    def available(self) -> bool:
        """Сервис отвечает и нужная модель скачана."""
        try:
            r = self._http.get(f"{self.url}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Языковая модель недоступна (%s): %s", self.url, exc)
            return False
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return True
        log.warning("Модель %s не скачана в сервисе llm — есть: %s", self.model, sorted(names))
        return False

    def ask(self, system: str, user: str, schema: dict[str, Any], *, num_ctx: int = 8192) -> Any:
        """Один вопрос — один ответ строго по JSON-схеме."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": settings.llm_keep_alive,
            # Детерминированно: одна и та же запись — один и тот же ответ.
            "options": {"temperature": 0, "num_ctx": num_ctx},
        }
        if self._think:
            # Рассуждения вслух (Qwen3) здесь только тратят время: ответ —
            # одно число или короткий текст.
            payload["think"] = False
        try:
            r = self._http.post(f"{self.url}/api/chat", json=payload)
            if r.status_code == 400 and self._think and "think" in r.text:
                self._think = False
                payload.pop("think")
                r = self._http.post(f"{self.url}/api/chat", json=payload)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return json.loads(content)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
