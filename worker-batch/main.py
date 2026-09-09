"""Batch-воркер: тянет задания из очереди и обрабатывает по одному.

Одно задание за раз — сознательно: на 12 ГБ VRAM параллельная обработка
двух записей крупной моделью просто не помещается.
"""
from __future__ import annotations

import logging
import signal
import time

import httpx

import diarize
import pipeline
from client import BackendClient
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("worker-batch")

_running = True


def _stop(signum, frame) -> None:
    global _running
    log.info("Получен сигнал %s, останавливаюсь после текущего задания", signum)
    _running = False


def wait_for_backend(client: BackendClient, attempts: int = 90) -> None:
    """Ждём backend. Тип __none__ заведомо не совпадёт ни с одним заданием,
    поэтому запрос работает как безопасный ping и ничего не забирает."""
    for i in range(attempts):
        try:
            client.next_job(types="__none__")
            log.info("Backend доступен")
            return
        except httpx.HTTPError:
            if i == 0:
                log.info("Жду backend...")
            time.sleep(2)
    raise RuntimeError("Backend так и не поднялся")


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    client = BackendClient()
    wait_for_backend(client)

    # Веса тяжёлые, поэтому греем заранее: иначе первая же запись
    # висит в обработке лишнюю минуту без всякого прогресса.
    diarize.warm_up()
    log.info("Готов. Опрашиваю очередь каждые %.1f c", settings.poll_interval)

    while _running:
        try:
            job = client.next_job()
        except httpx.HTTPError as exc:
            log.warning("Backend недоступен (%s), повтор через 5 с", exc)
            time.sleep(5)
            continue

        if job is None:
            time.sleep(settings.poll_interval)
            continue

        job_id = job["id"]
        log.info("Взял задание %s (%s) для сессии %s",
                 job_id, job["type"], job["session_id"])
        started = time.monotonic()
        try:
            result = pipeline.process(
                job,
                report=lambda stage, value: client.progress(job_id, stage, value),
            )
            client.submit_result(job_id, result)
            log.info("Задание %s готово за %.1f c", job_id, time.monotonic() - started)
        except Exception as exc:
            log.exception("Задание %s провалено", job_id)
            client.submit_failure(job_id, f"{type(exc).__name__}: {exc}")

    client.close()
    log.info("Остановлен")


if __name__ == "__main__":
    main()
