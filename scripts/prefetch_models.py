"""Разовая загрузка всех весов в ./models.

Запускается ОДИН раз, пока есть интернет. После этого сервис живёт с
HF_HUB_OFFLINE=1 и в сеть не ходит вообще.

    docker compose run --rm -e HF_HUB_OFFLINE=0 -v ./scripts:/scripts \
        worker-batch python /scripts/prefetch_models.py
    docker compose run --rm -e HF_HUB_OFFLINE=0 -v ./scripts:/scripts \
        worker-live python /scripts/prefetch_models.py

Скрипт запускают в обоих воркерах: у них разные окружения, и каждый
тянет то, что умеет. Отсутствие библиотеки — не ошибка, а ожидаемое
состояние в чужом контейнере.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HF_HOME = os.environ.get("HF_HOME", "/models")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

BATCH_ASR = os.environ.get("BATCH_ASR_MODEL", "large-v3")
LIVE_ASR = os.environ.get("LIVE_ASR_MODEL", "large-v3-turbo")
DIARIZATION = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)

# Кирпичи, из которых diart собирает онлайн-диаризацию. Скачиваем явно:
# при первом запуске без сети он иначе просто падает.
DIART_MODELS = [
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    # Модель эмбеддингов по умолчанию в diart. Без неё живая диаризация
    # в оффлайне падает на LocalEntryNotFoundError.
    "pyannote/embedding",
]

results: list[tuple[str, str]] = []


def report(name: str, status: str) -> None:
    results.append((name, status))
    print(f"  {status:<9} {name}", flush=True)


def fetch_whisper(name: str) -> None:
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        report(f"whisper:{name}", "пропуск")
        return
    try:
        # download_model сам сопоставляет короткое имя с репозиторием,
        # поэтому названия репозиториев тут не хардкодятся.
        # cache_dir не задаём намеренно: библиотеки по умолчанию берут
        # $HF_HOME/hub, и если писать в $HF_HOME напрямую, получаются два
        # разных кеша — скачанное лежит в одном, а ищут в другом.
        download_model(name)
        report(f"whisper:{name}", "готово")
    except Exception as exc:
        report(f"whisper:{name}", "ОШИБКА")
        print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)


def fetch_pyannote_pipeline(name: str) -> None:
    try:
        import pyannote.audio
        from pyannote.audio import Pipeline
    except ImportError:
        report(f"pyannote:{name}", "пропуск")
        return

    # В live-контейнере стоит pyannote 3.x — его тянет diart. Пайплайн
    # community-1 умеет только 4.x, и живому режиму он не нужен: там своя
    # диаризация на segmentation-3.0 и wespeaker. Так что это не ошибка,
    # а просто чужой контейнер.
    major = int(str(pyannote.audio.__version__).split(".")[0] or 0)
    if major < 4:
        report(f"pyannote:{name}", "пропуск")
        print(f"      нужен pyannote.audio 4.x, установлен "
              f"{pyannote.audio.__version__} — это окружение live-воркера")
        return

    try:
        pipeline = Pipeline.from_pretrained(name, token=HF_TOKEN)
        if pipeline is None:
            raise RuntimeError(
                "Hugging Face вернул пусто. Обычно это значит, что лицензия "
                f"модели не принята: откройте https://huggingface.co/{name} "
                "и нажмите Agree, затем повторите с корректным HF_TOKEN."
            )
        report(f"pyannote:{name}", "готово")
    except Exception as exc:
        report(f"pyannote:{name}", "ОШИБКА")
        print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)


def fetch_simulstreaming_checkpoint() -> None:
    """Оригинальный чекпоинт OpenAI для декодера SimulStreaming.

    Живому режиму мало CTranslate2-модели: политика SimulStreaming берёт
    внимание из исходной модели Whisper, а это отдельный файл на полтора
    гигабайта. Штатный загрузчик whisperlivekit не умеет докачку и на
    нестабильном канале раз за разом отдаёт обрезанный файл, который потом
    отваливается по контрольной сумме. Поэтому качаем сами, с возобновлением
    и проверкой хеша — он, кстати, зашит прямо в путь URL.
    """
    try:
        from whisperlivekit.whisper import _MODELS
    except ImportError:
        report("checkpoint:simulstreaming", "пропуск")
        return

    name = LIVE_ASR
    url = _MODELS.get(name)
    if url is None:
        report(f"checkpoint:{name}", "пропуск")
        return

    import urllib.request

    # Хеш зашит в путь URL — отдельного файла с суммой не нужно.
    expected = url.split("/")[-2]
    target = Path(HF_HOME) / f"{name}.pt"

    if target.exists() and _sha256(target) == expected:
        report(f"checkpoint:{name}", "готово")
        return

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Докачиваем с того места, где оборвалось в прошлый раз.
        for _ in range(5):
            done = target.stat().st_size if target.exists() else 0
            request = urllib.request.Request(url)
            if done:
                request.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(request, timeout=120) as response, \
                    target.open("ab" if done else "wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if _sha256(target) == expected:
                report(f"checkpoint:{name}", "готово")
                return
        raise RuntimeError("контрольная сумма не сошлась после 5 попыток")
    except Exception as exc:
        report(f"checkpoint:{name}", "ОШИБКА")
        print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_repo(repo: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        report(repo, "пропуск")
        return
    try:
        snapshot_download(repo, token=HF_TOKEN)
        report(repo, "готово")
    except Exception as exc:
        report(repo, "ОШИБКА")
        print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> int:
    if os.environ.get("HF_HUB_OFFLINE") in ("1", "true", "True"):
        print(
            "HF_HUB_OFFLINE=1 — скачивать нечем. Запустите с "
            "-e HF_HUB_OFFLINE=0.",
            file=sys.stderr,
        )
        return 2

    print(f"Каталог моделей: {HF_HOME}")
    print(f"Токен HF: {'задан' if HF_TOKEN else 'не задан'}\n")

    print("Модели распознавания:")
    for name in dict.fromkeys([BATCH_ASR, LIVE_ASR]):
        fetch_whisper(name)
    # Нужен только живому режиму; в batch-контейнере тихо пропускается.
    fetch_simulstreaming_checkpoint()

    print("\nДиаризация:")
    fetch_pyannote_pipeline(DIARIZATION)
    for repo in DIART_MODELS:
        fetch_repo(repo)

    failed = [name for name, status in results if status == "ОШИБКА"]
    done = [name for name, status in results if status == "готово"]

    print(f"\nСкачано: {len(done)}, ошибок: {len(failed)}")
    if failed:
        print("Не получилось: " + ", ".join(failed), file=sys.stderr)
        print(
            "\nЧаще всего причина одна из двух: не принята лицензия модели "
            "на huggingface.co или в .env нет HF_TOKEN.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
