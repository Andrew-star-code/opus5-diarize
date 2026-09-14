"""Сквозной замер: доходит ли начало новой реплики до нового говорящего.

Гонит записи из библиотеки через /ws/live в реальном времени, как браузер
(webm/opus порциями по 200 мс), берёт итоговые реплики живого черновика
и сопоставляет их слова со словами чистовика по тексту. Дальше — как на
стенде: доля слов под чужим именем после подбора соответствия меток и
на каждой смене говорящего по чистовику — сколько первых слов новой
реплики досталось предыдущему (и наоборот).

Время слов здесь — от SimulStreaming, как в жизни, а не от чистового
Whisper, как на стенде (eval_live.py). Поэтому решения о правилах
живого черновика проверяются здесь: сдвиг разметки на 0,08 с стенд
одобрил, а в живом режиме он оказался хуже.

Нужен работающий worker-live; доводку на время замера лучше выключить
(AUTO_REFINE_LIVE=false), чтобы она не делила видеокарту. Записи
попадают в библиотеку с заголовком «Проверка границ реплик — можно
удалить».

    docker compose cp bench/live_boundary_e2e.py worker-live:/srv/
    docker compose cp bench/cache/live_refs.json worker-live:/srv/
    docker compose exec worker-live python /srv/live_boundary_e2e.py <метка> [sid ...]
"""
import asyncio
import difflib
import json
import re
import subprocess
import sys
import time
from urllib.parse import quote

import numpy as np
import websockets
from scipy.optimize import linear_sum_assignment

REFS = json.load(open("/srv/live_refs.json", encoding="utf-8"))
DEFAULT = ["97fa62f740c24171", "3f99a4c9049d4e8a", "2fa8e75dab7f442d",
           "2fcb2b22be974cc6", "63a6c0e98aec4e00"]
CHUNK_MS = 200
TITLE = "Проверка границ реплик — можно удалить"


def webm(path: str) -> tuple[bytes, float]:
    data = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-c:a", "libopus",
                           "-b:a", "32k", "-f", "webm", "pipe:1"], capture_output=True, check=True).stdout
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", path], capture_output=True, text=True,
                               check=True).stdout.strip())
    return data, dur


async def stream(path: str) -> list[dict]:
    data, seconds = webm(path)
    n = max(1, int(seconds * 1000 / CHUNK_MS))
    size = max(1, len(data) // n)
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    lines: list[dict] = []
    done = asyncio.Event()
    url = f"ws://localhost:8001/ws/live?language=ru&title={quote(TITLE)}"
    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            nonlocal lines
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "transcript" and msg.get("lines"):
                    lines = msg["lines"]
                elif msg.get("type") in ("finalized", "error"):
                    done.set()
                    return
        task = asyncio.create_task(reader())
        started = time.monotonic()
        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            await asyncio.sleep(max(0.0, started + (i + 1) * CHUNK_MS / 1000 - time.monotonic()))
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            pass
        task.cancel()
    return lines


def norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower().replace("ё", "е"))


def score(lines: list[dict], ref_words: list) -> dict:
    live = [(norm(w), line["speaker"]) for line in lines for w in line["text"].split() if norm(w)]
    ref = [(norm(w[3]), w[2]) for w in ref_words if norm(w[3])]
    sm = difflib.SequenceMatcher(a=[x[0] for x in ref], b=[x[0] for x in live], autojunk=False)
    pairs = [(a + k, b + k) for a, b, size in sm.get_matching_blocks() for k in range(size)]
    ref_labels = sorted({ref[i][1] for i, _ in pairs})
    hyp_labels = sorted({live[j][1] for _, j in pairs})
    counts = np.zeros((len(hyp_labels), len(ref_labels)))
    for i, j in pairs:
        counts[hyp_labels.index(live[j][1]), ref_labels.index(ref[i][1])] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping = {hyp_labels[r]: ref_labels[c] for r, c in zip(rows, cols)}
    ref_seq = [ref[i][1] for i, _ in pairs]
    hyp_seq = [mapping.get(live[j][1]) for _, j in pairs]
    changes = late = late_words = early = early_words = 0
    for i in range(1, len(ref_seq)):
        if ref_seq[i] == ref_seq[i - 1]:
            continue
        changes += 1
        k = 0
        while i + k < len(ref_seq) and k < 5 and ref_seq[i + k] == ref_seq[i] and hyp_seq[i + k] == ref_seq[i - 1]:
            k += 1
        late += bool(k)
        late_words += k
        k = 0
        while i - 1 - k >= 0 and k < 5 and ref_seq[i - 1 - k] == ref_seq[i - 1] and hyp_seq[i - 1 - k] == ref_seq[i]:
            k += 1
        early += bool(k)
        early_words += k
    wrong = sum(1 for a, b in zip(ref_seq, hyp_seq) if a != b)
    return {"matched": len(pairs), "ref_words": len(ref), "wrong": wrong, "changes": changes,
            "late": late, "late_words": late_words, "early": early, "early_words": early_words,
            "lines": len(lines)}


async def main() -> None:
    tag, sids = sys.argv[1], sys.argv[2:] or DEFAULT
    total: dict = {}
    for sid in sids:
        lines = await stream("/data/" + REFS[sid]["audio"])
        r = score(lines, REFS[sid]["words"])
        print(f"{tag} {sid[:8]}: слов сопоставлено {r['matched']}/{r['ref_words']}, под чужим именем "
              f"{r['wrong'] / max(1, r['matched']) * 100:5.1f}%, смен {r['changes']}, начало ушло предыдущему "
              f"{r['late']} ({r['late_words']} слов), конец ушёл новому {r['early']} ({r['early_words']} слов)",
              flush=True)
        for key, value in r.items():
            total[key] = total.get(key, 0) + value
    n = max(1, total["changes"])
    print(f"ИТОГ {tag}: под чужим именем {total['wrong'] / max(1, total['matched']) * 100:.1f}%, смен {total['changes']}, "
          f"начало ушло предыдущему {total['late'] / n * 100:.1f}% смен ({total['late_words']} слов), "
          f"конец ушёл новому {total['early'] / n * 100:.1f}% смен ({total['early_words']} слов), "
          f"реплик {total['lines']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
