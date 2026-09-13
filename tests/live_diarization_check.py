"""Проверка живой диаризации: несколько записей подряд или одновременно.

Гонит webm/opus в /ws/live с реальной скоростью, как это делает браузер,
и для каждой записи считает, сколько разных говорящих оказалось в
живых репликах. Баг, который проверяем: после первой записи diart
переставал получать звук, и все последующие шли одним «Спикером 1».

    python live_diarization_check.py seq 2 /data/stream_dialog.webm
    python live_diarization_check.py par 2 /data/stream_dialog.webm
"""
import asyncio
import json
import subprocess
import sys
import time
from urllib.parse import quote

import websockets

CHUNK_MS = 200
TITLE = "Проверка диаризации — можно удалить"


def duration_of(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


async def one_session(tag: str, path: str, seconds: float) -> dict:
    data = open(path, "rb").read()
    n = max(1, int(seconds * 1000 / CHUNK_MS))
    size = max(1, len(data) // n)
    chunks = [data[i:i + size] for i in range(0, len(data), size)]

    url = f"ws://localhost:8001/ws/live?language=ru&title={quote(TITLE)}"
    result = {"tag": tag, "session": None, "lines": 0, "speakers": [],
              "finalized": False, "refine": None, "error": None}
    lines: list[dict] = []
    done = asyncio.Event()

    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            nonlocal lines
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "session":
                    result["session"] = msg["session_id"]
                elif kind == "transcript":
                    if msg.get("lines"):
                        lines = msg["lines"]
                elif kind == "finalized":
                    result["finalized"] = True
                    result["refine"] = msg.get("refine")
                    done.set()
                    return
                elif kind == "error":
                    result["error"] = msg.get("message")
                    done.set()
                    return

        task = asyncio.create_task(reader())
        started = time.monotonic()
        for i, chunk in enumerate(chunks):
            if done.is_set():
                break
            await ws.send(chunk)
            # держим реальную скорость: браузер не присылает быстрее
            target = started + (i + 1) * CHUNK_MS / 1000
            await asyncio.sleep(max(0.0, target - time.monotonic()))
        if not done.is_set():
            await ws.send(json.dumps({"type": "stop"}))
            try:
                await asyncio.wait_for(done.wait(), timeout=120)
            except asyncio.TimeoutError:
                result["error"] = "финализация не пришла за 120 с"
        task.cancel()

    speakers = sorted({l.get("speaker") for l in lines if l.get("speaker")})
    result["lines"] = len(lines)
    result["speakers"] = speakers
    return result


async def main() -> int:
    mode, count, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seconds = duration_of(path)
    print(f"режим {mode}, записей {count}, поток {seconds:.1f} с", flush=True)

    if mode == "seq":
        results = []
        for k in range(count):
            results.append(await one_session(f"{mode}{k + 1}", path, seconds))
            print("RESULT " + json.dumps(results[-1], ensure_ascii=False), flush=True)
    else:
        async def staggered(k):
            await asyncio.sleep(k * 1.5)  # не впритык: так ближе к жизни
            return await one_session(f"{mode}{k + 1}", path, seconds)
        results = await asyncio.gather(*(staggered(k) for k in range(count)))
        for r in results:
            print("RESULT " + json.dumps(r, ensure_ascii=False), flush=True)

    ok = all(len(r["speakers"]) > 1 and r["finalized"] and not r["error"] for r in results)
    print("ИТОГ:", "у каждой записи больше одного говорящего" if ok
          else "ЕСТЬ ЗАПИСИ С ОДНИМ ГОВОРЯЩИМ ИЛИ ОШИБКОЙ", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
