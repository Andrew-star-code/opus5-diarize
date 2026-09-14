"""Сквозной тест backend: обе схемы работы, редактирование, экспорт."""
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="scribe-test-"))
os.environ["DB_PATH"] = str(WORK / "test.db")
os.environ["DATA_DIR"] = str(WORK / "data")
os.environ["AUTO_REFINE_LIVE"] = "true"
(WORK / "data" / "audio").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

failures = []


def check(name, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"  <{detail}>" if not condition else ""))
    if not condition:
        failures.append(name)


def make_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))


DRAFT = [
    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Привет, начнём.", "words": []},
    {"start": 2.4, "end": 5.0, "speaker": "SPEAKER_01", "text": "Да, я готова.", "words": []},
]

FINAL = [
    {
        "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00",
        "text": "Привет, начнём планёрку.",
        "words": [
            {"w": "Привет,", "s": 0.1, "e": 0.6, "p": 0.99},
            {"w": "начнём", "s": 0.7, "e": 1.2, "p": 0.97},
            {"w": "планёрку.", "s": 1.3, "e": 2.0, "p": 0.95},
        ],
    },
    {
        "start": 2.4, "end": 5.0, "speaker": "SPEAKER_01",
        "text": "Да, я готова.",
        "words": [
            {"w": "Да,", "s": 2.4, "e": 2.7, "p": 0.99},
            {"w": "я", "s": 2.8, "e": 2.9, "p": 0.98},
            {"w": "готова.", "s": 3.0, "e": 3.6, "p": 0.96},
        ],
    },
]

with TestClient(app) as client:
    print("служебное")
    r = client.get("/api/health")
    check("health отвечает", r.status_code == 200 and r.json()["ok"], r.text[:120])
    cfg = client.get("/api/config").json()
    check("config отдаёт форматы", len(cfg["formats"]) == 6, cfg.get("formats"))

    print("\nживая запись")
    r = client.post("/api/internal/live/sessions", json={"title": "Планёрка", "language": "ru"})
    check("сессия создана", r.status_code == 201, r.text[:200])
    live_id = r.json()["id"]
    check("статус recording", r.json()["status"] == "recording", r.json()["status"])

    audio = WORK / "data" / "audio" / live_id / "live.wav"
    make_wav(audio, 5.0)

    r = client.post(
        f"/api/internal/live/sessions/{live_id}/finalize",
        json={
            "audio_path": f"audio/{live_id}/live.wav",
            "duration_sec": 5.0,
            "segments": DRAFT,
            "model_info": {"pipeline": "live"},
            "refine": True,
        },
    )
    check("черновик сохранён", r.status_code == 200, r.text[:200])
    check("помечен как draft", r.json()["quality"] == "draft", r.json()["quality"])
    check("два говорящих", r.json()["speaker_count"] == 2, r.json()["speaker_count"])
    check("лента построена", len(r.json()["ribbon"]) == 240, len(r.json()["ribbon"]))
    ribbon = set(r.json()["ribbon"])
    check("в ленте оба говорящих", {0, 1} <= ribbon, sorted(ribbon))
    # -1 — пауза между репликами (2.0-2.4 с), она должна быть видна
    check("пауза в ленте отмечена", -1 in ribbon, sorted(ribbon))

    print("\nдоводка через очередь")
    r = client.get("/api/internal/jobs/next", params={"types": "batch,refine"})
    check("задание доводки выдано", r.status_code == 200 and r.json(), r.text[:200])
    job = r.json()
    check("тип refine", job["type"] == "refine", job["type"])
    check("путь к аудио передан", job["audio_path"].endswith("live.wav"), job["audio_path"])

    r2 = client.get("/api/internal/jobs/next", params={"types": "batch,refine"})
    check("второй раз очередь пуста", r2.status_code == 204, r2.status_code)

    r = client.post(f"/api/internal/jobs/{job['id']}/progress",
                    json={"stage": "asr", "progress": 0.4})
    check("прогресс принят", r.status_code == 204, r.status_code)

    r = client.post(
        f"/api/internal/jobs/{job['id']}/result",
        json={
            "language": "ru", "duration_sec": 5.0,
            "model_info": {"asr": "large-v3", "pipeline": "batch"},
            "segments": FINAL,
            # Полировка языковой моделью: название, краткое содержание, имя.
            "title": "Планёрка по отчёту",
            "summary": "Обсудили отчёт\nРешили начать планёрку",
            "speaker_names": {"SPEAKER_00": "Антон"},
        },
    )
    check("результат принят", r.status_code == 204, r.text[:200])

    detail = client.get(f"/api/sessions/{live_id}").json()
    check("стал чистовиком", detail["quality"] == "final", detail["quality"])
    check("статус ready", detail["status"] == "ready", detail["status"])
    check("пословные тайминги приехали", len(detail["segments"][0]["words"]) == 3)
    check("аудио доступно", detail["audio_url"] is not None)
    check("краткое содержание сохранено", detail["summary"] == "Обсудили отчёт\nРешили начать планёрку",
          detail.get("summary"))
    check("своё название не перезаписано", detail["title"] == "Планёрка", detail["title"])
    suggested = {s["label"]: s["suggested_name"] for s in detail["speakers"]}
    check("имя подсказано только найденному говорящему",
          suggested == {"SPEAKER_00": "Антон", "SPEAKER_01": ""}, suggested)

    print("\nавтоназвание заменяется названием от модели")
    r = client.post("/api/internal/live/sessions", json={"language": "ru"})
    auto_id = r.json()["id"]
    check("автоназвание по дате", r.json()["title"].startswith("Запись "), r.json()["title"])
    make_wav(WORK / "data" / "audio" / auto_id / "live.wav", 5.0)
    client.post(f"/api/internal/live/sessions/{auto_id}/finalize", json={
        "audio_path": f"audio/{auto_id}/live.wav", "duration_sec": 5.0,
        "segments": DRAFT, "model_info": {"pipeline": "live"}, "refine": True,
    })
    auto_job = client.get("/api/internal/jobs/next", params={"types": "batch,refine"}).json()
    client.post(f"/api/internal/jobs/{auto_job['id']}/result", json={
        "language": "ru", "duration_sec": 5.0, "segments": FINAL, "title": "Итоги квартала",
    })
    check("нетронутое название заменено", client.get(f"/api/sessions/{auto_id}").json()["title"]
          == "Итоги квартала")
    client.delete(f"/api/sessions/{auto_id}")

    print("\nредактирование")
    speaker = detail["speakers"][0]
    r = client.patch(f"/api/speakers/{speaker['id']}", json={"display_name": "Антон"})
    check("говорящий переименован", r.json()["display_name"] == "Антон", r.text[:150])
    check("подсказка имени сброшена", r.json()["suggested_name"] == "", r.json().get("suggested_name"))

    segment = detail["segments"][0]
    r = client.patch(f"/api/segments/{segment['id']}", json={"text": "Привет, начинаем планёрку."})
    check("текст исправлен", r.json()["text"] == "Привет, начинаем планёрку.", r.text[:150])
    check("тайминги слов сброшены", r.json()["words"] == [], r.json()["words"])
    check("помечен как правленый", r.json()["edited"] is True)

    other = detail["speakers"][1]
    r = client.patch(f"/api/segments/{segment['id']}", json={"speaker_id": other["id"]})
    check("говорящий переназначен", r.json()["speaker_id"] == other["id"], r.text[:150])
    r = client.patch(f"/api/segments/{segment['id']}", json={"speaker_id": speaker["id"]})
    check("возврат назад работает", r.json()["speaker_id"] == speaker["id"])

    r = client.patch(f"/api/segments/{segment['id']}", json={"speaker_id": 99999})
    check("чужой говорящий отвергнут", r.status_code == 400, r.status_code)

    print("\nстатистика")
    stats = client.get(f"/api/sessions/{live_id}/stats").json()
    check("два говорящих в статистике", len(stats["speakers"]) == 2, stats["speakers"])
    check("время речи посчитано", stats["speech_time_sec"] > 0, stats["speech_time_sec"])
    check("доли в сумме дают единицу",
          abs(sum(s["share"] for s in stats["speakers"]) - 1.0) < 0.01,
          [s["share"] for s in stats["speakers"]])
    check("имя подхватилось", any(s["display_name"] == "Антон" for s in stats["speakers"]))

    print("\nэкспорт")
    for fmt, probe in [
        ("txt", lambda b: "Антон" in b.decode("utf-8")),
        ("md", lambda b: b.decode("utf-8").startswith("#")),
        ("srt", lambda b: b.decode("utf-8").startswith("1\n") and "-->" in b.decode("utf-8")),
        ("vtt", lambda b: b.decode("utf-8").startswith("WEBVTT")),
        ("json", lambda b: b"\"segments\"" in b),
        ("docx", lambda b: b[:2] == b"PK"),
    ]:
        r = client.get(f"/api/sessions/{live_id}/export", params={"format": fmt})
        ok = r.status_code == 200 and probe(r.content)
        check(f"формат {fmt}", ok, f"{r.status_code} {r.content[:60]!r}")
        check(f"{fmt}: имя файла в UTF-8",
              "filename*=UTF-8''" in r.headers.get("content-disposition", ""),
              r.headers.get("content-disposition"))

    r = client.get(f"/api/sessions/{live_id}/export", params={"format": "pdf"})
    check("неизвестный формат отвергнут", r.status_code == 400, r.status_code)

    # SRT должен резать длинную реплику на несколько субтитров
    srt = client.get(f"/api/sessions/{live_id}/export", params={"format": "srt"}).text
    check("в SRT есть таймкоды с запятой", ",0" in srt or "," in srt.split("\n")[1], srt[:80])

    print("\nзагрузка файла")
    upload = WORK / "sample.wav"
    make_wav(upload, 2.0)
    with upload.open("rb") as fh:
        r = client.post("/api/uploads", files={"file": ("Совещание.wav", fh, "audio/wav")})
    check("файл принят", r.status_code == 201, r.text[:200])
    upload_id = r.json()["id"]
    check("имя из файла", r.json()["title"] == "Совещание", r.json()["title"])
    check("в очереди", r.json()["status"] == "queued", r.json()["status"])

    with (WORK / "bad.txt").open("wb") as fh:
        fh.write(b"not audio")
    with (WORK / "bad.txt").open("rb") as fh:
        r = client.post("/api/uploads", files={"file": ("bad.txt", fh, "text/plain")})
    check("чужой формат отвергнут", r.status_code == 415, r.status_code)

    r = client.get("/api/internal/jobs/next", params={"types": "batch"})
    check("batch-задание выдано", r.status_code == 200 and r.json()["type"] == "batch", r.text[:150])
    batch_job = r.json()
    client.post(f"/api/internal/jobs/{batch_job['id']}/fail", json={"error": "ffmpeg упал"})
    failed = client.get(f"/api/sessions/{upload_id}").json()
    check("статус failed", failed["status"] == "failed", failed["status"])
    check("ошибка видна", failed["error"] == "ffmpeg упал", failed["error"])

    print("\nсписок и удаление")
    rows = client.get("/api/sessions").json()
    check("в списке две записи", len(rows) == 2, len(rows))
    check("цвета говорящих отданы",
          any(len(row["speaker_colors"]) == 2 for row in rows),
          [row["speaker_colors"] for row in rows])

    r = client.get("/api/sessions/несуществующая")
    check("404 на чужой id", r.status_code == 404, r.status_code)

    r = client.delete(f"/api/sessions/{upload_id}")
    check("запись удалена", r.status_code == 204, r.status_code)
    check("файлы удалены", not (WORK / "data" / "audio" / upload_id).exists())
    check("в списке одна запись", len(client.get("/api/sessions").json()) == 1)

shutil.rmtree(WORK, ignore_errors=True)

print()
if failures:
    print(f"ПРОВАЛЕНО: {len(failures)}")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
print("Все проверки backend пройдены")
