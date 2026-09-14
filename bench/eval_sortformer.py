"""Прогон NVIDIA Streaming Sortformer по записям стенда.

Sortformer — сквозная потоковая модель диаризации (NVIDIA, 2025): звук
идёт порциями, говорящие держатся в кеше в порядке появления. Её советует
сам WhisperLiveKit вместо diart.

Два режима, сегменты [начало, конец, говорящий] складываются в кеш, а
считает их eval_live.py тем же кодом выравнивания, что и живой режим:

    lat0.32, lat1.04       — эталонный потоковый прогон NeMo (diarize());
    stream0.32, stream1.04 — наш класс живого режима
                             (worker-live/sortformer_stream.py): звук
                             подаётся кусками по 0,1 с, как в жизни.
                             Должен совпасть с эталоном — это проверка,
                             что живой режим повторяет замер.

Запуск в отдельном образе (NeMo несовместим с diart в worker-live):

    docker build -t scribe-bench-sortformer bench/sortformer
    docker run --rm --gpus all -v <bench>:/bench -v <data>:/data \\
        -v <models>:/models -v <worker-live>:/live -e HF_HOME=/models \\
        scribe-bench-sortformer python -u /bench/eval_sortformer.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
OUT = Path("/bench/cache/sortformer")
SR = 16000
PIECE = 0.1  # живой режим подаёт звук кусками примерно такой длины

sys.path.insert(0, "/live")  # код worker-live


def recordings():
    refs = json.loads(Path("/bench/cache/live_refs.json").read_text(encoding="utf-8"))
    for sid, meta in refs.items():
        yield sid, Path("/data") / meta["audio"]
    for wav in sorted(Path("/bench/data").glob("*.wav")):
        yield wav.stem, wav


def to_wav(src: Path, dst: Path) -> float:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                   check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(dst)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def read_wav(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def parse(segments) -> list[list]:
    """NeMo отдаёт сегменты строками вида «0.00 1.23 speaker_0»."""
    out = []
    for item in segments:
        if isinstance(item, str):
            parts = re.split(r"[,\s]+", item.strip())
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
        else:
            start, end, label = float(item[0]), float(item[1]), item[2]
        speaker = int(re.sub(r"\D", "", str(label)) or 0)
        out.append([start, end, speaker])
    return sorted(out)


def run_diarize(model, wav: Path) -> list[list]:
    import torch

    with torch.inference_mode():
        result = model.diarize(audio=[str(wav)], batch_size=1)
    return parse(result[0])


def run_stream(model, wav: Path) -> list[list]:
    from sortformer_stream import SortformerStream

    audio = read_wav(wav)
    stream = SortformerStream(model)
    piece = int(PIECE * SR)
    segments = []
    for i in range(0, len(audio), piece):
        stream.push(audio[i:i + piece])
        segments += stream.step()
    segments += stream.finish()
    return sorted([list(s) for s in segments])


def main() -> None:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from sortformer_stream import configure

    model = SortformerEncLabelModel.from_pretrained(MODEL)
    model.eval()
    model.to(torch.device("cuda"))
    print("потоковый режим:", getattr(model, "streaming_mode", "?"), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        wavs = {}
        for sid, path in recordings():
            wav = Path(tmp) / f"{sid}.wav"
            wavs[sid] = (wav, to_wav(path, wav))

        for latency in (0.32, 1.04, 10.0, 30.4):
            configure(model, latency)
            # Эталон NeMo нужен, чтобы проверить класс живого режима, — на
            # малых задержках это сделано; на больших хватает самого класса.
            kinds = (("lat", run_diarize), ("stream", run_stream)) if latency < 2 else (("stream", run_stream),)
            for kind, run in kinds:
                name = f"{kind}{latency}"
                (OUT / name).mkdir(parents=True, exist_ok=True)
                for sid, (wav, seconds) in wavs.items():
                    out = OUT / name / f"{sid}.json"
                    if out.exists():
                        continue  # уже посчитано — берём из кеша
                    started = time.time()
                    segments = run(model, wav)
                    elapsed = time.time() - started
                    out.write_text(json.dumps(
                        {"segments": segments, "seconds": seconds, "elapsed": elapsed}), encoding="utf-8")
                    speakers = sorted({s[2] for s in segments})
                    print(f"{name} {sid}: {len(segments)} сегментов, говорящие {speakers}, "
                          f"{seconds:.0f} с звука за {elapsed:.1f} с", flush=True)


if __name__ == "__main__":
    main()
