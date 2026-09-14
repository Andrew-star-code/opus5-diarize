"""Поправка стыков реплик языковой моделью — замер на встречах AMI.

Эталон — разметка людей (RTTM из pyannote/AMI-diarization-setup, как в
остальном стенде). Слова и метки берутся из batch-пайплайна: распознавание,
разметка, раздача слов говорящим — ровно то, что попадает в чистовик.
Эталонная метка слова — тем же merge.assign_speakers по отрезкам RTTM.

Дальше поправка стыков (polish.adjust_boundaries) с каждой из моделей и
сравнение с чистовиком без неё:

    слов под чужим именем — после подбора соответствия меток;
    начало ушло предыдущему — на скольких сменах говорящего по эталону
        первые слова новой реплики достались прежнему говорящему;
    конец ушёл новому — наоборот;
    сдвинуто / спрошено — сколько границ модель передвинула из заданных;
    время — сколько заняла поправка (загрузка модели не входит).

Встречи английские, а инструкция модели русская — это замер подхода, а не
русского языка.

    docker compose run --rm --no-deps -v "$(pwd)/bench:/bench" worker-batch python -u /bench/eval_llm.py cache
    docker compose run --rm -v "$(pwd)/bench:/bench" worker-batch python -u /bench/eval_llm.py eval qwen3:8b
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/srv")  # код batch-воркера

import asr  # noqa: E402
import decode  # noqa: E402
import diarize  # noqa: E402
import merge  # noqa: E402
import polish  # noqa: E402

DATA = Path("/bench/data")
CACHE = Path("/bench/cache/llm")


def load_rttm(path: Path) -> list[merge.Turn]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            turns.append(merge.Turn(start=start, end=start + dur, label=parts[7]))
    return turns


def build_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for wav in sorted(DATA.glob("*.wav")):
        rttm = wav.with_suffix(".rttm")
        out = CACHE / f"{wav.stem}.json"
        if not rttm.exists():
            continue
        if out.exists():
            print(f"{wav.stem}: уже в кеше", flush=True)
            continue
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            wav16 = decode.to_wav16k(wav, Path(tmp) / "audio.wav")
            duration = decode.duration_of(wav16)
            words, _, _ = asr.transcribe(wav16, language="en", duration=duration)
            turns = diarize.diarize(wav16, num_speakers=0, max_speakers=8,
                                    on_progress=lambda fraction: None)
        labels = merge.assign_speakers(words, turns)
        ref = merge.assign_speakers(words, load_rttm(rttm))
        out.write_text(json.dumps({
            "words": [[w.w, w.s, w.e] for w in words], "labels": labels, "ref": ref,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"{wav.stem}: {len(words)} слов, {time.time() - started:.0f} с", flush=True)


def score(labels: list, ref: list) -> dict:
    from scipy.optimize import linear_sum_assignment

    pairs = [(h, r) for h, r in zip(labels, ref) if h is not None and r is not None]
    hyp_labels = sorted({h for h, _ in pairs})
    ref_labels = sorted({r for _, r in pairs})
    counts = np.zeros((len(hyp_labels), len(ref_labels)))
    for h, r in pairs:
        counts[hyp_labels.index(h), ref_labels.index(r)] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping = {hyp_labels[i]: ref_labels[j] for i, j in zip(rows, cols)}
    wrong = sum(1 for h, r in pairs if mapping.get(h) != r)

    seq = [(mapping.get(h) if h is not None else None, r) for h, r in zip(labels, ref) if r is not None]
    changes = late = early = 0
    for i in range(1, len(seq)):
        if seq[i][1] == seq[i - 1][1]:
            continue
        changes += 1
        k = 0
        while i + k < len(seq) and k < 5 and seq[i + k][1] == seq[i][1] and seq[i + k][0] == seq[i - 1][1]:
            k += 1
        late += bool(k)
        k = 0
        while i - 1 - k >= 0 and k < 5 and seq[i - 1 - k][1] == seq[i - 1][1] and seq[i - 1 - k][0] == seq[i][1]:
            k += 1
        early += bool(k)
    return {"words": len(pairs), "wrong": wrong, "changes": changes, "late": late, "early": early}


def run_eval(models: list[str], *, max_shift: int, ask_all: bool,
             pauses: bool = False, sentence_end: bool = False) -> None:
    from llm import LLM

    files = sorted(CACHE.glob("*.json"))
    if not files:
        raise SystemExit("кеш пуст — сначала eval_llm.py cache")
    data = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}

    print(f"{'модель':48} │ {'под чужим именем':>16} │ {'начало ушло':>11} {'конец ушёл':>10} │ "
          f"{'сдвинуто/спрошено':>17} {'время':>7} │ по встречам")
    for model in [None, *models]:
        client = LLM(model=model) if model else None
        if client is not None:
            if not client.available():
                print(f"{model:48} │ не скачана — пропуск")
                continue
            # Загрузка модели в видеопамять — не часть поправки.
            client.ask("Ответь числом.", "1+1", {"type": "object", "properties": {"n": {"type": "integer"}}})
        total = {"words": 0, "wrong": 0, "changes": 0, "late": 0, "early": 0,
                 "asked": 0, "moved": 0, "elapsed": 0.0}
        cells = []
        for name, d in data.items():
            words = [merge.Word(w=w, s=s, e=e) for w, s, e in d["words"]]
            labels = d["labels"]
            started = time.time()
            if client is not None:
                labels, stats = polish.adjust_boundaries(
                    words, labels, client.ask, max_shift=max_shift, skip_confident=not ask_all,
                    show_pauses=pauses, sentence_end_only=sentence_end,
                )
                total["asked"] += stats["asked"]
                total["moved"] += stats["moved"]
            total["elapsed"] += time.time() - started
            s = score(labels, d["ref"])
            for key in ("words", "wrong", "changes", "late", "early"):
                total[key] += s[key]
            cells.append(f"{name} {s['wrong'] / max(1, s['words']) * 100:5.2f}%")
        if client is not None:
            client.close()
        n = max(1, total["changes"])
        print(f"{model or 'без модели':48} │ {total['wrong'] / max(1, total['words']) * 100:15.2f}% │ "
              f"{total['late'] / n * 100:10.1f}% {total['early'] / n * 100:9.1f}% │ "
              f"{total['moved']:8d}/{total['asked']:<8d} {total['elapsed']:6.0f}с │ " + "  ".join(cells),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["cache", "eval"])
    parser.add_argument("models", nargs="*")
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="спрашивать и уверенные стыки")
    parser.add_argument("--pauses", action="store_true", help="показывать модели паузы")
    parser.add_argument("--sentence-end", action="store_true",
                        help="граница только сразу после конца предложения")
    args = parser.parse_args()
    if args.what == "cache":
        build_cache()
    else:
        run_eval(args.models, max_shift=args.max_shift, ask_all=args.all,
                 pauses=args.pauses, sentence_end=args.sentence_end)


if __name__ == "__main__":
    main()
