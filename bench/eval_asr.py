"""Распознавание русского: Whisper large-v3 (как в чистовике) против GigaAM-v3.

Наборы лежат в bench/data/asr:

    golos  — Golos farfield, тестовая часть: дальний микрофон, 1916 фраз.
             GigaAM учили в том числе на Golos, так что ему здесь проще —
             смотреть вместе с FLEURS.
    fleurs — FLEURS ru, тестовая часть: чтение фраз из Википедии; ни одна
             из моделей его не видела.

Whisper распознаёт кодом самого batch-воркера (asr.transcribe) — ровно
так, как в чистовике. GigaAM — в двух вариантах: v3_e2e_rnnt пишет с
пунктуацией и цифрами, v3_rnnt — без пунктуации, числа словами.

Метрика — доля ошибок в словах (WER) и в буквах (CER) после приведения к
одному виду: нижний регистр, ё → е, цифры словами, без пунктуации.

    docker build -t scribe-bench-asr bench/asr
    docker run --rm --gpus all -v <bench>:/bench -v <models>:/models \\
        -e HF_HOME=/models scribe-bench-asr python -u /bench/eval_asr.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/srv")  # код batch-воркера

DATA = Path("/bench/data/asr")
OUT = Path("/bench/cache/asr_results.json")
# Распознанное каждой моделью сохраняется сразу: сбой на подсчёте не должен
# стоить повторного прогона Whisper.
HYPS = Path("/bench/cache/asr_hyps")


# ------------------------------------------------------------------ наборы

def load_golos(tmp: Path, limit: int | None) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(DATA / "golos_farfield_test.parquet")
    items = []
    for i, row in enumerate(df.itertuples(index=False)):
        if limit and i >= limit:
            break
        audio = row.audio
        path = tmp / f"golos_{i}.wav"
        path.write_bytes(audio["bytes"])
        # У части фраз эталон пустой — pandas отдаёт его как NaN. Фраза
        # остаётся: всё, что модель там «услышит», засчитывается ей в ошибки.
        ref = row.transcription if isinstance(row.transcription, str) else ""
        items.append({"id": f"golos_{i}", "path": path, "ref": ref})
    return items


def load_fleurs(limit: int | None) -> list[dict]:
    # tsv без заголовка: id, файл, исходный текст, нормализованный текст, …
    items = []
    with open(DATA / "fleurs_ru_test.tsv", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            path = DATA / "test" / row[1]
            if path.exists():
                items.append({"id": row[1], "path": path, "ref": row[2]})
            if limit and len(items) >= limit:
                break
    return items


# ----------------------------------------------------------------- метрика

def _numbers_to_words(text: str) -> str:
    from num2words import num2words

    def spell(match: re.Match) -> str:
        try:
            return f" {num2words(match.group().replace(',', '.'), lang='ru')} "
        except Exception:
            return match.group()

    return re.sub(r"\d+(?:[.,]\d+)?", spell, text)


def normalize(text: str) -> str:
    text = text if isinstance(text, str) else ""
    text = _numbers_to_words(text.lower().replace("ё", "е"))
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def edits(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def score(items: list[dict], hyps: dict[str, str]) -> dict:
    w_err = w_tot = c_err = c_tot = 0
    for item in items:
        ref, hyp = normalize(item["ref"]), normalize(hyps[item["id"]])
        w_err += edits(ref.split(), hyp.split())
        w_tot += len(ref.split())
        c_err += edits(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        c_tot += len(ref.replace(" ", ""))
    return {"wer": w_err / max(1, w_tot), "cer": c_err / max(1, c_tot)}


# ------------------------------------------------------------------ модели

def run_whisper(items: list[dict]) -> dict[str, str]:
    import asr
    import soundfile as sf

    hyps = {}
    for item in items:
        words, segments, _ = asr.transcribe(
            item["path"], language="ru", duration=sf.info(str(item["path"])).duration
        )
        hyps[item["id"]] = " ".join(seg.text for seg in segments)
    asr._model = None  # освобождаем видеопамять под следующую модель
    return hyps


MAX_PIECE = 24.0  # GigaAM.transcribe принимает не длиннее 25 с


def _pieces(audio, sr: int) -> list:
    """Режет длинную фразу по самому тихому месту в её середине, пока куски не станут короче MAX_PIECE."""
    import numpy as np

    if len(audio) <= MAX_PIECE * sr:
        return [audio]
    frame = int(0.02 * sr)
    lo, hi = len(audio) // 4, 3 * len(audio) // 4
    energy = [float(np.square(audio[i:i + frame]).mean()) for i in range(lo, hi - frame, frame)]
    cut = lo + int(np.argmin(energy)) * frame + frame // 2
    return _pieces(audio[:cut], sr) + _pieces(audio[cut:], sr)


def run_gigaam(variant: str, items: list[dict]) -> dict[str, str]:
    import gigaam
    import soundfile as sf
    import torch

    try:
        model = gigaam.load_model(variant, download_root="/models/gigaam")
    except TypeError:
        model = gigaam.load_model(variant)

    def text_of(path: str) -> str:
        result = model.transcribe(path)
        return result if isinstance(result, str) else getattr(result, "text", str(result))

    hyps = {}
    with torch.inference_mode(), tempfile.TemporaryDirectory() as tmp:
        for item in items:
            audio, sr = sf.read(str(item["path"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) <= MAX_PIECE * sr:
                hyps[item["id"]] = text_of(str(item["path"]))
                continue
            texts = []
            for k, piece in enumerate(_pieces(audio, sr)):
                path = Path(tmp) / f"piece_{k}.wav"
                sf.write(str(path), piece, sr)
                texts.append(text_of(str(path)))
            hyps[item["id"]] = " ".join(texts)
    del model
    torch.cuda.empty_cache()
    return hyps


MODELS = {
    "whisper-large-v3": run_whisper,
    "gigaam-v3-e2e-rnnt": lambda items: run_gigaam("v3_e2e_rnnt", items),
    "gigaam-v3-rnnt": lambda items: run_gigaam("v3_rnnt", items),
}


def main() -> None:
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="фраз из каждого набора")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        sets = {"golos": load_golos(Path(tmp), args.limit), "fleurs": load_fleurs(args.limit)}
        seconds = {name: sum(sf.info(str(i["path"])).duration for i in items)
                   for name, items in sets.items()}
        for name, items in sets.items():
            print(f"{name}: {len(items)} фраз, {seconds[name] / 60:.0f} мин звука", flush=True)

        results = {}
        for model_name in args.models:
            for set_name, items in sets.items():
                cache = HYPS / f"{model_name}_{set_name}_{args.limit or 'all'}.json"
                if cache.exists():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    hyps, elapsed = saved["hyps"], saved["elapsed"]
                else:
                    started = time.time()
                    hyps = MODELS[model_name](items)
                    elapsed = time.time() - started
                    HYPS.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"hyps": hyps, "elapsed": elapsed}, ensure_ascii=False),
                                     encoding="utf-8")
                gc.collect()
                s = score(items, hyps)
                s["rtf"] = elapsed / seconds[set_name]
                results[f"{model_name}/{set_name}"] = {**s, "hyps": hyps}
                print(f"{model_name:20} {set_name:7} WER {s['wer'] * 100:5.1f}%  "
                      f"CER {s['cer'] * 100:5.1f}%  скорость {1 / s['rtf']:5.0f}× реального времени",
                      flush=True)

        refs = {i["id"]: i["ref"] for items in sets.values() for i in items}
        OUT.write_text(json.dumps({"refs": refs, "results": results}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\nподробности по фразам: {OUT}")


if __name__ == "__main__":
    main()
