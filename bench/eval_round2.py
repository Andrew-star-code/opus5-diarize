"""Второй раунд замеров диаризации.

1. Слияние «крошечных» говорящих: доля речи мала, а эмбеддинг голоса
   близок к кому-то из остальных — значит это обрывок чужого голоса.
2. Какая часть сглаживания помогает, а какая мешает на реальных встречах.
3. Эксклюзивная или обычная разметка как источник меток для слов.

    docker compose run --rm --no-deps -v <bench>:/bench worker-batch \
        python -u /bench/eval_round2.py /bench/data
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/srv")
sys.path.insert(0, "/bench")

import decode  # noqa: E402
import diarize  # noqa: E402
import merge  # noqa: E402
from eval_diarization import STRICT, load_rttm, score, words_to_annotation  # noqa: E402

CACHE = Path("/bench/cache")
AUTO = {"min_speakers": 1, "max_speakers": 8}

# Сетка правила слияния показана целиком, чтобы видеть устойчивость,
# а не подгонять порог под три записи.
MERGE_GRID = [(share, sim) for share in (0.02, 0.05) for sim in (0.30, 0.40)]


def tiny_speaker_renames(output, max_share: float, min_sim: float) -> dict[str, str]:
    """Кого с кем слить. Индексы эмбеддингов идут по меткам обычной разметки."""
    regular = output.speaker_diarization
    emb = output.speaker_embeddings
    labels = regular.labels()
    if emb is None or len(labels) < 2 or len(emb) != len(labels):
        return {}
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = unit @ unit.T
    durations = [regular.label_duration(label) for label in labels]
    total = sum(durations) or 1.0

    renames: dict[str, str] = {}
    for i in sorted(range(len(labels)), key=lambda k: durations[k]):
        if durations[i] / total >= max_share:
            continue
        targets = [
            j for j in range(len(labels))
            if j != i and labels[j] not in renames
        ]
        if not targets:
            continue
        j = max(targets, key=lambda k: sim[i, k])
        if sim[i, j] >= min_sim:
            renames[labels[i]] = labels[j]
    return renames


def apply_renames(annotation, renames: dict[str, str]):
    if not renames:
        return annotation
    return annotation.rename_labels(mapping=renames).support()


def words_for(name: str, wav: Path) -> list[merge.Word]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.words.json"
    if path.exists():
        return [merge.Word(*w) for w in json.loads(path.read_text(encoding="utf-8"))]
    import asr

    words, _, _ = asr.transcribe(wav, language="en", duration=0)
    path.write_text(
        json.dumps([[w.w, w.s, w.e, w.p] for w in words], ensure_ascii=False),
        encoding="utf-8",
    )
    return words


def turns_of(annotation) -> list[merge.Turn]:
    return [
        merge.Turn(float(s.start), float(s.end), str(label))
        for s, _, label in annotation.itertracks(yield_label=True)
    ]


SMOOTHING = {
    "без сглаживания": {"smooth": False},
    "только короткие": {"smooth": True, "sandwich": 0.0},
    "полное (сейчас)": {"smooth": True, "sandwich": None},
}


def word_confusion(ref, words, turns, variant: dict) -> tuple[float, int]:
    original_smooth = merge._smooth
    original_sandwich = merge.SANDWICH_MAX_SEC
    try:
        if not variant["smooth"]:
            merge._smooth = lambda w, l: None
        if variant.get("sandwich") is not None:
            merge.SANDWICH_MAX_SEC = variant["sandwich"]
        labels = merge.assign_speakers(words, turns)
        segs = merge.build_segments(words, labels)
    finally:
        merge._smooth = original_smooth
        merge.SANDWICH_MAX_SEC = original_sandwich
    hyp = words_to_annotation(segs, ref.uri)
    return score(ref, hyp, **STRICT)["путаница"], len(hyp.labels())


def main() -> int:
    data = Path(sys.argv[1])
    pairs = sorted(p for p in data.glob("*.wav") if p.with_suffix(".rttm").exists())
    pipe = diarize.get_pipeline()

    turn_rows: dict[str, list[dict]] = {}
    word_rows: dict[str, list[float]] = {}

    def add(table, key, value):
        table.setdefault(key, []).append(value)

    for wav in pairs:
        ref = load_rttm(wav.with_suffix(".rttm"))
        true_count = len(ref.labels())
        print(f"\n=== {wav.stem} (говорящих по эталону: {true_count})", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            clean = decode.to_wav16k(wav, Path(tmp) / "a.wav")
            audio = diarize._load_waveform(clean)
            output = pipe(audio, **AUTO)
            oracle = pipe(audio, num_speakers=true_count)
            words = words_for(wav.stem, clean)

        exclusive = diarize._annotation(output)
        variants = {
            "как в сервисе": exclusive,
            "известное число": diarize._annotation(oracle),
        }
        for share, sim in MERGE_GRID:
            renames = tiny_speaker_renames(output, share, sim)
            variants[f"слияние {share:.0%}/{sim:.2f}"] = apply_renames(exclusive, renames)
            print(f"  правило {share:.0%}/{sim:.2f}: слито {renames or 'ничего'}")

        for name, hyp in variants.items():
            hyp.uri = ref.uri
            row = score(ref, hyp, **STRICT)
            row["найдено"] = len(hyp.labels())
            add(turn_rows, name, row)
            print(f"  {name:<18} DER {row['DER']:6.1%}  путаница {row['путаница']:5.1%}  "
                  f"говорящих {row['найдено']}/{true_count}", flush=True)

        chosen = apply_renames(exclusive, tiny_speaker_renames(output, 0.02, 0.30))
        sources = {
            "эксклюзивная": exclusive,
            "обычная": output.speaker_diarization,
            "эксклюзивная+слияние": chosen,
        }
        print("  --- метки на словах (путаница) ---")
        for source_name, annotation in sources.items():
            turns = turns_of(annotation)
            for smooth_name, variant in SMOOTHING.items():
                conf, found = word_confusion(ref, words, turns, variant)
                key = f"{source_name} / {smooth_name}"
                add(word_rows, key, conf)
                print(f"  {key:<40} {conf:5.2%}  говорящих {found}", flush=True)

    print("\n=== СРЕДНЕЕ: разметка по времени ===")
    for name, rows in turn_rows.items():
        der = sum(r["DER"] for r in rows) / len(rows)
        conf = sum(r["путаница"] for r in rows) / len(rows)
        counts = " ".join(str(r["найдено"]) for r in rows)
        print(f"  {name:<18} DER {der:6.2%}  путаница {conf:5.2%}  говорящих [{counts}]")
    print("\n=== СРЕДНЕЕ: метки на словах, путаница ===")
    for key, values in word_rows.items():
        print(f"  {key:<40} {sum(values) / len(values):5.2%}   по записям: "
              + " ".join(f"{v:5.2%}" for v in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
