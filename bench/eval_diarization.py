"""Замер качества диаризации на записях с эталонной разметкой.

Считает DER (diarization error rate) и её составляющие для нескольких
конфигураций пайплайна, чтобы решения о настройке опирались на цифры.

Запуск внутри worker-batch (там уже есть модели, pyannote и pyannote.metrics):

    docker compose run --rm --no-deps -v <каталог bench>:/bench \
        worker-batch python /bench/eval_diarization.py /bench/data [--with-asr]

В каталоге data лежат пары: <имя>.wav и <имя>.rttm.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/srv")  # модули самого воркера

import decode  # noqa: E402
import diarize  # noqa: E402
import merge  # noqa: E402
from pyannote.core import Annotation, Segment  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

# Строгая метрика, как в опубликованных замерах pyannote: без «воротника»
# вокруг границ и с учётом перекрывающейся речи. Мягкая — для сравнения.
STRICT = dict(collar=0.0, skip_overlap=False)
LENIENT = dict(collar=0.25, skip_overlap=False)


def load_rttm(path: Path) -> Annotation:
    ann = Annotation(uri=path.stem)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        ann[Segment(start, start + dur)] = spk
    return ann


def score(ref: Annotation, hyp: Annotation, **kw) -> dict:
    metric = DiarizationErrorRate(**kw)
    d = metric(ref, hyp, detailed=True)
    total = d["total"] or 1.0
    return {
        "DER": d["diarization error rate"],
        "пропуск": d["missed detection"] / total,
        "ложная": d["false alarm"] / total,
        "путаница": d["confusion"] / total,
    }


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict):
            out[key] = deep_update(out.get(key, {}), value)
        else:
            out[key] = value
    return out


def run_pipeline(audio: dict, *, params: dict, count: dict, exclusive: bool):
    pipe = diarize.get_pipeline()
    pipe.instantiate(params)
    output = pipe(audio, **count)
    if exclusive:
        return diarize._annotation(output)
    ann = getattr(output, "speaker_diarization", None)
    return ann if ann is not None else output


def words_to_annotation(segments: list[merge.Segment], uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    for s in segments:
        if s.speaker and s.end > s.start:
            ann[Segment(s.start, s.end)] = s.speaker
    return ann


def main() -> int:
    data = Path(sys.argv[1])
    with_asr = "--with-asr" in sys.argv
    pairs = sorted(p for p in data.glob("*.wav") if p.with_suffix(".rttm").exists())
    if not pairs:
        print(f"В {data} нет пар .wav + .rttm")
        return 1

    pipe = diarize.get_pipeline()
    base = copy.deepcopy(pipe.parameters(instantiated=True))
    print("Параметры по умолчанию:", json.dumps(base, ensure_ascii=False))

    auto = {"min_speakers": 1, "max_speakers": 8}  # так вызывает сервис
    configs: list[tuple[str, dict, str, bool]] = [
        ("как в сервисе", {}, "auto", True),
        ("обычная разметка", {}, "auto", False),
        ("известное число", {}, "oracle", True),
        ("порог 0.50", {"clustering": {"threshold": 0.50}}, "auto", True),
        ("порог 0.55", {"clustering": {"threshold": 0.55}}, "auto", True),
        ("порог 0.65", {"clustering": {"threshold": 0.65}}, "auto", True),
        ("порог 0.70", {"clustering": {"threshold": 0.70}}, "auto", True),
        ("min_off 0.25", {"segmentation": {"min_duration_off": 0.25}}, "auto", True),
    ]

    results: dict[str, list[dict]] = {name: [] for name, *_ in configs}
    asr_results: dict[str, list[dict]] = {}

    for wav in pairs:
        ref = load_rttm(wav.with_suffix(".rttm"))
        true_count = len(ref.labels())
        print(f"\n=== {wav.stem}: {ref.get_timeline().duration() / 60:.1f} мин речи, "
              f"говорящих по эталону: {true_count}")

        with tempfile.TemporaryDirectory() as tmp:
            clean = decode.to_wav16k(wav, Path(tmp) / "a.wav")
            audio = diarize._load_waveform(clean)

            default_hyp = None
            for name, over, count_mode, exclusive in configs:
                count = {"num_speakers": true_count} if count_mode == "oracle" else auto
                started = time.time()
                hyp = run_pipeline(
                    audio, params=deep_update(base, over), count=count, exclusive=exclusive
                )
                hyp.uri = ref.uri
                row = score(ref, hyp, **STRICT)
                row["DER мягкая"] = score(ref, hyp, **LENIENT)["DER"]
                row["найдено"] = len(hyp.labels())
                row["сек"] = time.time() - started
                results[name].append(row)
                print(f"  {name:<18} DER {row['DER']:6.1%}  (мягкая {row['DER мягкая']:6.1%})  "
                      f"путаница {row['путаница']:5.1%}  пропуск {row['пропуск']:5.1%}  "
                      f"ложная {row['ложная']:5.1%}  говорящих {row['найдено']}/{true_count}  "
                      f"{row['сек']:.0f} с")
                if name == "как в сервисе":
                    default_hyp = hyp
            pipe.instantiate(base)

            if with_asr and default_hyp is not None:
                # Влияние нашей постобработки: слияние со словами и сглаживание.
                import asr

                words, _, _ = asr.transcribe(clean, language="en", duration=0)
                turns = [
                    merge.Turn(float(s.start), float(s.end), str(label))
                    for s, _, label in default_hyp.itertracks(yield_label=True)
                ]
                original_smooth = merge._smooth
                for label, smooth in (("слова без сглаживания", lambda w, l: None),
                                      ("слова со сглаживанием", original_smooth)):
                    merge._smooth = smooth
                    labels = merge.assign_speakers(words, turns)
                    segs = merge.build_segments(words, labels)
                    hyp = words_to_annotation(segs, ref.uri)
                    row = score(ref, hyp, **STRICT)
                    asr_results.setdefault(label, []).append(row)
                    print(f"  {label:<22} путаница {row['путаница']:5.1%}  "
                          f"(DER {row['DER']:6.1%}: пропуск растёт из-за "
                          f"речи без слов)")
                merge._smooth = original_smooth

    print("\n=== СРЕДНЕЕ ПО ЗАПИСЯМ ===")
    for name, rows in results.items():
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in ("DER", "DER мягкая", "путаница")}
        print(f"  {name:<18} DER {avg['DER']:6.1%}  мягкая {avg['DER мягкая']:6.1%}  "
              f"путаница {avg['путаница']:5.1%}")
    for name, rows in asr_results.items():
        avg = sum(r["путаница"] for r in rows) / len(rows)
        print(f"  {name:<22} путаница {avg:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
