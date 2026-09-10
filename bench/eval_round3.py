"""Третий раунд: проверка того, что уже встроено в сервис.

Считает по production-коду воркера (diarize.diarize и merge.assign_speakers),
а не по копии логики в стенде: так проверяется ровно то, что уйдёт
пользователю. Слова берутся из кеша второго раунда.

    docker compose run --rm --no-deps -v <bench>:/bench worker-batch \
        python -u /bench/eval_round3.py /bench/data
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/srv")
sys.path.insert(0, "/bench")

import decode  # noqa: E402
import diarize  # noqa: E402
import merge  # noqa: E402
from config import settings  # noqa: E402
from eval_diarization import STRICT, load_rttm, score, words_to_annotation  # noqa: E402
from pyannote.core import Annotation, Segment  # noqa: E402


def turns_to_annotation(turns, uri):
    ann = Annotation(uri=uri)
    for t in turns:
        ann[Segment(t.start, t.end)] = t.label
    return ann


def main() -> int:
    data = Path(sys.argv[1])
    pairs = sorted(p for p in data.glob("*.wav") if p.with_suffix(".rttm").exists())
    default_share = settings.tiny_speaker_max_share
    print(f"Настройки сервиса: слияние {settings.tiny_speaker_max_share:.0%} / "
          f"{settings.tiny_speaker_min_similarity:.2f}, "
          f"сглаживание {'вкл' if settings.smooth_speaker_turns else 'выкл'}")

    combos = [
        ("было: без слияния, со сглаживанием", False, True),
        ("только выключено сглаживание", False, False),
        ("только слияние", True, True),
        ("стало: как в сервисе", True, settings.smooth_speaker_turns),
    ]
    totals: dict[str, list[tuple[float, float, int]]] = {c[0]: [] for c in combos}

    for wav in pairs:
        ref = load_rttm(wav.with_suffix(".rttm"))
        true_count = len(ref.labels())
        words = [
            merge.Word(*w)
            for w in json.loads((Path("/bench/cache") / f"{wav.stem}.words.json")
                                .read_text(encoding="utf-8"))
        ]
        print(f"\n=== {wav.stem} (говорящих по эталону: {true_count}, слов {len(words)})",
              flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            clean = decode.to_wav16k(wav, Path(tmp) / "a.wav")
            settings.tiny_speaker_max_share = 0.0
            turns_plain = diarize.diarize(clean, num_speakers=0, max_speakers=8)
            settings.tiny_speaker_max_share = default_share
            turns_merged = diarize.diarize(clean, num_speakers=0, max_speakers=8)

        for name, use_merge, smooth in combos:
            turns = turns_merged if use_merge else turns_plain
            turn_der = score(ref, turns_to_annotation(turns, ref.uri), **STRICT)["DER"]
            labels = merge.assign_speakers(words, turns, smooth=smooth)
            hyp = words_to_annotation(merge.build_segments(words, labels), ref.uri)
            conf = score(ref, hyp, **STRICT)["путаница"]
            found = len(hyp.labels())
            totals[name].append((conf, turn_der, found))
            print(f"  {name:<38} путаница слов {conf:5.2%}  DER разметки {turn_der:6.2%}  "
                  f"говорящих {found}/{true_count}", flush=True)

    print("\n=== СРЕДНЕЕ ===")
    for name, rows in totals.items():
        conf = sum(r[0] for r in rows) / len(rows)
        der = sum(r[1] for r in rows) / len(rows)
        counts = " ".join(str(r[2]) for r in rows)
        print(f"  {name:<38} путаница слов {conf:5.2%}  DER {der:6.2%}  говорящих [{counts}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
