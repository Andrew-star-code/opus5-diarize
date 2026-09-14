"""Замер живой диаризации (diart) на записях с опорной разметкой.

Живой режим размечает говорящих потоково: окно 5 с, шаг 0,5 с, решение
принимается сразу и больше не пересматривается. Здесь меряется, сколько
слов живой черновик отдаёт не тому говорящему, при разных настройках diart
и разных способах собирать его разметку в сегменты.

Опорная разметка — чистовик batch-пайплайна (pyannote community-1,
офлайн). Живая диаризация не может быть лучше офлайновой, так что близость
к чистовику — честная мера того, насколько живой режим отстаёт.

Считается в два этапа, потому что дорогая часть от настроек не зависит:

    cache  — видеокарта, один раз на запись: сегментация и эмбеддинги
             каждого окна производственными моделями живого воркера;
    eval   — процессор, параллельно: кластеризация, склейка окон и порог
             diart с нужными настройками, затем раздача слов говорящим
             кодом выравнивания WhisperLiveKit — как в живом черновике.

Запуск в образе worker-live (нужны его модели и библиотеки):

    docker compose run --rm --no-deps -v <каталог bench>:/bench \\
        worker-live python -u /bench/eval_live.py cache
    docker compose run --rm --no-deps -v <каталог bench>:/bench \\
        worker-live python -u /bench/eval_live.py eval variants

Опорная разметка — /bench/cache/live_refs.json: для каждой сессии путь к
звуку относительно /data и слова [начало, конец, говорящий].
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CACHE = Path("/bench/cache/live")
REFS = Path("/bench/cache/live_refs.json")
SR = 16000

# У этих записей сам чистовик ненадёжен (сцена из фильма через колонки:
# почти всё ушло одному говорящему) — показываем, но в среднее не берём.
UNTRUSTED = {"aecbc57332104826"}

DEFAULTS = dict(tau=0.6, rho=0.3, delta=1.0)  # значения diart по умолчанию

# Встречи AMI (bench/fetch_ami.sh): эталон размечен людьми, а не нашим же
# чистовиком. Слов в эталоне нет — меряется DER по времени.
AMI = Path("/bench/data")
SORTFORMER = Path("/bench/cache/sortformer")


def ami_refs() -> dict:
    return {
        wav.stem: {"audio": str(wav), "rttm": str(wav.with_suffix(".rttm")), "words": []}
        for wav in sorted(AMI.glob("*.wav")) if wav.with_suffix(".rttm").exists()
    }


def load_rttm(path: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            ann[Segment(start, start + dur)] = parts[7]
    return ann


# --------------------------------------------------------------- этап 1

def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build_cache(refs: dict) -> None:
    sys.path.insert(0, "/srv")
    import engine
    import torch
    from diart import SpeakerDiarization

    with engine._trusted_checkpoints():
        shared = engine._SharedDiartModels(sample_rate=SR)
    config = shared.new_config()
    pipeline = SpeakerDiarization(config)
    win = int(round(config.duration * SR))
    hop = int(round(config.step * SR))
    CACHE.mkdir(parents=True, exist_ok=True)

    for sid, meta in refs.items():
        out = CACHE / f"{sid}.npz"
        if out.exists():
            print(f"{sid}: уже в кеше")
            continue
        started = time.time()
        audio = decode(Path("/data") / meta["audio"])
        # Окна ровно как у rearrange_audio_stream: первое — [0, 5], дальше
        # со сдвигом на шаг; неполный хвост diart тоже не разбирает.
        starts = list(range(0, len(audio) - win + 1, hop))
        segs, embs = [], []
        with torch.inference_mode():
            for i in range(0, len(starts), 32):
                batch = torch.stack([
                    torch.from_numpy(audio[s:s + win].reshape(-1, 1)) for s in starts[i:i + 32]
                ])
                seg = pipeline.segmentation(batch)
                emb = pipeline.embedding(batch, seg)
                segs.append(seg.cpu().numpy())
                embs.append(emb.cpu().numpy())
        np.savez(
            out,
            seg=np.concatenate(segs), emb=np.concatenate(embs),
            starts=np.array(starts, dtype=np.float64) / SR,
            duration=config.duration, step=config.step,
            max_speakers=config.max_speakers,
        )
        print(f"{sid}: {len(starts)} окон, {len(audio) / SR:.0f} с звука, "
              f"{time.time() - started:.0f} с", flush=True)


# --------------------------------------------------------------- этап 2

class _ExclusiveBinarize:
    """Binarize из diart, но в каждом кадре — один говорящий, самый уверенный.

    Метки и отсчёт времени те же, что у diart.blocks.Binarize: реплика
    начинается в середине первого кадра и кончается в середине кадра,
    где говорящий сменился или замолчал.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, segmentation):
        from pyannote.core import Annotation, Segment

        data = segmentation.data
        frames = segmentation.sliding_window
        best = data.argmax(axis=1)
        active = data[np.arange(len(data)), best] > self.threshold
        labels = np.where(active, best, -1)
        annotation = Annotation(modality="speech")
        current, start = -1, 0
        for t in range(len(labels) + 1):
            label = labels[t] if t < len(labels) else -1
            if label != current:
                if current >= 0:
                    region = Segment(frames[start].middle, frames[t].middle)
                    annotation[region, int(current)] = f"speaker{current}"
                current, start = label, t
        return annotation


def _collect_fixed(annotation, segments: list) -> None:
    """Сегменты шага по времени, без выдуманных пауз, со склейкой соседних."""
    for segment, _track, label in sorted(
        annotation.itertracks(yield_label=True), key=lambda item: item[0].start
    ):
        speaker = int(str(label).removeprefix("speaker"))
        if segments and segments[-1][2] == speaker and segment.start <= segments[-1][1] + 0.05:
            segments[-1][1] = max(segments[-1][1], segment.end)
        else:
            segments.append([segment.start, segment.end, speaker])


_FILES: dict = {}


def _load_files(names: list[str], refs: dict) -> None:
    import torch

    torch.set_num_threads(1)
    for sid in names:
        data = np.load(CACHE / f"{sid}.npz")
        _FILES[sid] = {
            "seg": data["seg"], "emb": torch.from_numpy(data["emb"]),
            "starts": data["starts"], "duration": float(data["duration"]),
            "step": float(data["step"]), "max_speakers": int(data["max_speakers"]),
            "words": refs[sid].get("words", []),
            "ref": load_rttm(refs[sid]["rttm"]) if refs[sid].get("rttm") else None,
        }


def simulate(sid: str, *, latency: float, tau: float, rho: float, delta: float,
             mode: str, variant: str | None = None, shift: float = 0.0) -> list:
    """Прогоняет кластеризацию и склейку diart по кешу; сегменты как в живом режиме.

    mode="sortformer" — готовые сегменты Streaming Sortformer из кеша
    eval_sortformer.py (variant — настройка задержки).
    """
    if mode == "sortformer":
        path = SORTFORMER / variant / f"{sid}.json"
        if not path.exists():
            raise SystemExit(f"нет сегментов Sortformer: {path} — сначала eval_sortformer.py")
        # shift — сдвиг разметки раньше на столько секунд: проверка, не
        # запаздывает ли модель относительно звука.
        segs = json.loads(path.read_text(encoding="utf-8"))["segments"]
        return [[max(0.0, s - shift), e - shift, spk] for s, e, spk in segs if e - shift > max(0.0, s - shift)]
    from diart.blocks import Binarize, DelayedAggregation, OnlineSpeakerClustering
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    f = _FILES[sid]
    clustering = OnlineSpeakerClustering(tau, rho, delta, "cosine", f["max_speakers"])
    aggregation = DelayedAggregation(f["step"], latency, strategy="hamming", cropping_mode="loose")
    binarize = _ExclusiveBinarize(tau) if mode == "exclusive" else Binarize(tau)

    observer = None
    if mode == "wlk":
        from whisperlivekit.diarization import diart_backend
        diart_backend.print = lambda *a, **k: None  # наблюдатель печатает каждый сегмент
        observer = diart_backend.DiarizationObserver()

    resolution = f["duration"] / f["seg"].shape[1]
    buffer, segments = [], []
    for k, start in enumerate(f["starts"]):
        window = SlidingWindowFeature(
            f["seg"][k], SlidingWindow(start=float(start), duration=resolution, step=resolution)
        )
        buffer.append(clustering(window, f["emb"][k]))
        annotation = binarize(aggregation(buffer))
        if observer is not None:
            # В живом конвейере до наблюдателя разметку видит накопитель
            # diart: его update() зовёт labels(), и pyannote только тогда
            # заполняет _labels, которое читает наблюдатель WhisperLiveKit.
            annotation.labels()
            observer.on_next((annotation, _AudioStub(float(start), f["duration"])))
        else:
            _collect_fixed(annotation, segments)
        if len(buffer) == aggregation.num_overlapping_windows:
            buffer = buffer[1:]

    if observer is not None:
        segments = [
            [s.start, s.end, int(str(s.speaker).removeprefix("speaker"))]
            for s in observer.diarization_segments
        ]
    return segments


class _AudioStub:
    """То, что наблюдатель WhisperLiveKit читает у звука: только границы."""

    class _Extent:
        def __init__(self, start, end):
            self.start, self.end = start, end

    class _Data:
        shape = (0,)

    def __init__(self, start: float, duration: float):
        self.extent = self._Extent(start, start + duration)
        self.data = self._Data()


_SENTENCE_END = (".", "!", "?", "…")


def _ends_sentence(word: list) -> bool:
    text = word[3] if len(word) > 3 else ""
    return text.rstrip("»\"')").endswith(_SENTENCE_END)


def snap_to_sentence(hyp: list, words: list, max_shift: int) -> list:
    """Граница говорящих посреди предложения переносится к его концу.

    diart ставит границу по звуку и часто промахивается на слово-два:
    «…Шептать. Надо» / «всегда находить…». Если в пределах max_shift слов
    от границы есть конец предложения, граница переезжает туда. Переносим
    только через слова одного говорящего — соседнюю границу не задеваем.
    """
    out = list(hyp)
    n = len(out)
    i = 1
    while i < n:
        if out[i] != out[i - 1] and not _ends_sentence(words[i - 1]):
            b = i - 1  # граница — после слова b
            best = None
            for d in range(1, max_shift + 1):
                for j in (b - d, b + d):
                    if not (0 <= j < n - 1) or not _ends_sentence(words[j]):
                        continue
                    lo, hi, want = (j + 1, b, out[b]) if j < b else (i, j, out[i])
                    if all(x == want for x in out[lo:hi + 1]):
                        best = j
                        break
                if best is not None:
                    break
            if best is not None:
                if best < b:
                    for k in range(best + 1, b + 1):
                        out[k] = out[i]
                else:
                    for k in range(i, best + 1):
                        out[k] = out[b]
                    i = best
        i += 1
    return out


def establish_speakers(hyp: list, min_words: int) -> list:
    """Новый говорящий появляется, только когда наговорил min_words слов.

    Sortformer иногда заводит лишнего говорящего на разговоре вдвоём —
    на пару фраз. Строго онлайн: пока у метки меньше min_words слов с
    начала записи, её слова остаются у последнего настоящего говорящего.
    Первый говорящий записи настоящий сразу.
    """
    out = list(hyp)
    counts, visible, last = {}, set(), None
    for i, label in enumerate(hyp):
        counts[label] = counts.get(label, 0) + 1
        if last is None or counts[label] >= min_words:
            visible.add(label)
        if label in visible:
            last = label
        else:
            out[i] = last
    return out


def smooth_labels(hyp: list, words: list, *, min_words: int = 0, min_sec: float = 0.0,
                  gap: float = 0.0, snap: int = 0, establish: int = 0,
                  lookahead: bool = False) -> list:
    """Реплика вместо слова: короткий обрывок не может сменить говорящего.

    Смотрит только назад, поэтому годится и для живого режима: обрывок
    остаётся у предыдущего говорящего. gap — смена говорящего допустима
    только на паузе не короче gap секунд.
    """
    out = establish_speakers(hyp, establish) if establish else list(hyp)
    if gap > 0:
        for i in range(1, len(out)):
            if out[i] != out[i - 1] and words[i][0] - words[i - 1][1] < gap:
                out[i] = out[i - 1]
    if snap:
        out = snap_to_sentence(out, words, snap)
    if (min_words or min_sec) and lookahead:
        return _absorb_lookahead(out, words, min_words, min_sec)
    if min_words or min_sec:
        src = list(out)
        i = 0
        while i < len(src):
            j = i
            while j + 1 < len(src) and src[j + 1] == src[i]:
                j += 1
            span = words[j][1] - words[i][0]
            if i > 0 and (j - i + 1 < min_words or span < min_sec):
                for k in range(i, j + 1):
                    out[k] = out[i - 1]
            i = j + 1
    return out


def _absorb_lookahead(labels: list, words: list, min_words: int, min_sec: float) -> list:
    """Короткие серии — но с оглядкой на следующую длинную.

    Метка на стыке дрожит: «А… | Б Б | А | Б Б Б Б…». Без заглядывания
    каждый короткий кусок уходит к предыдущему, и начало реплики Б целиком
    достаётся А. Здесь группа коротких кусков между длинными А и Б делится
    там, где впервые появился Б: всё с этого места — начало его реплики.
    Если продолжения ещё нет (край живого черновика) — к предыдущему, как
    раньше.
    """
    runs, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        runs.append((i, j, labels[i]))
        i = j + 1

    def short(run) -> bool:
        a, b, _ = run
        return (b - a + 1 < min_words) or (words[b][1] - words[a][0] < min_sec)

    out = list(labels)
    k, prev = 0, None
    while k < len(runs):
        if prev is None or not short(runs[k]):
            prev = runs[k][2]
            k += 1
            continue
        g = k
        while g < len(runs) and short(runs[g]):
            g += 1
        nxt = runs[g][2] if g < len(runs) else None
        split = g
        if nxt is not None and nxt != prev:
            split = next((m for m in range(k, g) if runs[m][2] == nxt), g)
        for m in range(k, g):
            label = prev if m < split else nxt
            for t in range(runs[m][0], runs[m][1] + 1):
                out[t] = label
        k = g
    return out


def runs_of(labels: list) -> list[int]:
    """Длины подряд идущих отрезков одного говорящего, в словах."""
    runs, n = [], 0
    for i, label in enumerate(labels):
        if i and label != labels[i - 1]:
            runs.append(n)
            n = 0
        n += 1
    if n:
        runs.append(n)
    return runs


def word_errors(sid: str, segments: list, smooth: dict | None = None) -> dict:
    """Раздаёт слова говорящим выравниванием WhisperLiveKit и сверяет с чистовиком."""
    from scipy.optimize import linear_sum_assignment
    from whisperlivekit.timed_objects import ASRToken, SpeakerSegment
    from whisperlivekit.tokens_alignment import TokensAlignment

    words = _FILES[sid]["words"]
    alignment = TokensAlignment.__new__(TokensAlignment)
    alignment.all_diarization_segments = [
        SpeakerSegment(start=s, end=e, speaker=spk) for s, e, spk in segments
    ]
    merged = alignment.concatenate_diar_segments()
    hyp = []
    if merged:
        cursor = 0
        for s, e, *_ in words:
            speaker, cursor = alignment._speaker_for_token(
                ASRToken(start=s, end=e, text=" w"), merged, cursor
            )
            hyp.append(speaker)
    else:
        hyp = [1] * len(words)
    if smooth:
        hyp = smooth_labels(hyp, words, **smooth)

    ref_labels = sorted({w[2] for w in words})
    hyp_labels = sorted(set(hyp))
    counts = np.zeros((len(hyp_labels), len(ref_labels)))
    for h, (_, _, r, *_) in zip(hyp, words):
        counts[hyp_labels.index(h), ref_labels.index(r)] += 1
    rows, cols = linear_sum_assignment(-counts)
    correct = counts[rows, cols].sum()
    shares = counts.sum(axis=1) / max(1, len(words))
    # Запаздывание смены говорящего: на каждой настоящей смене — сколько
    # первых слов новой реплики (до пяти) досталось предыдущему говорящему
    # и сколько последних слов прежней реплики ушло новому.
    mapping = {hyp_labels[r]: ref_labels[c] for r, c in zip(rows, cols)}
    mapped = [mapping.get(x) for x in hyp]
    ref_seq = [w[2] for w in words]
    changes = late = late_words = early = early_words = 0
    for i in range(1, len(ref_seq)):
        if ref_seq[i] == ref_seq[i - 1]:
            continue
        changes += 1
        k = 0
        while i + k < len(ref_seq) and k < 5 and ref_seq[i + k] == ref_seq[i] and mapped[i + k] == ref_seq[i - 1]:
            k += 1
        late += bool(k)
        late_words += k
        k = 0
        while i - 1 - k >= 0 and k < 5 and ref_seq[i - 1 - k] == ref_seq[i - 1] and mapped[i - 1 - k] == ref_seq[i]:
            k += 1
        early += bool(k)
        early_words += k

    hyp_runs = runs_of(hyp)
    return {
        "changes": changes, "late": late, "late_words": late_words,
        "early": early, "early_words": early_words,
        "err": 1 - correct / max(1, len(words)),
        "spk": int((shares >= 0.02).sum()),
        "ref_spk": len(ref_labels),
        # Дробление — то, что видно глазу: сколько реплик в черновике
        # против чистовика и сколько из них обрывки короче трёх слов.
        "turns": len(hyp_runs),
        "ref_turns": len(runs_of([w[2] for w in words])),
        "scraps": sum(1 for n in hyp_runs if n < 3),
    }


def ami_score(sid: str, segments: list) -> dict:
    """Строгий DER, как в опубликованных замерах: без воротника, с перекрытиями."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    hyp = Annotation()
    for start, end, speaker in segments:
        hyp[Segment(start, end)] = f"spk{speaker}"
    ref = _FILES[sid]["ref"]
    d = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, detailed=True)
    total = d["total"] or 1.0
    return {"der": d["diarization error rate"], "conf": d["confusion"] / total,
            "spk": len(hyp.labels()), "ref_spk": len(ref.labels())}


def evaluate(config: dict) -> dict:
    result = {"config": config, "files": {}}
    params = {k: v for k, v in config.items() if k != "smooth"}
    for sid in _FILES:
        segments = simulate(sid, **params)
        if _FILES[sid]["ref"] is not None:
            result["files"][sid] = ami_score(sid, segments)
        else:
            result["files"][sid] = word_errors(sid, segments, config.get("smooth"))
    ami = [v for v in result["files"].values() if "der" in v]
    result["der"] = float(np.mean([v["der"] for v in ami])) if ami else float("nan")
    result["conf"] = float(np.mean([v["conf"] for v in ami])) if ami else float("nan")
    trusted = {k: v for k, v in result["files"].items() if k not in UNTRUSTED and "der" not in v}
    result["mean"] = float(np.mean([v["err"] for v in trusted.values()]))
    result["worst"] = float(np.max([v["err"] for v in trusted.values()]))
    result["turns"] = sum(v["turns"] for v in trusted.values())
    result["ref_turns"] = sum(v["ref_turns"] for v in trusted.values())
    result["scraps"] = sum(v["scraps"] for v in trusted.values())
    for key in ("changes", "late", "late_words", "early", "early_words"):
        result[key] = sum(v.get(key, 0) for v in trusted.values())
    return result


def configs_for(stage: str, args) -> list[dict]:
    if stage == "variants":
        return [
            dict(mode=mode, latency=lat, **DEFAULTS)
            for mode in ("wlk", "fixed", "exclusive")
            for lat in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
        ]
    if stage == "grid":
        return [
            dict(mode=args.mode, latency=lat, tau=tau, rho=rho, delta=delta)
            for lat in args.latency
            for tau, rho, delta in itertools.product(args.taus, args.rhos, args.deltas)
        ]
    if stage == "smooth":
        best = dict(mode="fixed", tau=0.6, rho=0.1, delta=1.0)
        variants = [None,
                    dict(min_words=2), dict(min_words=3), dict(min_words=4), dict(min_words=6),
                    dict(min_sec=0.5), dict(min_sec=1.0), dict(min_sec=1.5), dict(min_sec=2.0),
                    dict(gap=0.1), dict(gap=0.2), dict(gap=0.3),
                    dict(gap=0.2, min_words=3), dict(gap=0.2, min_sec=1.0)]
        configs = [dict(mode="wlk", latency=0.5, **DEFAULTS)]
        for lat in args.latency:
            configs += [dict(best, latency=lat, smooth=v) for v in variants]
        return configs
    if stage == "snap":
        best = dict(mode="fixed", tau=0.6, rho=0.1, delta=1.0)
        variants = [None, dict(min_words=4)]
        variants += [dict(snap=k) for k in (1, 2, 3)]
        variants += [dict(snap=k, min_words=4) for k in (1, 2, 3)]
        return [dict(best, latency=lat, smooth=v) for lat in args.latency for v in variants]
    if stage == "compare":
        best = dict(mode="fixed", tau=0.6, rho=0.1, delta=1.0)
        live = dict(min_words=4, snap=1)  # правила нынешнего живого режима
        configs = []
        try:
            import diart  # noqa: F401
        except ImportError:
            # В штатном образе worker-live diart больше нет (NeMo 3 требует
            # numpy 2) — считаем только Sortformer.
            print("diart не установлен — его строки пропускаются", flush=True)
        else:
            configs += [dict(mode="wlk", latency=0.5, **DEFAULTS),
                        dict(best, latency=0.5), dict(best, latency=0.5, smooth=live),
                        dict(best, latency=1.0, smooth=live)]
        # lat* — эталонный потоковый прогон NeMo, stream* — наш класс
        # живого режима на том же звуке; они должны совпасть.
        for variant, lat in (("lat0.32", 0.32), ("stream0.32", 0.32),
                             ("lat1.04", 1.04), ("stream1.04", 1.04)):
            if (SORTFORMER / variant).exists():
                sf = dict(mode="sortformer", variant=variant, latency=lat, tau=0.0, rho=0.0, delta=0.0)
                configs += [sf, dict(sf, smooth=live)]
        return configs
    if stage == "phantom":
        live = dict(min_words=4, snap=1)
        bases = [dict(mode="fixed", tau=0.6, rho=0.1, delta=1.0, latency=0.5)]
        for variant, lat in (("lat0.32", 0.32), ("lat1.04", 1.04)):
            bases.append(dict(mode="sortformer", variant=variant, latency=lat, tau=0.0, rho=0.0, delta=0.0))
        return [dict(b, smooth=dict(live, **({"establish": n} if n else {})))
                for b in bases for n in (0, 10, 20, 40)]
    if stage == "boundary":
        # Кто отдаёт начало новой реплики предыдущему говорящему: сама
        # разметка или правила черновика поверх неё.
        variants = [None, dict(min_words=4), dict(snap=1), dict(min_words=4, snap=1)]
        configs = []
        for variant, lat in (("stream0.32", 0.32), ("stream1.04", 1.04),
                             ("stream10.0", 10.0), ("stream30.4", 30.4)):
            if not (SORTFORMER / variant).exists():
                continue
            sf = dict(mode="sortformer", variant=variant, latency=lat, tau=0.0, rho=0.0, delta=0.0)
            configs += [dict(sf, smooth=v) for v in variants]
        return configs
    if stage == "lag":
        live = dict(min_words=4, snap=1)
        ahead = dict(min_words=4, snap=1, lookahead=True)
        configs = []
        for shift in (0.0, 0.08, 0.16, 0.24, 0.32, 0.4):
            sf = dict(mode="sortformer", variant="stream0.32", latency=0.32,
                      tau=0.0, rho=0.0, delta=0.0, shift=shift)
            configs += [dict(sf, smooth=v) for v in (None, live, ahead)]
        return configs
    if stage == "one":
        return [dict(mode=args.mode, latency=args.latency[0],
                     tau=args.tau, rho=args.rho, delta=args.delta)]
    raise SystemExit(f"неизвестный этап: {stage}")


def short(sid: str) -> str:
    return sid[:6] + ("*" if sid in UNTRUSTED else "")


def run_eval(refs: dict, args) -> None:
    names = [sid for sid in refs if (CACHE / f"{sid}.npz").exists()]
    configs = configs_for(args.stage, args)
    print(f"записей {len(names)}, конфигураций {len(configs)}, процессов {args.jobs}", flush=True)
    started = time.time()
    with Pool(args.jobs, initializer=_load_files, initargs=(names, refs)) as pool:
        results = pool.map(evaluate, configs, chunksize=1)
    print(f"посчитано за {time.time() - started:.0f} с\n")

    out = CACHE.parent / f"live_results_{args.stage}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    results.sort(key=lambda r: r["mean"])
    header = (f"{'режим':9} {'задерж':>6} {'tau':>4} {'rho':>4} {'delta':>5} {'сглаживание':>22} │ "
              f"{'среднее':>7} {'худшее':>6} {'реплик':>9} {'обрывков':>8} {'DER AMI':>7} │ ")
    header += " ".join(f"{short(s):>11}" for s in names)
    print(header)
    print("─" * len(header))
    for r in results[: args.top]:
        c = r["config"]
        cells = " ".join(
            f"{r['files'][s].get('der', r['files'][s].get('err', 0)) * 100:5.1f}% "
            f"{r['files'][s]['spk']}/{r['files'][s]['ref_spk']}"
            for s in names
        )
        sm = ",".join(f"{k}={v}" for k, v in (c.get("smooth") or {}).items()) or "—"
        label = c.get("variant") or c["mode"]
        print(f"{label:10} {c['latency']:5.2f} {c['tau']:4.2f} {c['rho']:4.2f} {c['delta']:5.2f} {sm:>22} │ "
              f"{r['mean'] * 100:6.1f}% {r['worst'] * 100:5.1f}% {r['turns']:4d}/{r['ref_turns']:<4d} "
              f"{r['scraps']:8d} {r['der'] * 100:6.1f}% │ {cells}")
    if args.stage in ("boundary", "lag"):
        print()
        print("смен говорящего на своих записях:", results[0]["changes"])
        print(f"{'вариант':12} {'сдвиг':>5} {'правила':>32} │ {'начало ушло предыдущему':>24} {'конец ушёл новому':>18} │ {'ошибка':>6} {'DER AMI':>7}")
        for r in sorted(results, key=lambda r: (r["config"].get("variant"), r["config"].get("shift", 0), str(r["config"].get("smooth")))):
            c = r["config"]
            sm = ",".join(f"{k}={v}" for k, v in (c.get("smooth") or {}).items()) or "—"
            n = max(1, r["changes"])
            print(f"{c.get('variant', c['mode']):12} {c.get('shift', 0):5.2f} {sm:>32} │ {r['late'] / n * 100:6.1f}% смен, {r['late_words']:4d} слов"
                  f"   {r['early'] / n * 100:6.1f}% смен, {r['early_words']:4d} слов │ {r['mean'] * 100:5.1f}% {r['der'] * 100:6.1f}%")
    print("\nв ячейке: доля слов под чужим именем (для встреч AMI — DER), найдено говорящих / в эталоне;"
          " * — чистовик ненадёжен, в среднее не входит;\n"
          "реплик — в черновике / в чистовике, обрывков — реплик короче трёх слов (без *)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["cache", "eval"])
    parser.add_argument("stage", nargs="?", default="variants")
    parser.add_argument("--mode", default="exclusive")
    parser.add_argument("--latency", type=float, nargs="+", default=[0.5])
    # Сетка для этапа grid: первый проход был по широкой, дальше — уточнение
    parser.add_argument("--taus", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.5])
    parser.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    parser.add_argument("--rho", type=float, default=DEFAULTS["rho"])
    parser.add_argument("--delta", type=float, default=DEFAULTS["delta"])
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    refs = json.loads(REFS.read_text(encoding="utf-8"))
    refs.update(ami_refs())
    if args.what == "cache":
        build_cache(refs)
    else:
        run_eval(refs, args)


if __name__ == "__main__":
    main()
