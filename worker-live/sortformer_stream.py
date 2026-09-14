"""Streaming Sortformer для живого режима: звук кусками → сегменты говорящих.

Повторяет потоковый прогон самого NeMo (forward_streaming +
streaming_feat_loader), но для звука, который приходит по кусочку. NeMo
считает признаки по всей записи сразу и режет их на порции; здесь они
считаются на скользящем окне с запасом по краям. Кадр признаков зависит
только от n_fft/2 сэмплов по обе стороны от своего центра, поэтому внутри
окна числа выходят те же, что по всей записи, а запасные кадры
выбрасываются.

Стенд (bench/eval_sortformer.py) гоняет этот же класс, так что замер
переносится на живой режим один к одному.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager

import numpy as np
import torch

SAMPLE_RATE = 16000

# Настройки порций из статьи Streaming Sortformer (arXiv 2507.18446,
# таблица 2). Кадр модели — 80 мс; задержка — порция плюс заглядывание
# вперёд. Точность у них почти одинаковая, 0,32 с отзывчивее.
PRESETS = {
    0.32: dict(chunk_len=3, chunk_right_context=1, fifo_len=188,
               spkcache_update_period=144, spkcache_len=188),
    1.04: dict(chunk_len=6, chunk_right_context=7, fifo_len=188,
               spkcache_update_period=144, spkcache_len=188),
    # Большие задержки — для второго прохода, который поправляет уже
    # показанные реплики: у модели перед глазами 10–30 с продолжения.
    10.0: dict(chunk_len=124, chunk_right_context=1, fifo_len=124,
               spkcache_update_period=124, spkcache_len=188),
    30.4: dict(chunk_len=340, chunk_right_context=40, fifo_len=40,
               spkcache_update_period=300, spkcache_len=188),
}


def configure(model, latency: float) -> None:
    """Выставляет модели настройки порций для заданной задержки."""
    try:
        preset = PRESETS[latency]
    except KeyError:
        raise ValueError(f"задержка Sortformer {latency} с — есть только {sorted(PRESETS)}") from None
    for key, value in preset.items():
        setattr(model.sortformer_modules, key, value)
    model.sortformer_modules._check_streaming_parameters()


# Настройки порций NeMo читает с модели прямо во время шага
# (streaming_update, _compress_spkcache, _gather_spkcache_and_preds).
# Чтобы потоки с разной задержкой делили одну модель, каждый выставляет
# свои настройки под этой блокировкой на время шага.
_MODEL_LOCK = threading.Lock()


class SortformerStream:
    """Потоковая диаризация одной записи.

    push() принимает звук, step() прогоняет все готовые порции и отдаёт
    новые сегменты (начало, конец, говорящий) в секундах от начала записи.
    Говорящие нумеруются с нуля в порядке появления. finish() дожимает
    хвост в конце записи.
    """

    def __init__(self, model, threshold: float = 0.5, latency: float | None = None):
        self.model = model
        sm = model.sortformer_modules
        feat = model.preprocessor.featurizer
        self.hop = int(feat.hop_length)
        self.half = int(feat.n_fft) // 2
        # latency — свой набор настроек порций из PRESETS; None — те, что
        # выставлены на модели сейчас (так работает стенд через configure).
        self.preset = PRESETS[latency] if latency is not None else None
        chunk_len = self.preset["chunk_len"] if self.preset else sm.chunk_len
        right_ctx = self.preset["chunk_right_context"] if self.preset else sm.chunk_right_context
        self.sub = int(sm.subsampling_factor)
        self.chunk = chunk_len * self.sub             # кадров признаков в порции
        self.left = sm.chunk_left_context * self.sub
        self.right = right_ctx * self.sub
        # Запас по краям окна: n_fft/2 сэмплов на кадр плюс предыскажение,
        # которое тянет за собой предыдущий сэмпл.
        self.margin = math.ceil(self.half / self.hop) + 2
        self.frame_sec = self.sub * self.hop / SAMPLE_RATE
        self.threshold = threshold
        self.n_spk = int(sm.n_spk)

        self.audio = np.zeros(0, dtype=np.float32)
        self.audio_start = 0   # номер первого сэмпла буфера от начала записи
        self.samples = 0       # всего сэмплов принято
        self.stt = 0           # первый кадр признаков следующей порции
        self.frames_out = 0    # кадров модели уже выдано
        with self._settings(), torch.inference_mode():
            self.state = sm.init_streaming_state(
                batch_size=1, async_streaming=model.async_streaming, device=model.device
            )

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.audio = np.concatenate([self.audio, pcm])
        self.samples += len(pcm)

    def step(self) -> list[tuple[float, float, int]]:
        """Все порции, для которых уже есть звук вместе с заглядыванием вперёд."""
        segments = []
        while self.stt + self.chunk + self.right <= self._ready_frames():
            segments += self._run(self.stt + self.chunk, self.right)
        return segments

    def finish(self) -> list[tuple[float, float, int]]:
        """Хвост записи: последние порции без полного заглядывания вперёд."""
        segments = self.step()
        total = self.samples // self.hop + 1  # кадров признаков во всей записи
        while self.stt < total:
            end = min(self.stt + self.chunk, total)
            segments += self._run(end, min(self.right, total - end))
        return segments

    # --------------------------------------------------------------- внутреннее

    @contextmanager
    def _settings(self):
        """Свои настройки порций на модели — на время шага, под общей блокировкой."""
        with _MODEL_LOCK:
            if self.preset:
                for key, value in self.preset.items():
                    setattr(self.model.sortformer_modules, key, value)
            yield

    def _ready_frames(self) -> int:
        """Сколько кадров признаков уже не изменит будущий звук."""
        if self.samples < self.half:
            return 0
        return (self.samples - self.half) // self.hop + 1

    def _features(self, first: int, last: int) -> torch.Tensor:
        """Кадры признаков [first, last) — те же, что по всей записи."""
        s0 = max(0, (first - self.margin) * self.hop)  # кратно шагу — кадры совпадут
        s1 = min(self.samples, (last + self.margin) * self.hop)
        x = self.audio[s0 - self.audio_start: s1 - self.audio_start]
        signal = torch.from_numpy(np.ascontiguousarray(x)).to(self.model.device).unsqueeze(0)
        length = torch.tensor([signal.shape[1]], device=self.model.device)
        feats, _ = self.model.preprocessor(input_signal=signal, length=length)
        k = s0 // self.hop
        return feats[:, :, first - k: last - k]

    def _run(self, end: int, right: int) -> list[tuple[float, float, int]]:
        left = min(self.left, self.stt)
        with self._settings(), torch.inference_mode():
            chunk = self._features(self.stt - left, end + right).transpose(1, 2)
            length = torch.tensor([chunk.shape[1]], device=self.model.device)
            empty = torch.zeros((1, 0, self.n_spk), device=self.model.device)
            # total_preds в NeMo только копится; передаём пустой и получаем
            # вероятности ровно для кадров этой порции.
            self.state, preds = self.model.forward_streaming_step(
                processed_signal=chunk,
                processed_signal_length=length,
                streaming_state=self.state,
                total_preds=empty,
                left_offset=left,
                right_offset=right,
            )
        self.stt = end
        self._trim()
        return self._segments(preds[0].float().cpu().numpy())

    def _trim(self) -> None:
        keep = max(0, (self.stt - self.left - self.margin) * self.hop)
        if keep > self.audio_start:
            self.audio = self.audio[keep - self.audio_start:]
            self.audio_start = keep

    def _segments(self, preds: np.ndarray) -> list[tuple[float, float, int]]:
        """Вероятности кадров порции → отрезки активности каждого говорящего."""
        base, fs = self.frames_out, self.frame_sec
        active = preds > self.threshold
        out = []
        for spk in range(active.shape[1]):
            start = None
            for i, on in enumerate(active[:, spk]):
                if on and start is None:
                    start = i
                elif not on and start is not None:
                    out.append(((base + start) * fs, (base + i) * fs, spk))
                    start = None
            if start is not None:
                out.append(((base + start) * fs, (base + len(active)) * fs, spk))
        self.frames_out += len(active)
        return sorted(out)
