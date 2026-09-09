# cudnn-runtime, а не base: CTranslate2 (движок faster-whisper) линкуется
# с cuDNN 9. CUDA 12.8 — минимум для Blackwell/sm_120 (RTX 50xx).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 помечает системный Python как externally-managed (PEP 668),
# поэтому ставим всё в отдельный venv.
RUN python3.12 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

# torch отдельным слоем: он весит больше всего и меняется реже остальных
RUN pip install --no-cache-dir torch torchaudio \
      --index-url https://download.pytorch.org/whl/cu128

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "main.py"]
