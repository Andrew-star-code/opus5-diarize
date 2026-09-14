# Образ стенда для NVIDIA Streaming Sortformer — только для замеров.
#
# Отдельно от worker-live намеренно: diart там держит numpy ниже 2, а NeMo
# 3 требует свежий. Если Sortformer выиграет замер, его встраивание будет
# отдельной задачей, а этот образ останется стендом для сравнения.
#
#   docker build -t scribe-bench-sortformer bench/sortformer
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# build-essential и python3.12-dev — для зависимостей NeMo, которые
# собираются из исходников (Cython-расширения).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip build-essential \
      ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

# Та же ветка torch, что в сервисах: 2.8 + cu128 держит sm_120 (RTX 50xx).
RUN pip install --no-cache-dir "torch==2.8.*" "torchaudio==2.8.*" \
      --index-url https://download.pytorch.org/whl/cu128

# 3.0 — та же ветка NeMo, что закреплена у WhisperLiveKit для Sortformer.
# torch перечислен явно, чтобы резолвер не заменил сборку cu128 на
# обычную с PyPI.
RUN pip install --no-cache-dir Cython packaging \
 && pip install --no-cache-dir "torch==2.8.*" "nemo_toolkit[asr]==3.0.0" "pyannote.metrics==3.2.*"

WORKDIR /bench
