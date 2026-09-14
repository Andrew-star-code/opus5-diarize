# Тот же CUDA-базис, что у batch-воркера, но окружение отдельное:
# живой режим стоит на NeMo 3 (Streaming Sortformer), а batch — на
# pyannote.audio 4.x. Разводить их по контейнерам дешевле, чем воевать
# с резолвером зависимостей.
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
      ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

# 2.8 + cu128 держит sm_120 (Blackwell), так что RTX 50xx не страдает.
RUN pip install --no-cache-dir "torch==2.8.*" "torchaudio==2.8.*" \
      --index-url https://download.pytorch.org/whl/cu128

WORKDIR /srv
COPY requirements.txt .
# torch перечислен ещё раз, чтобы резолвер не заменил сборку cu128
# обычной с PyPI, когда её попросит какая-нибудь зависимость NeMo.
RUN pip install --no-cache-dir Cython packaging \
 && pip install --no-cache-dir "torch==2.8.*" "torchaudio==2.8.*" -r requirements.txt

COPY . .

EXPOSE 8001
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
