# Тот же CUDA-базис, что у batch-воркера, но окружение отдельное:
# diart тянет свою ветку pyannote и конфликтует с pyannote.audio 4.x,
# который стоит в batch-образе. Разводить их по контейнерам дешевле,
# чем воевать с резолвером зависимостей.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# libportaudio2 нужен не нам, а diart: он безусловно импортирует
# sounddevice ради микрофонных источников, хотя аудио мы подаём сами.
# Без библиотеки импорт падает с "PortAudio library not found".
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ffmpeg libsndfile1 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

# diart официально поддерживает только Python 3.11-3.12 (ограничение numpy),
# поэтому версия здесь зафиксирована базовым образом Ubuntu 24.04.
RUN python3.12 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

# Версии закреплены, и это не перестраховка, а необходимость.
#
# torchaudio: pyannote.audio 3.x, который тянет за собой diart, обращается
# к torchaudio.AudioMetaData. В torchaudio 2.9 этот атрибут вырезали, и
# импорт diart падает. В 2.8 он ещё есть — проверено.
#
# torchvision: напрямую не нужен, но его тянет diart, и с обычного PyPI
# приезжает сборка под другую версию torch. Итог — падение на импорте
# с загадочным "operator torchvision::nms does not exist". Поэтому берём
# из того же индекса и той же ветки.
#
# 2.8 + cu128 держит sm_120 (Blackwell), так что RTX 50xx не страдает.
RUN pip install --no-cache-dir \
      "torch==2.8.*" "torchvision==0.23.*" "torchaudio==2.8.*" \
      --index-url https://download.pytorch.org/whl/cu128

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
