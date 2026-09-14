# Образ стенда распознавания: batch-воркер плюс GigaAM-v3 (Сбер, MIT).
#
# Поверх образа worker-batch, чтобы Whisper мерился ровно тем кодом и теми
# версиями, что в чистовике, а GigaAM — в том же окружении рядом с ним.
# Сервисный образ не трогается.
#
#   docker compose build worker-batch
#   docker build -t scribe-bench-asr bench/asr
FROM scribe-worker-batch:local

# git в образе нет — GigaAM ставится из архива ветки. Его ядро закрепляет
# onnxruntime 1.23; faster-whisper (Silero VAD) с этой версией работает.
# pyarrow — чтобы читать parquet с тестовым набором Golos; num2words —
# привести цифры к словам: эталон Golos пишет «пять минут», а Whisper и
# GigaAM с пунктуацией — «5 минут».
RUN pip install --no-cache-dir \
      "gigaam @ https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip" \
      "pyarrow>=17" "num2words>=0.5.13"
