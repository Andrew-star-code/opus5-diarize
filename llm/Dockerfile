# Языковая модель для полировки чистовика — Ollama.
#
# Не готовый образ ollama/ollama, а сборка по тому же рецепту, что
# официальный установщик: база CUDA (та же, что у воркеров, — уже есть в
# кеше Docker) плюс архив Ollama из релиза на GitHub. Так образ собирается
# и на медленном канале, где Docker Hub рвёт соединение при установке TLS,
# а curl с повторами и длинными таймаутами справляется. CUDA-библиотеки
# Ollama привозит в архиве сама.
#
#   docker compose build llm
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl zstd \
 && rm -rf /var/lib/apt/lists/*

ARG OLLAMA_VERSION=0.34.0
RUN curl -fL --retry 8 --retry-all-errors --connect-timeout 60 \
      -o /tmp/ollama.tar.zst \
      "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
 && tar --zstd -xf /tmp/ollama.tar.zst -C /usr \
 && rm /tmp/ollama.tar.zst \
 && ollama --version

ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
