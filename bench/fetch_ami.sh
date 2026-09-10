#!/usr/bin/env bash
# Скачивает три встречи корпуса AMI с эталонной разметкой для стенда.
#
#   bash bench/fetch_ami.sh
#
# Около 109 МБ. Аудио — зеркало корпуса AMI Эдинбургского университета
# (лицензия CC BY 4.0), разметка — репозиторий pyannote/AMI-diarization-setup,
# на котором авторы pyannote публикуют свои замеры.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data

AUDIO="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
RTTM="https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test"

for meeting in ES2004a IS1009a TS3003a; do
  echo "== ${meeting}"
  curl -fsSL -o "data/${meeting}.rttm" "${RTTM}/${meeting}.rttm"
  # -C - докачивает оборванное: канал бывает нестабильным
  curl -fSL --retry 3 -C - -o "data/${meeting}.wav" \
    "${AUDIO}/${meeting}/audio/${meeting}.Mix-Headset.wav"
done

echo "Готово: $(ls data | wc -l) файлов в bench/data"
