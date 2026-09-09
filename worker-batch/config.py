"""Настройки batch-воркера."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    backend_url: str = "http://backend:8000"
    data_dir: Path = Path("/data")

    batch_asr_model: str = "large-v3"
    compute_type: str = "float16"
    default_language: str = "ru"

    diarization_model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int = 0
    max_speakers: int = 8

    # Отсечение тишины до распознавания. Выключено намеренно — см.
    # развёрнутое объяснение в asr.py: на записях с фоновой музыкой или
    # шумом фильтр молча съедает больше половины речи.
    vad_filter: bool = False
    vad_min_silence_ms: int = 500

    # Опираться на предыдущий текст при распознавании следующего окна.
    # Включено: без контекста рвётся пунктуация и разъезжается написание
    # имён. От галлюцинаций защищают пороги в asr.py, а не выключение.
    use_previous_context: bool = True

    # Затравка для модели: имена участников, термины, названия. Самый
    # сильный рычаг для имён собственных — без неё незнакомое слово
    # записывается на слух как придётся.
    initial_prompt: str = ""

    hf_token: str = ""
    log_level: str = "INFO"

    # опрос очереди
    poll_interval: float = 3.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
