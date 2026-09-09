"""Настройки сервиса. Все значения приходят из .env через docker compose."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # пути
    db_path: Path = Path("/db/scribe.db")
    data_dir: Path = Path("/data")

    # ASR / диаризация — сюда backend не лезет, но отдаёт значения фронту,
    # чтобы в «Настройках» было видно, чем именно всё считается.
    batch_asr_model: str = "large-v3"
    live_asr_model: str = "large-v3-turbo"
    default_language: str = "ru"
    compute_type: str = "float16"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    live_diarization: str = "diart"
    # Затравка по умолчанию: применяется к записям, где своя не задана.
    initial_prompt: str = ""
    num_speakers: int = 0
    max_speakers: int = 8

    # Обработка микрофона средствами браузера: подавление эха,
    # шумоподавление, автогромкость. Выключено — это средства для
    # звонков, и для расшифровки они режут речь. Подробности в
    # frontend/src/lib/live.ts.
    mic_processing: bool = False

    # поведение
    max_live_sessions: int = 2
    auto_refine_live: bool = True
    retention_days: int = 0
    log_level: str = "INFO"

    # задание считается зависшим, если воркер молчит дольше этого (сек)
    job_stale_after: int = 600

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
