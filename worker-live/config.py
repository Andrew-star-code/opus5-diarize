"""Настройки live-воркера."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    backend_url: str = "http://backend:8000"
    data_dir: Path = Path("/data")

    live_asr_model: str = "large-v3-turbo"
    compute_type: str = "float16"
    default_language: str = "ru"

    # diart | sortformer | none
    live_diarization: str = "diart"
    max_speakers: int = 8
    max_live_sessions: int = 2

    # Кирпичи онлайн-диаризации. wespeaker вместо pyannote/embedding:
    # у второго отдельная лицензия, которую надо принимать вручную, а
    # при непринятой модель грузится как None и падает много позже.
    live_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    live_segmentation_model: str = "pyannote/segmentation-3.0"

    # Отсечение тишины в потоке. Экономит GPU, но на записи с фоновой
    # музыкой или шумом принимает речь за тишину и глотает её — ровно
    # та же беда, из-за которой VAD выключен в batch-пайплайне.
    # Здесь оставлено включённым: в стриминге фильтр ещё и держит
    # задержку, а черновик всё равно уточняется полной обработкой.
    # Если живой текст теряет слова — поставьте LIVE_VAD=false.
    live_vad: bool = True

    auto_refine_live: bool = True
    log_level: str = "INFO"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def diarization_enabled(self) -> bool:
        return self.live_diarization.lower() not in ("", "none", "off", "false")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
