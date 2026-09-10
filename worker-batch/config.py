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

    # Слияние «крошечных» говорящих — обрывков чужого голоса, выделенных
    # в отдельного человека. Сливается тот, у кого доля речи меньше
    # max_share, в самого похожего по голосу, если сходство эмбеддингов
    # не ниже min_similarity. Подобрано на трёх встречах AMI: строгий
    # вариант исправил лишнего пятого говорящего и не тронул остальные
    # записи, мягкий (5% и 0,30) склеил четверых настоящих людей в двоих.
    # Подробности — в merge.tiny_speaker_renames.
    tiny_speaker_max_share: float = 0.02
    tiny_speaker_min_similarity: float = 0.40

    # Сглаживание коротких вставок чужого говорящего на уровне слов.
    # Выключено по замеру на AMI: доля слов с неверным говорящим росла
    # с 2,20% до 2,48%, хуже на каждой из трёх записей. См. merge._smooth.
    smooth_speaker_turns: bool = False

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
