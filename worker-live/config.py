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

    # Настройки потоковой кластеризации diart. Подобраны замером на
    # записях с опорной разметкой (bench/eval_live.py), значения diart по
    # умолчанию — в скобках.
    #
    # rho_update (0.3) — какую долю пятисекундного окна человек должен
    # проговорить, чтобы diart завёл его как нового говорящего. При 0.3
    # это полторы секунды: в разговоре втроём короткие реплики третьего
    # не дотягивали, и его растаскивало по двум другим.
    live_diar_rho_update: float = 0.1
    # tau_active (0.6) — порог, с которого говорящий считается активным.
    live_diar_tau_active: float = 0.6
    # delta_new (1.0) — насколько голос должен отличаться от известных,
    # чтобы стать новым говорящим.
    live_diar_delta_new: float = 1.0
    # Задержка разметки, секунды (0.5 — минимум, равен шагу diart).
    # Больше — diart сводит больше перекрывающихся окон и реже ошибается,
    # но слово дольше остаётся серым, пока его не покроет разметка.
    live_diar_latency: float = 0.5
    # Серия реплик одного говорящего короче стольких слов не может сменить
    # говорящего — остаётся у предыдущего. Убирает дрожь меток на стыке
    # реплик («Но их» / «нельзя»). 1 — выключить.
    live_min_turn_words: int = 4
    # Если граница говорящих прошла посреди предложения, а в пределах
    # стольких слов предложение кончается, граница переезжает туда
    # («…Шептать. Надо» / «всегда…» -> «…Шептать.» / «Надо всегда…»).
    # 0 — выключить.
    live_snap_words: int = 1

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
