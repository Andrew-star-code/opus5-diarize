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

    # sortformer | diart | none. diart в образе больше не стоит (numpy
    # ниже 2 против NeMo 3); его ветка в engine.py оставлена для тех, кто
    # соберёт образ с ним сам.
    live_diarization: str = "sortformer"
    max_speakers: int = 8
    max_live_sessions: int = 2

    # NVIDIA Streaming Sortformer: сквозная потоковая модель, говорящие
    # держатся в кеше в порядке появления. Не больше четырёх говорящих —
    # остальных разберёт чистовик. Замер (bench/eval_live.py compare):
    # путаница говорящих на AMI 3,1% против 10% у diart, на записи втроём
    # находит всех троих.
    live_sortformer_model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    # Задержка разметки, секунды: 0.32 или 1.04 (настройки из статьи).
    # 1.04 чуть точнее (13,3% слов под чужим именем против 14,1%), но на
    # 0,7 с медленнее.
    live_sortformer_latency: float = 0.32
    # Второй проход: та же модель с запасом по времени идёт следом и
    # поправляет говорящих в уже показанных репликах. 10.0 или 30.4 (из
    # статьи); 0 — выключить. Сквозной замер (bench/live_boundary_e2e.py):
    # 13,5% слов под чужим именем против 15,2%.
    live_sortformer_refine_latency: float = 10.0

    # Кирпичи онлайн-диаризации diart. wespeaker вместо pyannote/embedding:
    # у второго отдельная лицензия, которую надо принимать вручную, а
    # при непринятой модель грузится как None и падает много позже.
    live_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    live_segmentation_model: str = "pyannote/segmentation-3.0"

    # Настройки потоковой кластеризации diart (только для diart). Подобраны замером на
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
