"""Полный структурированный дамп — для внешней обработки и бэкапа."""
from __future__ import annotations

import json

from ..schemas import SessionDetail, SessionStats


def to_json(detail: SessionDetail, stats: SessionStats) -> str:
    payload = {
        "id": detail.id,
        "title": detail.title,
        "created_at": detail.created_at.isoformat(),
        "source_type": detail.source_type,
        "quality": detail.quality,
        "language": detail.language,
        "duration_sec": detail.duration_sec,
        "model_info": detail.model_info,
        "speakers": [sp.model_dump() for sp in detail.speakers],
        "segments": [
            {
                "idx": s.idx,
                "start": s.start,
                "end": s.end,
                "speaker_id": s.speaker_id,
                "text": s.text,
                "words": [w.model_dump() for w in s.words],
            }
            for s in detail.segments
        ],
        "stats": stats.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
