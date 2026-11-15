"""Live-match ingestion and idempotent persistence orchestration."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.data.models import MatchRecord
from app.data.normalization import normalize_cito_live_match
from app.storage.repository import save_match


def persist_live_rows(session: Session, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize and upsert valid provider rows in the caller's transaction."""
    persisted: list[dict[str, Any]] = []
    unmapped = 0
    for row in rows:
        normalized = normalize_cito_live_match(row)
        if normalized is None:
            unmapped += 1
            continue
        # MatchRecord.raw is intentionally retained for provider-specific live fields.
        canonical = MatchRecord.model_validate({
            **{key: value for key, value in normalized.items() if key != "live"},
            "raw": row,
        })
        record = save_match(session, canonical)
        persisted.append({
            "id": record.id,
            "provider": record.provider,
            "external_id": record.external_id,
            "status": record.status,
            "player_1_name": record.player_1_name,
            "player_2_name": record.player_2_name,
        })

    return {
        "provider": "cito",
        "fetched_count": len(rows),
        "persisted_count": len(persisted),
        "unmapped_count": unmapped,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "matches": persisted,
    }
