"""Idempotent historical archive persistence and leakage-safe features."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.storage.models import HistoricalMatchSnapshot


def _nested_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    result = value.get("canonical_id") or value.get("id")
    return str(result) if result is not None else None


def persist_historical_rows(
    session: Session, rows: list[dict[str, Any]], *, provider: str = "cito"
) -> dict[str, int]:
    inserted = updated = skipped = 0
    for row in rows:
        external_id = row.get("canonical_id") or row.get("id") or row.get("match_id")
        if not isinstance(external_id, (str, int)) or not str(external_id).strip():
            skipped += 1
            continue
        p1 = row.get("player1") or row.get("player_1")
        p2 = row.get("player2") or row.get("player_2")
        winner = row.get("winner")
        p1_id = _nested_id(p1)
        p2_id = _nested_id(p2)
        winner_id = _nested_id(winner)
        # Cito's archive can order the winner as player1. Canonicalize the
        # two participant slots by stable ID so model outcomes are not biased
        # by provider presentation order. Keep winner_id tied to the identity.
        if p1_id and p2_id and p2_id < p1_id:
            p1_id, p2_id = p2_id, p1_id
        # Some archive rows carry winner_id directly.
        if winner_id is None and row.get("winner_id") is not None:
            winner_id = str(row["winner_id"])
        # Prefer an exact scheduled/actual timestamp only when Cito explicitly
        # marks the start time as known. Otherwise retain the date-only value:
        # never manufacture a kickoff time from a calendar date.
        starts_at = row.get("starts_at") or row.get("scheduled_at")
        if row.get("start_time_known") is True and isinstance(starts_at, str):
            match_date = starts_at
        else:
            date_value = row.get("date") or row.get("match_date") or row.get("scheduled_at")
            match_date = date_value if isinstance(date_value, str) else None
        stats_available = row.get("stats_available")
        if not isinstance(stats_available, bool):
            stats_available = None
        stats_code = row.get("stats_unavailable_code")
        existing = session.scalar(
            select(HistoricalMatchSnapshot).where(
                HistoricalMatchSnapshot.provider == provider,
                HistoricalMatchSnapshot.external_id == str(external_id),
            )
        )
        values = {
            "match_date": match_date,
            "tour": str(row.get("tour")) if row.get("tour") is not None else None,
            "surface": str(row.get("surface")).lower() if row.get("surface") is not None else None,
            "player_1_id": p1_id,
            "player_2_id": p2_id,
            "winner_id": winner_id,
            "stats_available": stats_available,
            "stats_unavailable_code": str(stats_code)[:120] if stats_code else None,
            "raw": row,
        }
        if existing is None:
            session.add(HistoricalMatchSnapshot(
                provider=provider, external_id=str(external_id), **values
            ))
            inserted += 1
        else:
            # Re-fetches can fill late-published stats and corrected archive fields.
            for key, value in values.items():
                setattr(existing, key, value)
            existing.updated_at = datetime.now(timezone.utc)
            updated += 1
    session.flush()
    return {"fetched": len(rows), "inserted": inserted, "updated": updated, "skipped": skipped}


def player_history_features(
    session: Session,
    *,
    player_id: str,
    before: str,
    surface: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Compute match-result features using only rows dated strictly before cutoff."""
    if not player_id.strip():
        raise ValueError("player_id is required.")
    try:
        cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    cutoff_text = cutoff.astimezone(timezone.utc).isoformat()
    query = select(HistoricalMatchSnapshot).where(
        HistoricalMatchSnapshot.provider == "cito",
        HistoricalMatchSnapshot.match_date.is_not(None),
        or_(
            HistoricalMatchSnapshot.player_1_id == player_id,
            HistoricalMatchSnapshot.player_2_id == player_id,
        ),
    )
    if surface:
        query = query.where(HistoricalMatchSnapshot.surface == surface.lower())
    candidates = session.scalars(
        query.order_by(HistoricalMatchSnapshot.match_date.desc())
    ).all()
    rows = []
    for row in candidates:
        try:
            match_time = datetime.fromisoformat(row.match_date.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        # Date-only archive rows are usable only on later calendar days;
        # treating their date as midnight could leak a same-day result.
        if match_time.tzinfo is None or match_time.utcoffset() is None:
            if match_time.date() < cutoff.astimezone(timezone.utc).date():
                rows.append(row)
        elif match_time.astimezone(timezone.utc) < cutoff.astimezone(timezone.utc):
            rows.append(row)
        if len(rows) >= limit:
            break
    decided = [r for r in rows if r.winner_id in {r.player_1_id, r.player_2_id}]
    wins = sum(r.winner_id == player_id for r in decided)
    losses = len(decided) - wins
    return {
        "player_id": player_id,
        "cutoff_exclusive": cutoff_text,
        "surface": surface.lower() if surface else None,
        "matches_considered": len(decided),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(decided) if decided else None,
        "source": "cito_historical_match_snapshots",
        "limitations": [
            "Only matches with a stored match_date strictly before cutoff are included.",
            "Rows without a resolved winner are excluded from win-rate calculations.",
            "Serve-point rates are not inferred from match outcomes.",
        ],
    }
