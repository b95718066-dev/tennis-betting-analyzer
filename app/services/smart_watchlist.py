"""Persistence helpers for the opt-in Smart Match Watchlist."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import SmartMatchWatchlist


def watch_match(session: Session, match_id: str, *, source: str = "manual") -> SmartMatchWatchlist:
    match_id = match_id.strip()
    if not match_id or len(match_id) > 180 or "/" in match_id:
        raise ValueError("match_id must be a valid Cito match ID.")
    row = session.scalar(select(SmartMatchWatchlist).where(
        SmartMatchWatchlist.provider == "cito",
        SmartMatchWatchlist.external_match_id == match_id,
    ))
    now = datetime.now(timezone.utc)
    if row is None:
        row = SmartMatchWatchlist(
            provider="cito", external_match_id=match_id, source=source,
            active=True, created_at=now, updated_at=now,
        )
        session.add(row)
    else:
        row.active = True
        row.source = source
        row.updated_at = now
    session.flush()
    return row


def unwatch_match(session: Session, match_id: str) -> bool:
    row = session.scalar(select(SmartMatchWatchlist).where(
        SmartMatchWatchlist.provider == "cito",
        SmartMatchWatchlist.external_match_id == match_id,
        SmartMatchWatchlist.active.is_(True),
    ))
    if row is None:
        return False
    row.active = False
    row.updated_at = datetime.now(timezone.utc)
    session.flush()
    return True


def list_watched_matches(session: Session, *, active_only: bool = True) -> list[SmartMatchWatchlist]:
    statement = select(SmartMatchWatchlist).where(
        SmartMatchWatchlist.provider == "cito"
    ).order_by(SmartMatchWatchlist.created_at.desc())
    if active_only:
        statement = statement.where(SmartMatchWatchlist.active.is_(True))
    return list(session.scalars(statement).all())
