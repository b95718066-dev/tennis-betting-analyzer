"""Persistence helpers for the opt-in Smart Match Watchlist."""

import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import SmartMatchWatchlist


_MATCH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,179}")


def validate_match_id(match_id: str) -> str:
    """Accept one opaque, path-safe Cito identifier; never accept URL/path syntax."""
    if not isinstance(match_id, str):
        raise ValueError("match_id must be a valid Cito match ID.")
    normalized = match_id.strip()
    if not _MATCH_ID_RE.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError("match_id must be a valid Cito match ID.")
    return normalized


def watch_match(session: Session, match_id: str, *, source: str = "manual") -> SmartMatchWatchlist:
    match_id = validate_match_id(match_id)
    now = datetime.now(timezone.utc)
    row = session.scalar(select(SmartMatchWatchlist).where(
        SmartMatchWatchlist.provider == "cito",
        SmartMatchWatchlist.external_match_id == match_id,
    ))
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
    match_id = validate_match_id(match_id)
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


def list_watched_matches(
    session: Session,
    *,
    active_only: bool = True,
    poll_priority: bool = False,
) -> list[SmartMatchWatchlist]:
    statement = select(SmartMatchWatchlist).where(
        SmartMatchWatchlist.provider == "cito"
    )
    if active_only:
        statement = statement.where(SmartMatchWatchlist.active.is_(True))
    if poll_priority:
        # Null timestamps first, then the oldest check; this prevents starvation
        # when the active watchlist contains more than one polling batch.
        statement = statement.order_by(
            SmartMatchWatchlist.last_checked_at.asc().nullsfirst(),
            SmartMatchWatchlist.created_at.asc(),
        )
    else:
        statement = statement.order_by(SmartMatchWatchlist.created_at.desc())
    return list(session.scalars(statement).all())


def extract_match_status(payload: object) -> str | None:
    """Read status from common Cito response envelopes without guessing."""
    candidates = [payload]
    if isinstance(payload, dict):
        for key in ("match", "data", "item", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
                for inner_key in ("match", "item", "result"):
                    inner = nested.get(inner_key)
                    if isinstance(inner, dict):
                        candidates.append(inner)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("status", "match_status", "matchStatus", "state"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower().replace("-", "_").replace(" ", "_")
    return None


TERMINAL_MATCH_STATUSES = frozenset({
    "finished", "finish", "complete", "completed", "ended", "retired",
    "walkover", "cancelled", "canceled", "abandoned", "final", "ft", "done",
})


def is_terminal_match_status(status: str | None) -> bool:
    return status in TERMINAL_MATCH_STATUSES
