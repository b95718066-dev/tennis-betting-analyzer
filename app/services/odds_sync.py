"""Odds ingestion, conservative cross-provider matching and persistence."""

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.providers.the_odds_api import TheOddsApiProvider
from app.storage.models import TennisMatch
from app.storage.repository import save_odds_snapshot, save_provider_match_link


def _normalized_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    folded = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def match_odds_event_to_cito(
    session: Session, event: dict[str, Any], *, max_time_difference_hours: int = 24
) -> bool:
    """Link only a unique exact participant pair with a close scheduled start."""
    source_id = event.get("id")
    home = _normalized_name(event.get("home_team"))
    away = _normalized_name(event.get("away_team"))
    commence = _parse_time(event.get("commence_time"))
    if not isinstance(source_id, str) or not source_id or not home or not away or commence is None:
        return False

    wanted = {home, away}
    candidates = []
    for match in session.scalars(select(TennisMatch).where(TennisMatch.provider == "cito")).all():
        pair = {
            _normalized_name(match.player_1_name),
            _normalized_name(match.player_2_name),
        }
        scheduled = match.scheduled_at
        if scheduled is None or pair != wanted:
            continue
        if scheduled.tzinfo is None or scheduled.utcoffset() is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        if abs((scheduled.astimezone(timezone.utc) - commence).total_seconds()) <= max_time_difference_hours * 3600:
            candidates.append(match)

    if len(candidates) != 1:
        return False

    canonical = candidates[0]
    save_provider_match_link(
        session,
        source_provider="the_odds_api",
        source_external_id=source_id,
        canonical_provider=canonical.provider,
        canonical_external_id=canonical.external_id,
        match_method="exact_player_pair_and_start_time",
        confidence=1.0,
    )
    return True


def persist_odds_events(
    session: Session, events: list[dict[str, Any]]
) -> dict[str, int | str]:
    inserted = 0
    duplicates = 0
    invalid = 0
    matched = 0
    for event in events:
        matched += int(match_odds_event_to_cito(session, event))
        snapshots = TheOddsApiProvider.normalize_event_odds(event)
        if not snapshots:
            invalid += 1
            continue
        for snapshot in snapshots:
            _, was_inserted = save_odds_snapshot(session, snapshot)
            inserted += int(was_inserted)
            duplicates += int(not was_inserted)
    return {
        "provider": "the_odds_api",
        "events_fetched": len(events),
        "snapshots_inserted": inserted,
        "snapshots_deduplicated": duplicates,
        "events_without_valid_odds": invalid,
        "events_matched_to_cito": matched,
        "events_unmatched_to_cito": len(events) - matched,
    }
