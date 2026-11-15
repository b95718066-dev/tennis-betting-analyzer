"""Odds ingestion and append-only snapshot persistence."""

from typing import Any

from sqlalchemy.orm import Session

from app.data.providers.the_odds_api import TheOddsApiProvider
from app.storage.repository import save_odds_snapshot


def persist_odds_events(
    session: Session, events: list[dict[str, Any]]
) -> dict[str, int | str]:
    inserted = 0
    duplicates = 0
    invalid = 0
    for event in events:
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
    }
