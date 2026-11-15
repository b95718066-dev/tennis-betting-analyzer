from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.services.odds_sync import persist_odds_events
from app.storage.database import create_schema
from app.storage.models import OddsObservation


def odds_event():
    updated = datetime.now(timezone.utc).isoformat()
    return {
        "id": "odds-event-1",
        "commence_time": updated,
        "bookmakers": [{
            "key": "book-a",
            "last_update": updated,
            "markets": [{
                "key": "h2h",
                "last_update": updated,
                "outcomes": [
                    {"name": "Player One", "price": 1.85},
                    {"name": "Player Two", "price": 2.05},
                ],
            }],
        }],
    }


def test_odds_events_persist_append_only_snapshots_and_deduplicate_replays():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    event = odds_event()

    with Session(engine) as session, session.begin():
        first = persist_odds_events(session, [event])
        second = persist_odds_events(session, [event])
        records = session.scalars(select(OddsObservation)).all()

        assert first["snapshots_inserted"] == 2
        assert second["snapshots_inserted"] == 0
        assert second["snapshots_deduplicated"] == 2
        assert len(records) == 2
        assert {record.market for record in records} == {"h2h"}


def test_odds_event_links_to_cito_match_by_exact_names_and_start_time():
    from datetime import timedelta

    from app.data.models import MatchRecord, MatchStatus, Surface, Tour
    from app.storage.models import ProviderMatchLink
    from app.storage.repository import save_match

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    event = odds_event()
    event.update({
        "home_team": "PLAYER TWO",
        "away_team": "Player One",
        "commence_time": (start + timedelta(minutes=10)).isoformat(),
    })
    record = MatchRecord(
        provider="cito",
        external_id="cito-canonical-77",
        tour=Tour.ATP,
        status=MatchStatus.LIVE,
        scheduled_at=start,
        surface=Surface.HARD,
        player_1_name="Player One",
        player_2_name="Player Two",
    )

    with Session(engine) as session, session.begin():
        save_match(session, record)
        result = persist_odds_events(session, [event])
        link = session.scalar(select(ProviderMatchLink))

        assert result["events_matched_to_cito"] == 1
        assert result["events_unmatched_to_cito"] == 0
        assert link.source_external_id == "odds-event-1"
        assert link.canonical_external_id == "cito-canonical-77"
        assert link.match_method == "exact_player_pair_and_start_time"


def test_odds_event_is_not_linked_when_cito_match_is_ambiguous():
    from datetime import timedelta

    from app.data.models import MatchRecord, MatchStatus, Surface, Tour
    from app.storage.models import ProviderMatchLink
    from app.storage.repository import save_match

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    event = odds_event()
    event.update({
        "home_team": "Player One",
        "away_team": "Player Two",
        "commence_time": start.isoformat(),
    })

    with Session(engine) as session, session.begin():
        for suffix, offset in (("a", 0), ("b", 60)):
            save_match(session, MatchRecord(
                provider="cito",
                external_id=f"cito-{suffix}",
                tour=Tour.ATP,
                status=MatchStatus.SCHEDULED,
                scheduled_at=start + timedelta(minutes=offset),
                surface=Surface.HARD,
                player_1_name="Player One",
                player_2_name="Player Two",
            ))
        result = persist_odds_events(session, [event])

        assert result["events_matched_to_cito"] == 0
        assert result["events_unmatched_to_cito"] == 1
        assert session.scalars(select(ProviderMatchLink)).all() == []
