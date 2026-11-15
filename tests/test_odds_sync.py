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
