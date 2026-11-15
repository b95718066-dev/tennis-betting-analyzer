from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.data.normalization import normalize_cito_live_match
from app.services.live_sync import persist_live_rows
from app.storage.database import create_schema
from app.storage.models import TennisMatch


def valid_row():
    return {
        "id": "cito-live-001",
        "canonical_id": "cito-live-001",
        "tour": "ATP",
        "status": "IN_PROGRESS",
        "scheduled_at": "2026-11-15T18:00:00+00:00",
        "tournament_name": "Test Open",
        "surface": "Hard",
        "best_of": 3,
        "player1": {"id": "p1", "name": "Player One", "current_game_score": "15", "is_serving": True},
        "player2": {"id": "p2", "name": "Player Two", "current_game_score": "0", "is_serving": False},
        "sets": [{"player1_games": 2, "player2_games": 1}],
        "current_set": 1,
        "game_score": "15-0",
        "server": 1,
    }


def test_live_rows_are_persisted_and_repeat_sync_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    row = valid_row()

    with Session(engine) as session, session.begin():
        first = persist_live_rows(session, [row])
        second = persist_live_rows(session, [row])

        records = session.scalars(select(TennisMatch)).all()
        assert first["persisted_count"] == 1
        assert second["persisted_count"] == 1
        assert first["matches"][0]["id"] == second["matches"][0]["id"]
        assert len(records) == 1
        assert records[0].raw["game_score"] == "15-0"
        assert records[0].status == "live"


def test_live_rows_skip_unmappable_provider_payloads():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)

    with Session(engine) as session, session.begin():
        result = persist_live_rows(session, [{"id": "missing-players"}])

        assert result["fetched_count"] == 1
        assert result["persisted_count"] == 0
        assert result["unmapped_count"] == 1
        assert session.scalars(select(TennisMatch)).all() == []
