from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.historical_sync import persist_historical_rows, player_history_features
from app.storage.database import create_schema


def test_historical_rows_upsert_and_late_stats_refresh():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    base = {
        "canonical_id": "m-1",
        "date": "2026-06-01",
        "tour": "ATP",
        "surface": "Clay",
        "player1": {"canonical_id": "p1"},
        "player2": {"canonical_id": "p2"},
        "winner": {"canonical_id": "p1"},
        "stats_available": False,
        "stats_unavailable_code": "stats_pending",
    }
    with Session(engine) as session, session.begin():
        first = persist_historical_rows(session, [base])
        refreshed = {**base, "stats_available": True, "stats_unavailable_code": None,
                     "stats": [{"name": "service_points", "numerator": 30, "denominator": 45}]}
        second = persist_historical_rows(session, [refreshed])
        assert first == {"fetched": 1, "inserted": 1, "updated": 0, "skipped": 0}
        assert second == {"fetched": 1, "inserted": 0, "updated": 1, "skipped": 0}
        features = player_history_features(
            session, player_id="p1", before="2026-06-02T00:00:00Z"
        )
        assert features["matches_considered"] == 1
        assert features["wins"] == 1
        assert features["win_rate"] == 1.0
    engine.dispose()


def test_date_only_same_day_match_is_excluded_from_as_of_features():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        {
            "id": "same-day", "date": "2026-06-01", "tour": "ATP",
            "player1": {"id": "p1"}, "player2": {"id": "p2"},
            "winner_id": "p1",
        },
        {
            "id": "prior-day", "date": "2026-05-31", "tour": "ATP",
            "player1": {"id": "p1"}, "player2": {"id": "p3"},
            "winner_id": "p1",
        },
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = player_history_features(
            session, player_id="p1", before="2026-06-01T12:00:00Z"
        )
        assert result["matches_considered"] == 1
        assert result["wins"] == 1
    engine.dispose()


def test_timestamped_match_after_cutoff_is_excluded_and_invalid_cutoff_rejected():
    import pytest

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, [{
            "id": "future", "date": "2026-06-01T13:00:00+00:00",
            "player1": {"id": "p1"}, "player2": {"id": "p2"},
            "winner_id": "p1",
        }])
        result = player_history_features(
            session, player_id="p1", before="2026-06-01T12:00:00Z"
        )
        assert result["matches_considered"] == 0
        with pytest.raises(ValueError, match="timezone"):
            player_history_features(session, player_id="p1", before="2026-06-01T12:00:00")
    engine.dispose()


def test_rows_without_ids_are_skipped():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        result = persist_historical_rows(session, [{"date": "2026-01-01"}])
        assert result["skipped"] == 1
    engine.dispose()



def test_verified_start_time_is_preserved_but_unknown_start_time_stays_date_only():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        {
            "id": "known-time", "date": "2026-06-01",
            "starts_at": "2026-06-01T14:30:00+00:00",
            "start_time_known": True, "start_time_kind": "scheduled",
            "player1": {"id": "p1"}, "player2": {"id": "p2"},
            "winner_id": "p1",
        },
        {
            "id": "unknown-time", "date": "2026-06-01",
            "starts_at": "2026-06-01T14:30:00+00:00",
            "start_time_known": False,
            "player1": {"id": "p1"}, "player2": {"id": "p3"},
            "winner_id": "p1",
        },
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        from app.storage.models import HistoricalMatchSnapshot
        known = session.query(HistoricalMatchSnapshot).filter_by(external_id="known-time").one()
        unknown = session.query(HistoricalMatchSnapshot).filter_by(external_id="unknown-time").one()
        assert known.match_date == "2026-06-01T14:30:00+00:00"
        assert unknown.match_date == "2026-06-01"
    engine.dispose()
