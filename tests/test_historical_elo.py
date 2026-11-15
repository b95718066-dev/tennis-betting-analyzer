from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.historical_sync import persist_historical_rows
from app.analytics.historical_elo import historical_elo_ratings
from app.storage.database import create_schema


def _match(mid, date, p1, p2, winner, surface="Hard"):
    return {
        "id": mid, "date": date, "surface": surface,
        "player1": {"id": p1}, "player2": {"id": p2},
        "winner_id": winner,
    }


def test_historical_elo_replays_chronologically_and_uses_surface_minimum():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _match("1", "2026-01-01", "a", "b", "a", "Clay"),
        _match("2", "2026-01-02", "a", "b", "a", "Clay"),
        _match("3", "2026-01-03", "a", "b", "a", "Clay"),
        _match("4", "2026-01-04", "a", "b", "a", "Clay"),
        _match("5", "2026-01-05", "a", "b", "a", "Clay"),
        _match("6", "2026-01-06", "a", "b", "a", "Hard"),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = historical_elo_ratings(
            session, player_id="a", opponent_id="b",
            before="2026-02-01T00:00:00Z", surface="clay",
        )
        assert result["player"]["overall_elo"] > 1500
        assert result["player"]["surface_matches"] == 5
        assert result["player"]["surface_adjustment_used"] is True
        assert result["player_win_probability_elo"] > 0.5
    engine.dispose()


def test_historical_elo_excludes_same_day_and_future_matches():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _match("prior", "2026-05-30", "a", "b", "a"),
        _match("same-day", "2026-06-01", "a", "b", "a"),
        _match("future", "2026-06-02T13:00:00+00:00", "b", "a", "b"),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = historical_elo_ratings(
            session, player_id="a", opponent_id="b",
            before="2026-06-01T12:00:00Z", surface="hard",
        )
        assert result["player"]["overall_matches"] == 1
        assert result["player"]["overall_elo"] > 1500
    engine.dispose()


def test_historical_elo_rejects_invalid_players_surface_and_cutoff():
    import pytest
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="distinct"):
            historical_elo_ratings(session, player_id="a", opponent_id="a",
                                   before="2026-06-01T00:00:00Z", surface="hard")
        with pytest.raises(ValueError, match="surface"):
            historical_elo_ratings(session, player_id="a", opponent_id="b",
                                   before="2026-06-01T00:00:00Z", surface="indoor")
        with pytest.raises(ValueError, match="timezone"):
            historical_elo_ratings(session, player_id="a", opponent_id="b",
                                   before="2026-06-01T00:00:00", surface="hard")
    engine.dispose()
