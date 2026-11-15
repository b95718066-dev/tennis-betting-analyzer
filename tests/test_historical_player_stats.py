from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.historical_sync import persist_historical_rows
from app.analytics.historical_player_stats import historical_player_stats
from app.storage.database import create_schema


def _row(mid, date, surface, p1, p2, stats):
    return {
        "id": mid, "date": date, "surface": surface,
        "player1": {"id": p1}, "player2": {"id": p2},
        "winner_id": p1, "stats_available": True, "stats": stats,
    }


def test_historical_stats_aggregate_player_side_and_surface():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row("clay", "2026-05-01", "Clay", "p1", "p2", [
            {"key": "service_points_won", "player1": {"numerator": 30, "denominator": 45},
             "player2": {"numerator": 20, "denominator": 40}},
            {"key": "return_points_won", "player1": {"numerator": 18, "denominator": 40},
             "player2": {"numerator": 15, "denominator": 45}},
        ]),
        _row("hard", "2026-05-02", "Hard", "p3", "p1", [
            {"key": "service_points_won", "player1": {"numerator": 25, "denominator": 40},
             "player2": {"numerator": 32, "denominator": 50}},
        ]),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        clay = historical_player_stats(session, player_id="p1", before="2026-06-01T00:00:00Z", surface="clay")
        all_court = historical_player_stats(session, player_id="p1", before="2026-06-01T00:00:00Z")
        assert clay["metrics"]["service_points_won"]["numerator"] == 30
        assert clay["metrics"]["service_points_won"]["denominator"] == 45
        assert all_court["metrics"]["service_points_won"]["numerator"] == 62
        assert all_court["metrics"]["service_points_won"]["denominator"] == 95
        assert all_court["metrics"]["return_points_won"]["numerator"] == 18
    engine.dispose()


def test_historical_stats_excludes_cutoff_day_and_invalid_or_missing_pairs():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row("prior", "2026-05-31", "Clay", "p1", "p2", [
            {"key": "service_points_won", "player1": {"numerator": 10, "denominator": 20}},
            {"key": "return_points_won", "player1": {"numerator": 4, "denominator": 20}},
        ]),
        _row("same-day", "2026-06-01", "Clay", "p1", "p3", [
            {"key": "service_points_won", "player1": {"numerator": 99, "denominator": 100}},
        ]),
        _row("invalid", "2026-05-30", "Clay", "p1", "p4", [
            {"key": "service_points_won", "player1": {"numerator": 50, "denominator": 20}},
        ]),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = historical_player_stats(session, player_id="p1", before="2026-06-01T12:00:00Z")
        assert result["metrics"]["service_points_won"]["numerator"] == 10
        assert result["metrics"]["service_points_won"]["denominator"] == 20
        assert result["metrics"]["return_points_won"]["rate"] == 0.2
    engine.dispose()


def test_zero_denominator_and_percentage_only_values_are_ignored():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [_row("zero", "2026-05-01", "Hard", "p1", "p2", [
        {"key": "service_points_won", "player1": {"numerator": 0, "denominator": 0, "pct": None}},
        {"key": "return_points_won", "player1": {"pct": 45.0, "display": "45%"}},
    ])]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = historical_player_stats(session, player_id="p1", before="2026-06-01T00:00:00Z")
        assert result["metrics"]["service_points_won"]["rate"] is None
        assert result["metrics"]["return_points_won"]["rate"] is None
    engine.dispose()
