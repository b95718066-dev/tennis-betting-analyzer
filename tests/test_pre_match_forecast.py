import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.historical_sync import persist_historical_rows
from app.services.pre_match_forecast import generate_pre_match_forecast
from app.storage.database import create_schema


def _row(mid, when, p1, p2, winner, surface="Hard"):
    return {
        "id": mid, "date": when, "surface": surface,
        "player1": {"id": p1}, "player2": {"id": p2},
        "winner_id": winner,
    }


def test_pre_match_forecast_builds_from_history_and_excludes_future_rows():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row("1", "2025-01-01T10:00:00+00:00", "A", "B", "A"),
        _row("2", "2025-01-02T10:00:00+00:00", "A", "C", "A"),
        _row("3", "2025-01-03T10:00:00+00:00", "D", "B", "D"),
        _row("4", "2025-01-04T10:00:00+00:00", "A", "B", "B"),
        _row("future", "2025-01-05T15:00:00+00:00", "A", "B", "A"),
        _row("date-only-same-day", "2025-01-05", "A", "B", "A"),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = generate_pre_match_forecast(
            session,
            player_1_id="A",
            player_2_id="B",
            surface="hard",
            before="2025-01-05T12:00:00+00:00",
            player_1_odds=1.8,
            player_2_odds=2.2,
        )
        assert result["forecast_type"] == "pre_match_historical"
        assert result["history_coverage"]["player_1_matches"] == 3
        assert result["history_coverage"]["player_2_matches"] == 3
        assert result["history_coverage"]["head_to_head_matches"] == 2
        assert result["history_coverage"]["stored_rows_used"] == 4
        assert result["as_of"] == "2025-01-05T12:00:00+00:00"
        assert "betting" in result
        assert sum(result["betting"]["market_probability_without_margin"].values()) == pytest.approx(1.0)
        assert result["methodology"]["calibration_status"] == "not_calibrated"
    engine.dispose()


def test_pre_match_forecast_rejects_naive_cutoff_and_one_sided_odds():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="timezone"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="hard", before="2025-01-05T12:00:00",
            )
        with pytest.raises(ValueError, match="Both player odds"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="hard", before="2025-01-05T12:00:00+00:00",
                player_1_odds=1.8,
            )
    engine.dispose()


def test_pre_match_forecast_rejects_invalid_player_pair_and_surface():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="distinct"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="A",
                surface="hard", before="2025-01-05T12:00:00+00:00",
            )
        with pytest.raises(ValueError, match="surface"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="indoor", before="2025-01-05T12:00:00+00:00",
            )
    engine.dispose()
