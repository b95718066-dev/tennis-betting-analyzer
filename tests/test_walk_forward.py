import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.analytics.walk_forward import walk_forward_historical_elo
from app.services.historical_sync import persist_historical_rows
from app.storage.database import create_schema


def _row(mid, date, p1, p2, winner, surface="Hard"):
    return {
        "id": mid, "date": date, "surface": surface,
        "player1": {"id": p1}, "player2": {"id": p2},
        "winner_id": winner,
    }


def test_walk_forward_scores_only_after_both_players_have_warmup():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row(str(i), f"2025-01-{i:02d}T10:00:00+00:00", "a", "b", "a")
        for i in range(1, 15)
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = walk_forward_historical_elo(session, min_training_matches=5)
        assert result["coverage"]["out_of_sample_matches"] == 9
        assert result["coverage"]["skipped_warmup"] == 5
        assert result["metrics"]["matches"] == 9
        assert result["sample_predictions"][0]["prior_matches_player_1"] == 5
        assert result["sample_predictions"][0]["player_1_win_probability"] > 0.5
    engine.dispose()


def test_walk_forward_same_timestamp_predictions_are_frozen_before_updates():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row(str(i), f"2025-01-{i:02d}T10:00:00+00:00", "a", "ֻǱ˞