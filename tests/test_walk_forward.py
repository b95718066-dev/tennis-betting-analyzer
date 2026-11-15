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
        _row(str(i), f"2025-01-{i:02d}T10:00:00+00:00", "a", "b", "a")
        for i in range(1, 7)
    ]
    rows += [
        _row("same-a", "2025-01-07T10:00:00+00:00", "a", "b", "a"),
        _row("same-b", "2025-01-07T10:00:00+00:00", "b", "a", "a"),
        _row("later", "2025-01-08T10:00:00+00:00", "a", "b", "a"),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = walk_forward_historical_elo(session, min_training_matches=5)
        scored = {x["match_id"]: x for x in result["sample_predictions"]}
        assert scored["same-a"]["player_1_pre_match_elo"] == scored["same-b"]["player_1_pre_match_elo"]
        assert scored["same-a"]["player_2_pre_match_elo"] == scored["same-b"]["player_2_pre_match_elo"]
        assert scored["same-a"]["player_1_win_probability"] == pytest.approx(scored["same-b"]["player_1_win_probability"], abs=1e-6)
        assert scored["later"]["player_1_pre_match_elo"] > scored["same-a"]["player_1_pre_match_elo"]
    engine.dispose()


def test_walk_forward_never_scores_date_only_rows_and_reports_no_roi_without_odds():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row(str(i), f"2025-02-{i:02d}", "a", "b", "a")
        for i in range(1, 9)
    ]
    rows.append(_row("timed", "2025-02-09T12:00:00+00:00", "a", "b", "a"))
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = walk_forward_historical_elo(session, min_training_matches=5)
        assert result["coverage"]["skipped_date_only"] == 8
        assert result["coverage"]["out_of_sample_matches"] == 1
        assert result["metrics"]["matches"] == 1
        assert "odds_roi" in result["methodology"]
        assert "not calculated" in result["methodology"]["odds_roi"]
    engine.dispose()


def test_walk_forward_validates_parameters_and_handles_no_sample():
    import pytest
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="min_training"):
            walk_forward_historical_elo(session, min_training_matches=0)
        with pytest.raises(ValueError, match="surface"):
            walk_forward_historical_elo(session, surface="indoor")
        result = walk_forward_historical_elo(session)
        assert result["metrics"] is None
        assert result["calibration"] == []
    engine.dispose()


def test_platt_calibration_fits_only_when_both_outcomes_and_minimum_sample_exist():
    from app.analytics.walk_forward import _apply_platt, _fit_platt
    assert _fit_platt([(0.6, True)] * 29) is None
    assert _fit_platt([(0.6, True)] * 40) is None
    sample = [(0.65, i % 3 != 0) for i in range(90)]
    parameters = _fit_platt(sample)
    assert parameters is not None
    calibrated = _apply_platt(0.65, parameters)
    assert 0.0 < calibrated < 1.0


def test_platt_calibration_is_bounded_for_extreme_probabilities():
    from app.analytics.walk_forward import _apply_platt
    assert 0.0 < _apply_platt(0.000001, {"slope": 1.0, "intercept": 0.0}) < 1.0
    assert 0.0 < _apply_platt(0.999999, {"slope": 1.0, "intercept": 0.0}) < 1.0
