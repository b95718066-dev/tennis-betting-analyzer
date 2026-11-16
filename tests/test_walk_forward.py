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
        assert scored["same-a"]["player_1_pre_match_elo"] == scored["same-b"]["player_2_pre_match_elo"]
        assert scored["same-a"]["player_1_win_probability"] == pytest.approx(1.0 - scored["same-b"]["player_1_win_probability"], abs=1e-6)
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
        assert result["odds_backtest"]["matches_with_paired_closing_odds"] == 0
        assert result["odds_backtest"]["bets_placed"] == 0
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



def test_walk_forward_includes_valuebetennis_utc_timestamps_and_closing_odds_roi():
    from app.storage.models import HistoricalMatchSnapshot

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        # Date-only Cito results remain unscored; Valuebetennis publishes UTC
        # match date-times without a trailing timezone marker.
        session.add(HistoricalMatchSnapshot(
            provider="cito", external_id="cito-date-only",
            match_date="2025-01-01", tour="ATP", surface="hard",
            player_1_id="atp_x", player_2_id="atp_y", winner_id="atp_x",
            stats_available=False, raw={},
        ))
        for day in range(1, 6):
            session.add(HistoricalMatchSnapshot(
                provider="valuebetennis", external_id=f"vb-clay-{day}",
                match_date=f"2025-01-{day:02d} 12:00:00",
                tour="ATP", surface="clay",
                player_1_id="vb_a", player_2_id="vb_b", winner_id="vb_a",
                stats_available=False, raw={},
            ))
        session.add(HistoricalMatchSnapshot(
            provider="valuebetennis", external_id="vb-hard-target",
            match_date="2025-01-06 12:00:00",
            tour="ATP", surface="hard",
            player_1_id="vb_a", player_2_id="vb_b", winner_id="vb_a",
            stats_available=False,
            raw={"cote1_cloture": "4.0", "cote2_cloture": "1.2"},
        ))

        result = walk_forward_historical_elo(
            session, min_training_matches=5, surface="hard", min_ev=0.03
        )

        assert result["coverage"]["rows_by_source"] == {
            "cito": 1, "valuebetennis": 6
        }
        assert result["coverage"]["skipped_date_only"] == 1
        assert result["coverage"]["skipped_surface_filter"] == 5
        assert result["coverage"]["out_of_sample_matches"] == 1
        prediction = result["sample_predictions"][0]
        assert prediction["match_id"] == "vb-hard-target"
        assert prediction["provider"] == "valuebetennis"
        assert prediction["match_date"] == "2025-01-06T12:00:00+00:00"
        assert prediction["prior_matches_player_1"] == 5
        assert prediction["closing_odds"]["player_1"] == 4.0
        assert prediction["simulated_bet"]["selection"] == "player_1"
        assert result["odds_backtest"]["matches_with_paired_closing_odds"] == 1
        assert result["odds_backtest"]["bets_placed"] == 1
        assert result["odds_backtest"]["net_profit_units"] == pytest.approx(3.0)
        assert result["odds_backtest"]["roi_pct"] == pytest.approx(300.0)
    engine.dispose()
