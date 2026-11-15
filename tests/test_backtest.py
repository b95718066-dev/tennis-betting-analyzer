import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.analytics.backtest import BacktestBet, chronological_split, run_backtest


def bet(index, *, probability=0.6, odds=2.0, outcome=True, stake=10.0):
    placed = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    return BacktestBet(
        event_id=f"match-{index}",
        selection="Player A",
        probability=probability,
        odds_decimal=odds,
        outcome=outcome,
        stake=stake,
        placed_at=placed,
        settled_at=placed + timedelta(hours=3),
    )


def test_backtest_calculates_profit_roi_yield_and_hit_rate():
    result = run_backtest([
        bet(0, outcome=True),
        bet(1, outcome=False),
    ], initial_bankroll=100.0)
    summary = result["summary"]
    assert summary["bets"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["net_profit"] == pytest.approx(0.0)
    assert summary["roi"] == pytest.approx(0.0)
    assert summary["yield"] == pytest.approx(0.0)
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["final_bankroll"] == pytest.approx(100.0)


def test_backtest_is_ordered_by_placement_time_and_tracks_drawdown():
    result = run_backtest([
        bet(2, outcome=True),
        bet(0, outcome=False),
        bet(1, outcome=False),
    ], initial_bankroll=100.0)
    curve = result["equity_curve"]
    assert curve[1]["bankroll"] == pytest.approx(90.0)
    assert curve[2]["bankroll"] == pytest.approx(80.0)
    assert curve[3]["bankroll"] == pytest.approx(90.0)
    assert result["summary"]["max_drawdown"] == pytest.approx(20.0)
    assert result["summary"]["max_drawdown_pct"] == pytest.approx(0.2)


def test_backtest_skips_bets_that_exceed_available_bankroll():
    result = run_backtest([
        bet(0, outcome=False, stake=60.0),
        bet(1, outcome=True, stake=60.0),
    ], initial_bankroll=100.0)
    assert result["summary"]["bets"] == 1
    assert result["summary"]["skipped_insufficient_bankroll"] == 1
    assert result["summary"]["final_bankroll"] == pytest.approx(40.0)


def test_backtest_reports_brier_log_loss_and_calibration():
    result = run_backtest([
        bet(0, probability=0.8, outcome=True),
        bet(1, probability=0.2, outcome=False),
    ], calibration_bins=5)
    assert result["summary"]["brier_score"] == pytest.approx(0.04)
    assert result["summary"]["log_loss"] > 0
    assert sum(row["count"] for row in result["calibration"]) == 2
    assert len(result["calibration"]) == 2


def test_backtest_requires_aware_timestamps_and_settlement_after_placement():
    with pytest.raises(ValidationError):
        BacktestBet(
            event_id="m1", selection="A", probability=0.5, odds_decimal=2.0,
            outcome=True, placed_at=datetime(2026, 1, 1),
            settled_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    item = bet(0).model_copy(update={"settled_at": bet(0).placed_at})
    with pytest.raises(ValueError, match="settle after"):
        run_backtest([item])


def test_backtest_rejects_empty_input_and_invalid_bankroll():
    with pytest.raises(ValueError):
        run_backtest([])
    with pytest.raises(ValueError):
        run_backtest([bet(0)], initial_bankroll=math.inf)


def test_chronological_split_keeps_future_records_out_of_training():
    rows = [
        {"placed_at": datetime(2026, 1, day, tzinfo=timezone.utc), "id": day}
        for day in (3, 1, 2, 4, 5)
    ]
    train, test = chronological_split(rows, train_fraction=0.6)
    assert [row["id"] for row in train] == [1, 2, 3]
    assert [row["id"] for row in test] == [4, 5]


def test_chronological_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        chronological_split([{"placed_at": 1}, {"placed_at": 2}], train_fraction=1.0)



def test_backtest_api_returns_metrics():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post("/backtest", json={
        "initial_bankroll": 100.0,
        "calibration_bins": 5,
        "bets": [{
            "event_id": "match-api-1",
            "selection": "Player A",
            "probability": 0.7,
            "odds_decimal": 1.8,
            "outcome": True,
            "stake": 10.0,
            "placed_at": "2026-01-01T10:00:00Z",
            "settled_at": "2026-01-01T12:00:00Z"
        }]
    })
    assert response.status_code == 200
    assert response.json()["summary"]["net_profit"] == pytest.approx(8.0)
    assert response.json()["summary"]["roi"] == pytest.approx(0.08)


def test_backtest_api_rejects_naive_timestamps():
    from fastapi.testclient import TestClient
    from app.main import app

    response = TestClient(app).post("/backtest", json={
        "bets": [{
            "event_id": "match-api-2",
            "selection": "Player A",
            "probability": 0.5,
            "odds_decimal": 2.0,
            "outcome": False,
            "placed_at": "2026-01-01T10:00:00",
            "settled_at": "2026-01-01T12:00:00Z"
        }]
    })
    assert response.status_code == 422
