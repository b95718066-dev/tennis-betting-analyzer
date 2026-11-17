from datetime import date

from experiments.historical_backtest import _day, _score_metrics, _summarize_clv


def test_day_parser_rejects_missing_or_invalid_dates():
    assert _day("2026-11-17 12:30:00") == date(2026, 11, 17)
    assert _day("not-a-date") is None
    assert _day(None) is None


def test_metrics_report_brier_skill_against_half_probability():
    metrics = _score_metrics([(0.9, True), (0.1, False), (0.8, True), (0.2, False)])
    assert metrics is not None
    assert metrics["matches"] == 4
    assert metrics["brier_score"] < 0.25
    assert metrics["brier_skill_vs_50pct"] > 0


def test_clv_summary_does_not_claim_actual_bets():
    summary = _summarize_clv([
        {"probability_clv": 0.02, "odds_clv": 0.04, "beat_closing_price": True},
        {"probability_clv": -0.02, "odds_clv": -0.03, "beat_closing_price": False},
    ])
    assert summary is not None
    assert summary["selection_observations"] == 2
    assert summary["match_pairs"] == 1
    assert summary["timestamp_validation"] == "not_verified"
    assert "no retrospective winner selection" in summary["method"]
