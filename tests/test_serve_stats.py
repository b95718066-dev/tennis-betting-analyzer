import pytest

from app.analytics.serve_stats import normalize_live_serve_rates


def test_normalizes_cito_live_service_point_counts_for_both_players():
    result = normalize_live_serve_rates({
        "stats_available": True,
        "stats": {
            "player1": {"service_points_won": 24, "service_points_played": 35},
            "player2": {"serve_points_won": 18, "serve_points": 30},
        },
    })

    assert result["ready_for_live_model"] is True
    assert result["serve_point_win_rates"][0]["serve_point_win_rate"] == pytest.approx(24 / 35)
    assert result["serve_point_win_rates"][1]["serve_point_win_rate"] == pytest.approx(0.6)


def test_reads_documented_numerator_denominator_stat_objects():
    result = normalize_live_serve_rates({
        "stats_available": True,
        "stats": {
            "player1": {"service_points": {"numerator": 20, "denominator": 31}},
            "player2": {"service_points": {"numerator": 19, "denominator": 29}},
        },
    })

    assert result["ready_for_live_model"] is True
    assert result["serve_point_win_rates"][0]["service_points_played"] == 31


def test_does_not_invent_rates_when_stats_are_missing_or_invalid():
    result = normalize_live_serve_rates({
        "stats_available": False,
        "stats_unavailable_code": "stats_pending",
        "stats": {"player1": {"service_points_won": 5, "service_points_played": 0}},
    })

    assert result["ready_for_live_model"] is False
    assert result["stats_unavailable_code"] == "stats_pending"
    assert result["serve_point_win_rates"][0]["serve_point_win_rate"] is None
    assert result["serve_point_win_rates"][1]["serve_point_win_rate"] is None


def test_rejects_won_count_larger_than_played_count():
    result = normalize_live_serve_rates({
        "stats": {
            "player1": {"service_points_won": 40, "service_points_played": 30},
            "player2": {"service_points_won": 20, "service_points_played": 30},
        },
    })

    assert result["ready_for_live_model"] is False
