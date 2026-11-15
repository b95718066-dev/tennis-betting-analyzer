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


def test_normalizes_cito_catalog_rows_with_per_player_numerator_denominator():
    result = normalize_live_serve_rates({
        "stats_available": True,
        "stats": [
            {
                "key": "service_points_won",
                "player1": {"numerator": 27, "denominator": 39, "pct": 69.2},
                "player2": {"numerator": 19, "denominator": 31, "pct": 61.3},
            }
        ],
    })

    assert result["ready_for_live_model"] is True
    assert result["serve_point_win_rates"][0]["serve_point_win_rate"] == pytest.approx(27 / 39)
    assert result["serve_point_win_rates"][1]["serve_point_win_rate"] == pytest.approx(19 / 31)


def test_normalizes_cito_flat_live_board_service_points_rows_by_side():
    result = normalize_live_serve_rates({
        "stats_available": True,
        "stats": [
            {"key": "service_points_won", "side": 1, "numerator": 18, "denominator": 29, "pct": 62.1},
            {"key": "total_points_won", "side": 1, "numerator": 36, "denominator": 58, "pct": 62.1},
            {"key": "service_points_won", "side": 2, "numerator": 11, "denominator": 29, "pct": 37.9},
        ],
    })

    assert result["ready_for_live_model"] is True
    assert result["serve_point_win_rates"][0]["serve_point_win_rate"] == pytest.approx(18 / 29)
    assert result["serve_point_win_rates"][1]["serve_point_win_rate"] == pytest.approx(11 / 29)
