import math

import pytest

from app.analytics.elo import expected_probability, surface_adjusted_rating, update_elo
from app.analytics.form import calculate_form
from app.analytics.h2h import analyze_h2h
from app.analytics.prediction import predict_match
from app.analytics.probability import expected_value, implied_probability, remove_margin, value_bet
from app.analytics.surface import analyze_surface


def test_elo_is_symmetric_and_equal_ratings_are_fifty_fifty():
    assert expected_probability(1500, 1500) == pytest.approx(0.5)
    assert expected_probability(1600, 1400) + expected_probability(1400, 1600) == pytest.approx(1.0)


def test_elo_update_preserves_combined_rating():
    before = 1500 + 1500
    winner, loser = update_elo(1500, 1500, 1.0)
    assert winner > 1500 > loser
    assert winner + loser == pytest.approx(before)


@pytest.mark.parametrize("ratings", [(math.nan, 1500), (1500, math.inf)])
def test_elo_rejects_non_finite_ratings(ratings):
    with pytest.raises(ValueError):
        expected_probability(*ratings)


def test_surface_rating_weight_boundaries():
    assert surface_adjusted_rating(1500, 1600, 0) == pytest.approx(1500)
    assert surface_adjusted_rating(1500, 1600, 1) == pytest.approx(1600)


def test_implied_probability_and_margin_removal():
    assert implied_probability(2.0) == pytest.approx(0.5)
    fair = remove_margin({"A": 1.8, "B": 2.2})
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["A"] > fair["B"]


@pytest.mark.parametrize("odds", [1.0, 0.9, math.nan, math.inf])
def test_invalid_odds_are_rejected(odds):
    with pytest.raises(ValueError):
        implied_probability(odds)


def test_margin_removal_requires_two_outcomes():
    with pytest.raises(ValueError):
        remove_margin({"A": 2.0})


def test_expected_value_and_value_bet_threshold():
    assert expected_value(0.60, 2.0) == pytest.approx(0.20)
    assert value_bet(0.60, 2.0, 0.10)
    assert not value_bet(0.50, 2.0, 0.03)


def test_form_weights_recent_matches_more_heavily():
    form = calculate_form([
        {"result": "W"},
        {"result": "L"},
        {"result": "W"},
    ])
    assert form["matches"] == 3
    assert form["win_rate"] == pytest.approx(2 / 3)
    assert form["form_score"] == pytest.approx(4 / 6)
    assert form["streak"] == 1


def test_surface_analysis_filters_matches():
    stats = analyze_surface([
        {"surface": "Hard", "result": "W"},
        {"surface": "Clay", "result": "L"},
        {"surface": "hard", "result": "L"},
    ], "HARD")
    assert stats["matches"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)


def test_h2h_counts_wins_from_requested_players_perspective():
    stats = analyze_h2h([
        {"player": "A", "opponent": "B", "winner": "A", "surface": "Hard"},
        {"player": "B", "opponent": "A", "winner": "B", "surface": "Clay"},
    ], "A", "B")
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["win_rate"] == pytest.approx(0.5)


def test_prediction_returns_complementary_probabilities():
    result = predict_match(
        player="A",
        opponent="B",
        player_rating=1600,
        opponent_rating=1500,
        player_matches=[{"result": "W", "surface": "Hard"}],
        opponent_matches=[{"result": "L", "surface": "Hard"}],
        h2h_matches=[],
        surface="Hard",
    )
    assert result["player_probability"] + result["opponent_probability"] == pytest.approx(1.0)
    assert 0.0 <= result["player_probability"] <= 1.0


def test_prediction_uses_elo_as_anchor_when_secondary_history_is_missing():
    result = predict_match(
        player="A", opponent="B", player_rating=1600, opponent_rating=1500,
        player_matches=[], opponent_matches=[], h2h_matches=[], surface="Hard",
    )
    assert result["player_probability"] == pytest.approx(expected_probability(1600, 1500), abs=0.0001)
    assert result["data_quality"]["coverage"] == "sparse"
    assert result["methodology"]["calibration_status"] == "not_calibrated"


def test_single_head_to_head_result_is_shrunk_and_cannot_dominate_elo():
    baseline = predict_match(
        player="A", opponent="B", player_rating=1600, opponent_rating=1500,
        player_matches=[], opponent_matches=[], h2h_matches=[], surface="Hard",
    )
    with_one_h2h = predict_match(
        player="A", opponent="B", player_rating=1600, opponent_rating=1500,
        player_matches=[], opponent_matches=[],
        h2h_matches=[{"player": "A", "opponent": "B", "winner": "A", "surface": "Hard"}],
        surface="Hard",
    )
    assert with_one_h2h["player_probability"] > baseline["player_probability"]
    assert with_one_h2h["player_probability"] - baseline["player_probability"] < 0.01
    assert with_one_h2h["signal_details"]["head_to_head"]["sample_support"] < 0.2


def test_form_and_surface_adjustments_are_sample_shrunk_and_probabilities_sum_to_one():
    result = predict_match(
        player="A", opponent="B", player_rating=1500, opponent_rating=1500,
        player_matches=[{"result": "W", "surface": "Hard"}],
        opponent_matches=[{"result": "L", "surface": "Hard"}],
        h2h_matches=[], surface="Hard",
    )
    assert result["player_probability"] > 0.5
    assert result["player_probability"] + result["opponent_probability"] == pytest.approx(1.0)
    assert result["signal_details"]["player_surface"]["sample_support"] < 0.2
    assert result["data_quality"]["meaning"].startswith("Input-history coverage")
