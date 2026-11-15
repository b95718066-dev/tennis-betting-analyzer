import pytest

from app.analytics.live_probability import calculate_live_probabilities


def base_state(**overrides):
    values = {
        "sets_player_1": 0,
        "sets_player_2": 0,
        "games_player_1": 0,
        "games_player_2": 0,
        "points_player_1": 0,
        "points_player_2": 0,
        "server": 1,
        "best_of": 3,
        "player_1_serve_point_win": 0.64,
        "player_2_serve_point_win": 0.62,
    }
    values.update(overrides)
    return values


def test_live_probability_returns_complementary_match_set_and_game_prices():
    result = calculate_live_probabilities(**base_state())

    assert result["player_1_match_probability"] + result["player_2_match_probability"] == pytest.approx(1.0)
    assert result["player_1_current_set_probability"] + result["player_2_current_set_probability"] == pytest.approx(1.0)
    assert result["player_1_current_game_probability"] + result["player_2_current_game_probability"] == pytest.approx(1.0)
    assert result["model"]["method"] == "point_state_markov"
    assert result["model"]["calibrated"] is False


def test_live_probability_prices_current_tiebreak_state():
    result = calculate_live_probabilities(**base_state(
        games_player_1=6,
        games_player_2=6,
        server=2,
        tiebreak_first_server=1,
        tiebreak_points_player_1=5,
        tiebreak_points_player_2=4,
    ))

    assert 0 < result["player_1_current_game_probability"] < 1
    assert result["player_1_match_probability"] + result["player_2_match_probability"] == pytest.approx(1.0)


def test_live_probability_adds_margin_removed_market_and_ev():
    result = calculate_live_probabilities(**base_state(
        player_1_odds=1.80,
        player_2_odds=2.10,
        min_ev=0.02,
    ))

    assert sum(result["market"]["fair_probability"].values()) == pytest.approx(1.0)
    assert result["market"]["overround"] > 0
    assert isinstance(result["market"]["player_1_value_bet"], bool)


def test_live_probability_rejects_partial_odds_pair():
    with pytest.raises(ValueError, match="Both player odds"):
        calculate_live_probabilities(**base_state(player_1_odds=1.9))


@pytest.mark.parametrize("invalid", [
    {"server": 3},
    {"best_of": 4},
    {"player_1_serve_point_win": 1.0},
    {"points_player_1": 4, "points_player_2": 1},
    {"games_player_1": 6, "games_player_2": 6, "tiebreak_points_player_1": -1},
])
def test_live_probability_rejects_invalid_score_or_parameters(invalid):
    with pytest.raises(ValueError):
        calculate_live_probabilities(**base_state(**invalid))

def test_live_probability_handles_deuce_in_extended_tiebreak():
    result = calculate_live_probabilities(**base_state(
        games_player_1=6,
        games_player_2=6,
        server=1,
        tiebreak_first_server=1,
        tiebreak_points_player_1=6,
        tiebreak_points_player_2=6,
    ))

    assert 0 < result["player_1_current_game_probability"] < 1
    assert result["player_1_match_probability"] + result["player_2_match_probability"] == pytest.approx(1.0)


def test_live_probability_requires_separate_tiebreak_score_at_six_all():
    with pytest.raises(ValueError, match="regular point scores must be zero"):
        calculate_live_probabilities(**base_state(
            games_player_1=6,
            games_player_2=6,
            points_player_1=1,
            server=1,
            tiebreak_first_server=1,
        ))
