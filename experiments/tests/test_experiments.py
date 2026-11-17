from experiments.model_01_market_anchored.model import MarketAnchoredLogit
from experiments.model_03_clv_lab.clv import calculate_clv
from app.analytics.live_probability import calculate_live_probabilities


def test_market_anchored_model_learns_correction():
    rows = []
    for index in range(40):
        feature = -1.0 if index < 20 else 1.0
        rows.append({
            "market_probability": 0.5,
            "features": {"signal": feature},
            "outcome": index >= 20,
        })
    model = MarketAnchoredLogit.fit(rows, learning_rate=0.05, epochs=1500, l2=0.1)
    low = model.predict_probability({
        "market_probability": 0.5, "features": {"signal": -1.0}
    })
    high = model.predict_probability({
        "market_probability": 0.5, "features": {"signal": 1.0}
    })
    assert low < 0.5 < high
    assert model.to_dict()["training_rows"] == 40


def test_clv_positive_when_selected_price_shortens():
    result = calculate_clv(
        entry_odds={"p1": 2.10, "p2": 1.80},
        closing_odds={"p1": 1.95, "p2": 1.92},
        selection="p1",
    )
    assert result["probability_clv"] > 0
    assert result["odds_clv"] > 0
    assert result["beat_closing_price"] is True


def test_live_markov_returns_complementary_probabilities():
    result = calculate_live_probabilities(
        sets_player_1=0, sets_player_2=0,
        games_player_1=0, games_player_2=0,
        points_player_1=0, points_player_2=0,
        server=1, best_of=3,
        player_1_serve_point_win=0.67,
        player_2_serve_point_win=0.64,
    )
    p1 = result["player_1_match_probability"]
    p2 = result["player_2_match_probability"]
    assert 0 < p1 < 1
    assert abs(p1 + p2 - 1.0) < 1e-6
