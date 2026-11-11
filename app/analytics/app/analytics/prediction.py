
from typing import Any

from .elo import expected_probability
from .form import calculate_form
from .surface import analyze_surface
from .h2h import analyze_h2h


WEIGHTS = {
    "elo": 0.55,
    "form": 0.20,
    "surface": 0.15,
    "h2h": 0.10,
}


def predict_match(
    player: str,
    opponent: str,
    player_rating: float,
    opponent_rating: float,
    player_matches: list[dict[str, Any]],
    opponent_matches: list[dict[str, Any]],
    h2h_matches: list[dict[str, Any]],
    surface: str,
) -> dict[str, Any]:

    elo_probability = expected_probability(
        player_rating,
        opponent_rating,
    )

    components = {
        "elo": elo_probability,
    }

    player_form = calculate_form(player_matches)
    opponent_form = calculate_form(opponent_matches)

    if player_form["matches"] and opponent_form["matches"]:
        components["form"] = (
            0.5
            + 0.5 * (
                player_form["form_score"]
                - opponent_form["form_score"]
            )
        )

    player_surface = analyze_surface(
        player_matches, surface
    )
    opponent_surface = analyze_surface(
        opponent_matches, surface
    )

    if player_surface["matches"] and opponent_surface["matches"]:
        components["surface"] = (
            0.5
            + 0.5 * (
                player_surface["win_rate"]
                - opponent_surface["win_rate"]
            )
        )

    h2h = analyze_h2h(
        h2h_matches,
        player,
        opponent,
        surface,
    )

    if h2h["matches"]:
        components["h2h"] = h2h["win_rate"]

    total_weight = sum(
        WEIGHTS[name] for name in components
    )

    probability = sum(
        value * WEIGHTS[name]
        for name, value in components.items()
    ) / total_weight

    probability = max(0.01, min(0.99, probability))

    return {
        "player": player,
        "opponent": opponent,
        "surface": surface,
        "player_probability": round(probability, 4),
        "opponent_probability": round(1 - probability, 4),
        "components": {
            name: round(value, 4)
            for name, value in components.items()
        },
        "model_version": "0.1.0",
    }
