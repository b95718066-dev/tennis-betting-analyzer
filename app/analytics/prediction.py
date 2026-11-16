"""Conservative pre-match probability model with sample-shrunk secondary signals.

Elo is the anchor. Form, surface and head-to-head are treated as small,
regularized log-odds adjustments rather than raw percentages averaged together.
The result is an experimental heuristic until a chronological holdout validates
and calibrates it on this project's own historical data.
"""
from __future__ import annotations

from math import exp, log
from typing import Any

from .elo import expected_probability
from .surface import analyze_surface
from .h2h import analyze_h2h


# Deliberately small until walk-forward testing supports stronger coefficients.
SIGNAL_STRENGTH = {
    "form": 0.12,
    "surface": 0.10,
    "h2h": 0.06,
}
PRIORS = {"form": 8.0, "surface": 8.0, "h2h": 6.0}


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    value = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + exp(-value))


def _smoothed_rate(wins: int, matches: int, prior_strength: float) -> float:
    """Beta(0.5*prior, 0.5*prior) shrinkage toward a neutral 50% rate."""
    return (wins + 0.5 * prior_strength) / (matches + prior_strength)


def _valid_results(matches: list[dict[str, Any]]) -> list[str]:
    results = []
    for match in matches:
        result = str(match.get("result", "")).upper()
        if result in {"W", "L"}:
            results.append(result)
    return results


def _signal(name: str, wins: int, matches: int) -> tuple[float, dict[str, Any]]:
    prior = PRIORS[name]
    posterior = _smoothed_rate(wins, matches, prior)
    # Difference from neutral in log-odds; shrinkage is already in posterior.
    strength = SIGNAL_STRENGTH[name]
    contribution = strength * _logit(posterior)
    support = matches / (matches + prior) if matches else 0.0
    return contribution, {
        "wins": wins,
        "matches": matches,
        "smoothed_win_rate": round(posterior, 4),
        "log_odds_contribution": round(contribution, 5),
        "sample_support": round(support, 4),
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
    if not player.strip() or not opponent.strip() or player.strip() == opponent.strip():
        raise ValueError("Two distinct player names are required.")
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("A supported court surface is required.")

    elo_probability = expected_probability(player_rating, opponent_rating)
    log_odds = _logit(elo_probability)

    p_form = _valid_results(player_matches)[-10:]
    o_form = _valid_results(opponent_matches)[-10:]
    p_form_signal, p_form_detail = _signal("form", p_form.count("W"), len(p_form))
    o_form_signal, o_form_detail = _signal("form", o_form.count("W"), len(o_form))
    form_contribution = p_form_signal - o_form_signal
    log_odds += form_contribution

    p_surface = analyze_surface(player_matches, surface)
    o_surface = analyze_surface(opponent_matches, surface)
    p_surface_signal, p_surface_detail = _signal(
        "surface", p_surface["wins"], p_surface["matches"]
    )
    o_surface_signal, o_surface_detail = _signal(
        "surface", o_surface["wins"], o_surface["matches"]
    )
    surface_contribution = p_surface_signal - o_surface_signal
    log_odds += surface_contribution

    h2h = analyze_h2h(h2h_matches, player, opponent, surface)
    h2h_contribution, h2h_detail = _signal("h2h", h2h["wins"], h2h["matches"])
    log_odds += h2h_contribution

    probability = max(0.01, min(0.99, _sigmoid(log_odds)))
    support_values = [
        p_form_detail["sample_support"], o_form_detail["sample_support"],
        p_surface_detail["sample_support"], o_surface_detail["sample_support"],
        h2h_detail["sample_support"],
    ]
    data_coverage_score = round(sum(support_values) / len(support_values), 4)
    coverage = (
        "good" if data_coverage_score >= 0.55
        else "moderate" if data_coverage_score >= 0.25
        else "sparse"
    )

    return {
        "player": player,
        "opponent": opponent,
        "surface": surface.lower(),
        "player_probability": round(probability, 4),
        "opponent_probability": round(1.0 - probability, 4),
        "components": {
            "elo": round(elo_probability, 4),
            "form": round(_sigmoid(form_contribution), 4),
            "surface": round(_sigmoid(surface_contribution), 4),
            "h2h": round(_sigmoid(h2h_contribution), 4),
        },
        "signal_details": {
            "player_recent_form": p_form_detail,
            "opponent_recent_form": o_form_detail,
            "player_surface": p_surface_detail,
            "opponent_surface": o_surface_detail,
            "head_to_head": h2h_detail,
        },
        "data_quality": {
            "coverage": coverage,
            "coverage_score": data_coverage_score,
            "meaning": "Input-history coverage only; not a statistical confidence interval.",
        },
        "methodology": {
            "primary_signal": "overall Elo rating",
            "secondary_signals": "Bayesian-shrunk recent form, surface results and H2H",
            "secondary_signal_weights": SIGNAL_STRENGTH,
            "calibration_status": "not_calibrated",
            "betting_status": "research_only_until_chronological_holdout_validation",
        },
        "model_version": "0.2.0-shrunk-signals",
    }
