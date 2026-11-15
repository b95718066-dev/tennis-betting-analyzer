"""Elo rating calculations."""

from math import isfinite, pow

DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0


def _validate_rating(rating: float) -> None:
    if not isfinite(rating):
        raise ValueError("Ratings must be finite numbers.")


def expected_probability(player_rating: float, opponent_rating: float) -> float:
    """Calculate expected match-win probability using the Elo formula."""
    _validate_rating(player_rating)
    _validate_rating(opponent_rating)
    exponent = (opponent_rating - player_rating) / 400.0
    return 1.0 / (1.0 + pow(10.0, exponent))


def update_elo(
    player_rating: float,
    opponent_rating: float,
    actual_result: float,
    k_factor: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Update both players' ratings after a completed match."""
    _validate_rating(player_rating)
    _validate_rating(opponent_rating)
    if not isfinite(actual_result) or not 0.0 <= actual_result <= 1.0:
        raise ValueError("Result must be finite and between 0 and 1.")
    if not isfinite(k_factor) or k_factor <= 0.0:
        raise ValueError("K-factor must be finite and positive.")

    player_expected = expected_probability(player_rating, opponent_rating)
    new_player = player_rating + k_factor * (actual_result - player_expected)
    new_opponent = opponent_rating + k_factor * ((1.0 - actual_result) - (1.0 - player_expected))
    return new_player, new_opponent


def surface_adjusted_rating(
    overall_rating: float,
    surface_rating: float,
    weight: float = 0.35,
) -> float:
    """Blend overall and surface-specific ratings."""
    _validate_rating(overall_rating)
    _validate_rating(surface_rating)
    if not isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("Weight must be finite and between 0 and 1.")
    return overall_rating * (1.0 - weight) + surface_rating * weight
