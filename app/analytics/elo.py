
from math import pow


DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0


def expected_probability(
    player_rating: float,
    opponent_rating: float,
) -> float:
    """Calculate expected match-win probability using Elo."""
    exponent = (opponent_rating - player_rating) / 400.0
    return 1.0 / (1.0 + pow(10.0, exponent))


def update_elo(
    player_rating: float,
    opponent_rating: float,
    actual_result: float,
    k_factor: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Update both players' ratings after a match."""

    if not 0.0 <= actual_result <= 1.0:
        raise ValueError("Result must be between 0 and 1.")

    if k_factor <= 0:
        raise ValueError("K-factor must be positive.")

    player_expected = expected_probability(
        player_rating,
        opponent_rating,
    )

    opponent_expected = 1.0 - player_expected

    new_player_rating = (
        player_rating
        + k_factor * (actual_result - player_expected)
    )

    opponent_result = 1.0 - actual_result

    new_opponent_rating = (
        opponent_rating
        + k_factor * (opponent_result - opponent_expected)
    )

    return new_player_rating, new_opponent_rating


def surface_adjusted_rating(
    overall_rating: float,
    surface_rating: float,
    weight: float = 0.35,
) -> float:
    """Combine overall and surface-specific player ratings."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("Weight must be between 0 and 1.")

    return (
        overall_rating * (1.0 - weight)
        + surface_rating * weight
    )
