
from typing import Dict


def implied_probability(odds: float) -> float:
    """Convert decimal odds into implied probability."""
    if odds <= 1:
        raise ValueError("Odds must be greater than 1.")

    return 1.0 / odds


def remove_margin(odds: Dict[str, float]) -> Dict[str, float]:
    """Normalize implied probabilities to remove bookmaker margin."""
    if not odds or any(value <= 1 for value in odds.values()):
        raise ValueError("All odds must be greater than 1.")

    probabilities = {
        name: implied_probability(value)
        for name, value in odds.items()
    }

    total = sum(probabilities.values())

    return {
        name: probability / total
        for name, probability in probabilities.items()
    }


def expected_value(probability: float, odds: float) -> float:
    """Calculate expected return per unit staked."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if odds <= 1:
        raise ValueError("Odds must be greater than 1.")

    return probability * odds - 1.0


def value_bet(
    probability: float,
    odds: float,
    min_ev: float = 0.03,
) -> bool:
    """Check whether a bet meets the minimum expected value."""
    ev = expected_value(probability, odds)

    return ev >= min_ev
