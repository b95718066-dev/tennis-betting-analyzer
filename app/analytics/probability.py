"""Probability and value calculations for decimal-odds markets."""

from math import isfinite
from typing import Mapping


def _validate_probability(probability: float) -> None:
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be finite and between 0 and 1.")


def _validate_odds(odds: float) -> None:
    if not isfinite(odds) or odds <= 1.0:
        raise ValueError("Odds must be finite and greater than 1.")


def implied_probability(odds: float) -> float:
    """Convert decimal odds into raw implied probability."""
    _validate_odds(odds)
    return 1.0 / odds


def remove_margin(odds: Mapping[str, float]) -> dict[str, float]:
    """Proportionally normalize a complete mutually-exclusive odds market."""
    if len(odds) < 2:
        raise ValueError("At least two outcomes are required to remove margin.")

    probabilities = {
        name: implied_probability(value)
        for name, value in odds.items()
    }
    total = sum(probabilities.values())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("The implied probability total must be positive and finite.")

    return {name: probability / total for name, probability in probabilities.items()}


def expected_value(probability: float, odds: float) -> float:
    """Expected net return per unit staked, before fees and limits."""
    _validate_probability(probability)
    _validate_odds(odds)
    return probability * odds - 1.0


def value_bet(
    probability: float,
    odds: float,
    min_ev: float = 0.03,
) -> bool:
    """Return whether expected net return reaches the configured threshold."""
    if not isfinite(min_ev) or min_ev < 0.0:
        raise ValueError("Minimum expected value must be finite and non-negative.")
    return expected_value(probability, odds) >= min_ev
