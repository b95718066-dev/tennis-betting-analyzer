"""Sample-size diagnostics for live serve-point rates.

The observed rate is preserved unchanged. Wilson intervals communicate uncertainty;
no historical prior or unverified shrinkage is silently introduced.
"""

from __future__ import annotations

from math import sqrt
from typing import Any


WILSON_Z_95 = 1.959963984540054
LOW_SAMPLE_POINTS = 30
MODERATE_SAMPLE_POINTS = 60


def wilson_interval(won: int, played: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if played <= 0 or won < 0 or won > played:
        raise ValueError("Require 0 <= won <= played and played > 0.")
    p = won / played
    z2 = z * z
    denominator = 1 + z2 / played
    center = (p + z2 / (2 * played)) / denominator
    margin = z * sqrt((p * (1 - p) / played) + (z2 / (4 * played * played))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def assess_serve_sample(won: int, played: int) -> dict[str, Any]:
    """Return the observed rate, 95% Wilson interval and transparent sample label."""
    if played <= 0 or won < 0 or won > played:
        raise ValueError("Require 0 <= won <= played and played > 0.")
    low, high = wilson_interval(won, played)
    if played < LOW_SAMPLE_POINTS:
        quality = "low"
        warning = "Small in-match serve sample; probability estimate has high statistical uncertainty."
    elif played < MODERATE_SAMPLE_POINTS:
        quality = "moderate"
        warning = "Serve sample is still limited; interpret the interval alongside other evidence."
    else:
        quality = "stronger"
        warning = None
    return {
        "observed_rate": won / played,
        "sample_size": played,
        "wilson_95_interval": [low, high],
        "sample_quality": quality,
        "warning": warning,
        "method": "wilson_score_interval_95",
    }
