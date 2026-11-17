"""Market-anchored logistic correction with L2 regularization.

The fair market log-odds are an offset; the model learns only a correction.
All feature engineering and time-based splitting remain the caller's duty.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any, Sequence


def _logit(probability: float) -> float:
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("market_probability must be finite and strictly between 0 and 1.")
    return log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + exp(-value))


@dataclass
class MarketAnchoredLogit:
    """A fitted market-offset logistic model."""

    feature_names: tuple[str, ...]
    coefficients: dict[str, float]
    bias: float
    l2: float
    training_rows: int

    @classmethod
    def fit(
        cls,
        rows: Sequence[dict[str, Any]],
        *,
        learning_rate: float = 0.03,
        epochs: int = 3000,
        l2: float = 1.0,
    ) -> "MarketAnchoredLogit":
        if len(rows) < 2:
            raise ValueError("At least two training rows are required.")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if not isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative.")

        first_features = rows[0].get("features")
        if not isinstance(first_features, dict) or not first_features:
            raise ValueError("Each row must contain a non-empty features mapping.")
        names = tuple(sorted(first_features))
        prepared: list[tuple[float, list[float], float]] = []
        outcomes = set()

        for row in rows:
            market = _logit(float(row["market_probability"]))
            features = row.get("features")
            if not isinstance(features, dict) or tuple(sorted(features)) != names:
                raise ValueError("All rows must contain the same feature names.")
            vector = [float(features[name]) for name in names]
            if not all(isfinite(value) for value in vector):
                raise ValueError("Feature values must be finite.")
            outcome = row.get("outcome")
            if not isinstance(outcome, bool):
                raise ValueError("outcome must be a boolean.")
            outcomes.add(outcome)
            prepared.append((market, vector, float(outcome)))

        if len(outcomes) < 2:
            raise ValueError("Training data must contain both win and loss outcomes.")

        weights = [0.0] * len(names)
        bias = 0.0
        n = len(prepared)

        # Full-batch gradient descent on logistic loss plus L2 penalty.
        for _ in range(epochs):
            grad_bias = 0.0
            gradients = [0.0] * len(names)
            for market, vector, target in prepared:
                probability = _sigmoid(market + bias + sum(
                    weight * value for weight, value in zip(weights, vector)
                ))
                error = probability - target
                grad_bias += error
                for index, value in enumerate(vector):
                    gradients[index] += error * value
            bias -= learning_rate * grad_bias / n
            for index in range(len(weights)):
                gradient = gradients[index] / n + l2 * weights[index] / n
                weights[index] -= learning_rate * gradient

        return cls(
            feature_names=names,
            coefficients=dict(zip(names, weights)),
            bias=bias,
            l2=l2,
            training_rows=n,
        )

    def predict_probability(self, row: dict[str, Any]) -> float:
        market = _logit(float(row["market_probability"]))
        features = row.get("features")
        if not isinstance(features, dict) or tuple(sorted(features)) != self.feature_names:
            raise ValueError("Prediction features must match the fitted feature names.")
        values = [float(features[name]) for name in self.feature_names]
        if not all(isfinite(value) for value in values):
            raise ValueError("Feature values must be finite.")
        correction = self.bias + sum(
            self.coefficients[name] * value
            for name, value in zip(self.feature_names, values)
        )
        return _sigmoid(market + correction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "market_anchored_logistic_correction",
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients,
            "bias": self.bias,
            "l2": self.l2,
            "training_rows": self.training_rows,
            "calibrated": False,
        }
