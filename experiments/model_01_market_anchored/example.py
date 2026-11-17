"""Small synthetic smoke test; not real tennis training data."""
from __future__ import annotations

from model import MarketAnchoredLogit


def main() -> None:
    rows = []
    # Toy signal: positive feature values are associated with wins.
    # Replace these rows with strictly pre-match, point-in-time training data.
    for index in range(100):
        signal = (index - 49.5) / 50.0
        rows.append({
            "market_probability": 0.50,
            "features": {"toy_signal": signal},
            "outcome": index >= 50,
        })

    model = MarketAnchoredLogit.fit(rows, learning_rate=0.03, epochs=3000, l2=0.5)
    example = {
        "market_probability": 0.54,
        "features": {"toy_signal": 0.70},
    }
    print("SYNTHETIC DEMO ONLY — not a tennis forecast")
    print("Predicted probability:", round(model.predict_probability(example), 4))
    print("Model:", model.to_dict())


if __name__ == "__main__":
    main()
