# Model 01 — Market-anchored logistic correction

This model starts from a complete two-sided (or multi-outcome) bookmaker market, removes its margin, and learns a regularized correction to the market log-odds.

For a two-outcome match:

```
logit(P_model) = logit(P_market_fair) + bias + sum(weight_i * feature_i)
```

The implementation is intentionally dependency-light and is a research baseline, not a reproduction of another repository's code.

## Input rows

Each training row is a dictionary:

```python
{
    "market_probability": 0.62,  # selected player's de-vigged probability
    "features": {
        "elo_diff_scaled": 0.35,
        "surface_elo_diff_scaled": 0.12,
        "serve_return_diff_scaled": -0.08,
    },
    "outcome": True,  # selected player won
}
```

Feature names and order are taken from the first row and must match on all rows. Scale features using training data only. Do not include raw player names, post-match statistics, or any information unavailable at placement time.

## Example

```python
from experiments.model_01_market_anchored.model import MarketAnchoredLogit

model = MarketAnchoredLogit.fit(training_rows, learning_rate=0.03, epochs=3000, l2=1.0)
probability = model.predict_probability({
    "market_probability": 0.62,
    "features": {"elo_diff_scaled": 0.35, "surface_elo_diff_scaled": 0.12,
                 "serve_return_diff_scaled": -0.08},
})
```

The model exposes coefficients for inspection. It deliberately does not choose a betting threshold or claim positive expected value.

Run the synthetic smoke example (not real tennis data):

```bash
python experiments/model_01_market_anchored/example.py
```
