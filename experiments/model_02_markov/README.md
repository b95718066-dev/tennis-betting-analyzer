# Model 02 — Point-state Markov runner

This experiment exposes the repository's existing in-play Markov engine as a standalone JSON runner. It prices match-winner probability from the exact score state and the two players' serve-point-win probabilities.

The algorithm lives in `app/analytics/live_probability.py`; this folder adds a repeatable input/output interface rather than maintaining a fork that could drift away from the main implementation.

## Run

```bash
python experiments/model_02_markov/run.py experiments/model_02_markov/example_input.json
```

Input fields are documented in the example JSON. The model assumes constant serve-point probabilities, advantage games and a standard 7-point tiebreak (win by two). It is not calibrated. Accurate serve-point inputs and the correct server/tiebreak state are essential.
