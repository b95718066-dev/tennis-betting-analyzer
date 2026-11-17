# Independent model experiments

These experiments are isolated from the production prediction pipeline. They do not change API routes, database schemas, provider configuration, or Production behavior.

## Included experiments

| Folder | Purpose | Run |
|---|---|---|
| `model_01_market_anchored` | Learn a regularized correction to a de-vigged market probability | `python -m pytest experiments/tests -q` |
| `model_02_markov` | Run the existing point-state Markov engine from a standalone JSON input | `python experiments/model_02_markov/run.py input.json` |
| `model_03_clv_lab` | Measure entry price versus closing fair price (CLV) | `python -m pytest experiments/tests -q` |

All code uses the repository's existing Python dependencies and standard library. No API keys are required for the offline examples.

## Important distinctions

- **Market-anchored** is a research implementation of the published modeling idea: de-vigged market log-odds are the offset, and tennis features learn a regularized correction. Features must be calculated strictly as-of the prediction timestamp and standardized/scaled by the caller using training data only.
- **Markov** reuses the project's existing transparent point-state engine in `app/analytics/live_probability.py`. It is a standalone runner, not a second implementation of the same algorithm.
- **CLV** is an evaluation metric/workflow, not a match prediction model. It compares the selected outcome's entry fair probability with its closing fair probability after separately removing the margin from both complete markets.

## Run

From the repository root:

```bash
python -m pytest experiments/tests -q
python experiments/model_02_markov/run.py experiments/model_02_markov/example_input.json
```

The market-anchored model is a small reusable Python class. Import `MarketAnchoredLogit` from `experiments.model_01_market_anchored.model`; see that folder's README for the input contract.

## Research safeguards

1. Use chronological train/validation/test splits. Never randomly split matches.
2. Build every feature using information available before the bet placement timestamp.
3. Keep opening/closing odds timestamped and bookmaker-specific; do not label undated historical prices as verified closing prices.
4. Compare against the de-vigged market baseline on exactly the same matches.
5. Report Brier score, log loss, calibration, sample size, ROI, yield, drawdown and CLV separately.
6. Do not select thresholds on the final test period. Paper-trade before considering real stakes.

## Source and attribution notes

The experiment code here is independently written for this repository; it does not vendor or copy the external projects. The design references are:

- [Tennis Analytics & Market Prediction Pipeline](https://github.com/theayobamiakin/tennis-analytics-prediction) — market-anchored log-odds correction concept. Its repository page did not expose a license file during review, so its source code is not redistributed here.
- [Tennis Value Betting Scanner](https://github.com/saviodambrosio/tennis-scanner) — Markov and CLV research concepts. Its README states MIT; this repository reuses its general ideas, not its source files.
- The Markov runner calls our own existing implementation in `app/analytics/live_probability.py`.

The historical data and odds have their own licenses and provenance requirements. This folder intentionally contains no copied datasets, trained weights, credentials, or third-party binaries.
