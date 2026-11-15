# Tennis Betting Analyzer

A modular ATP/WTA tennis analytics platform for pre-match and live research.

## Current status

The repository currently contains an initial FastAPI prediction service and analytics modules for Elo, recent form, surface performance, head-to-head records, implied probabilities, bookmaker-margin normalization and expected value. It is **not yet a production betting system**: provider ingestion, persistent history, live odds, backtesting, calibration and the UI remain under development.

## Development strategy

The project uses a hybrid data strategy:

- **Historical / development data:** public datasets may be used for prototyping only after checking their provenance and license. Jeff Sackmann-derived ATP/WTA archives are commonly distributed under CC BY-NC-SA 4.0, which is non-commercial and share-alike; they must not be used in a commercial product without appropriate permission and license review.
- **Operational data:** integrate a licensed provider behind a provider-neutral adapter. Do not hard-code a vendor or place API keys in source control.
- **Live coverage:** scores, point-by-point data and in-play odds are separate capabilities. Each provider adapter must declare its actual update frequency and market coverage; pre-match odds must never be presented as live odds.

## Planned modules

1. Provider adapters and normalized ATP/WTA data model
2. Historical match and player database
3. Pre-match prediction and market pricing
4. Live match state and in-play market analysis
5. Backtesting with time-aware splits and no data leakage
6. Bet ledger, ROI, yield, drawdown and calibration reports
7. Web dashboard and monitoring

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API schema and `/health` for the health check.

Run tests:

```bash
python -m pytest -q
```

## Configuration and security

Provider credentials belong in environment variables (for example, `TENNIS_DATA_API_KEY`) or a deployment secret manager. Never commit real keys, passwords or personal tokens.

## Responsible use

Model probabilities are estimates, not guarantees. Do not treat a positive expected value as proof of profitability. Validate every model on out-of-sample historical data and paper-trade before considering real stakes.
