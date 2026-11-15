# Tennis Betting Analyzer

A modular ATP/WTA tennis analytics platform for pre-match and live research.

## Project direction

The selected scope is **Hybrid + All**: ATP and WTA coverage, pre-match and in-play analysis, and every market for which a connected provider supplies reliable data. Historical results, live scores, point-by-point feeds and bookmaker odds are separate data capabilities and must be tracked independently.

## Current implementation

- FastAPI prediction endpoint and initial Elo / recent-form / surface / H2H analytics.
- Decimal odds validation, proportional bookmaker-margin removal and expected-value calculations.
- Canonical provider-neutral match and odds snapshot contracts.
- Async HTTPS JSON transport with configurable authentication headers and sanitized upstream errors.
- Independent environment selection for historical, live-score and odds providers.
- Unit tests and GitHub Actions test workflow.

The provider transport is infrastructure, not a claim that a commercial or free data account is connected. Concrete adapters, ingestion jobs, persistent storage, live market coverage, backtesting, calibration and the dashboard remain planned work.

## Hybrid data strategy

- **Historical development data:** use public datasets only after checking provenance and license. Jeff Sackmann-derived ATP/WTA archives are commonly distributed under CC BY-NC-SA 4.0; that license is non-commercial and share-alike. Review the exact source terms before using data in a commercial product.
- **Operational match data:** add provider-specific adapters behind the canonical contracts. The free tier can be used for prototyping where its quota and coverage are sufficient.
- **Live scores / point-by-point:** treat as a separate provider capability; record provider timestamps and update delays.
- **Odds:** connect a licensed odds source independently. Store each bookmaker quote as an immutable timestamped observation. Do not assume a live-score provider also supplies odds.
- **All markets:** enable only markets actually supplied by the selected provider and validated by the adapter. Never infer unavailable prices.

Provider credentials belong in environment variables or a deployment secret manager. Start from `.env.example`; never commit real keys.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API schema and `/health` for the health check.

Run tests:

```bash
python -m pytest -q
```

## Next milestones

1. Implement concrete historical, live and odds provider adapters.
2. Persist normalized matches, player identities and odds snapshots in PostgreSQL.
3. Build market-specific pricing and live-state models.
4. Add walk-forward backtesting with strict time-based leakage controls.
5. Track bets, ROI, yield, drawdown, Brier score, log loss and calibration.
6. Deliver the dashboard and deployment pipeline.

## Responsible use

Model probabilities are estimates, not guarantees. A positive expected value is not proof of profitability. Validate models out of sample and paper-trade before considering real stakes.
