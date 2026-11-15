# Tennis Betting Analyzer

A modular ATP/WTA tennis analytics platform for pre-match and live research.

## Project direction

The selected scope is **Hybrid + All**: ATP and WTA coverage, pre-match and in-play analysis, and every market for which a connected provider supplies reliable data. Historical results, live scores, point-by-point feeds and bookmaker odds are separate data capabilities and must be tracked independently.

## Current implementation

- FastAPI prediction endpoint and initial Elo / recent-form / surface / H2H analytics.
- Decimal odds validation, proportional bookmaker-margin removal and expected-value calculations.
- Canonical provider-neutral match and odds snapshot contracts.
- Async HTTPS JSON transport with configurable authentication headers and sanitized upstream errors.
- SQLAlchemy persistence models for matches, append-only odds observations and provider sync cursors.
- Idempotent repository operations for match upserts and odds-snapshot deduplication.
- SQLite local-development default and PostgreSQL URL support.
- Independent environment selection for historical, live-score and odds providers.
- Unit tests and GitHub Actions test workflow.

Provider transport is infrastructure, not a claim that a commercial or free data account is connected. Concrete external adapters, scheduled ingestion, migrations, live market coverage, backtesting, calibration and the dashboard remain under development.

## Hybrid data strategy

- **Historical development data:** use public datasets only after checking provenance and license. Jeff Sackmann-derived ATP/WTA archives are commonly distributed under CC BY-NC-SA 4.0; that license is non-commercial and share-alike. Review the exact source terms before using data in a commercial product.
- **Operational match data:** add provider-specific adapters behind the canonical contracts. The free tier can be used for prototyping where its quota and coverage are sufficient.
- **Live scores / point-by-point:** treat as a separate provider capability; record provider timestamps and update delays.
- **Odds:** connect a licensed odds source independently. Store each bookmaker quote as an immutable timestamped observation. Do not assume a live-score provider also supplies odds.
- **All markets:** enable only markets actually supplied by the selected provider and validated by the adapter. Never infer unavailable prices.

Provider credentials belong in environment variables or a deployment secret manager. Start from .env.example; never commit real keys.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the API schema and /health for the health check.

The default local database is SQLite (./tennis.db). For PostgreSQL, set DATABASE_URL to a SQLAlchemy URL such as postgresql+psycopg://user:password@localhost:5432/tennis. Create the initial schema in a controlled startup or migration step by calling create_schema(engine) from app.storage.database.

Run tests:

```bash
python -m pytest -q
```

## Next milestones

1. Implement concrete historical, live and odds provider adapters.
2. Add Alembic migrations and a scheduled ingestion worker with rate-limit-aware checkpoints.
3. Persist player identity, tournament and market catalogs.
4. Build market-specific pricing and live-state models.
5. Add walk-forward backtesting with strict time-based leakage controls.
6. Track bets, ROI, yield, drawdown, Brier score, log loss and calibration.
7. Deliver the dashboard and deployment pipeline.

## Responsible use

Model probabilities are estimates, not guarantees. A positive expected value is not proof of profitability. Validate models out of sample and paper-trade before considering real stakes.


## Live API endpoints (Sprint 4)

When `TENNIS_LIVE_PROVIDER=cito` and `TENNIS_DATA_API_KEY` are configured:

- `GET /providers` reports configured provider capabilities (Cito does not currently supply bookmaker odds).
- `GET /live` returns the current live board.
- `GET /live?tour=ATP` or `GET /live?tour=WTA` filters the board.
- `GET /live?include_point_history=true` requests point history where the provider has it.
- `GET /live/{match_id}` returns detailed live state for one match.

The live routes return provider payloads without guessing player identity or score mappings. Provider errors are surfaced as sanitized HTTP 502 responses; a missing API key returns HTTP 503. Cito documents a free allowance of 500 requests/month and approximately 60-second live-data delay on Free and Starter plans; confirm current limits and plan terms in the provider dashboard before production use.
