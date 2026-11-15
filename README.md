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
5. Expand the Sprint 5 backtest engine with walk-forward model retraining, market-specific benchmarks and transaction-cost models.
6. Track additional market-level metrics and persist prediction snapshots and settled bet ledgers.
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

The live board returns both the raw provider payload and rows that satisfy the canonical match contract in `normalized_matches`; `normalized_count` and `unmapped_count` make mapping coverage explicit. Provider errors are surfaced as sanitized HTTP 502 responses; a missing API key returns HTTP 503. Cito's Free response indicates delayed live data (the tested dashboard showed approximately 60 seconds); confirm current limits and plan terms in the provider dashboard before production use.

### Live probability endpoint

`POST /live/probability` prices the current game, current set and match winner from an explicit score state. Required inputs include sets, games, point score, current server, best-of format and each player's point-win probability on serve. At 6-6, pass the tiebreak points and first server. Optional paired decimal odds produce margin-removed market probabilities, overround, expected value and value-bet flags.

This is a transparent Markov baseline using constant serve-point probabilities, advantage games and a standard 7-point tiebreak (win by two). It is **not calibrated**, does not fetch odds, and must not be treated as an automated betting recommendation. Cito's connected feed supplies scores, not bookmaker prices; a separate licensed odds feed and validated serve statistics are required for production live pricing.


## Sprint 5: backtesting and model evaluation

The `POST /backtest` endpoint evaluates a supplied list of already-generated predictions against settled outcomes. Each item includes `event_id`, `selection`, model `probability`, executable decimal `odds_decimal`, boolean `outcome`, `stake`, timezone-aware `placed_at` and `settled_at`. The request may also include `initial_bankroll` and `calibration_bins`.

The engine processes placement and settlement events chronologically, reserves stake when a bet is placed, and does not reuse cash tied up in overlapping matches. It reports net profit, bankroll ROI, stake yield, hit rate, average odds, maximum drawdown, Brier score, log loss, calibration buckets and an equity curve. Bets that cannot be funded from available bankroll are skipped and counted.

ROI is net profit divided by initial bankroll; yield is net profit divided by executed stake. Fees, taxes, slippage and cash-out are not yet modelled. This is an evaluation framework, not proof of profitability: predictions must be generated using information available before each placement timestamp. Use chronological splits, not random train/test splits; `chronological_split` provides a simple time-ordered split helper. Calibration and scoring metrics apply to executed selections only.
