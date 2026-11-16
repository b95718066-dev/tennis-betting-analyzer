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


## Sprint 6: live ingestion and persistence

- `POST /live/sync?tour=ATP` (or `WTA`) fetches the configured Cito live board, normalizes rows with known player/tour identity and idempotently upserts them into the configured database.
- `GET /matches?tour=ATP&status=live&limit=100` reads the stored canonical match records. Supported filters are tour, status and a limit from 1 to 500.
- The sync response reports fetched, persisted and unmapped row counts. Provider-specific live fields remain in the stored raw payload; unknown or incomplete rows are skipped rather than assigned invented player identities.
- This first ingestion slice initializes the current SQLAlchemy schema on demand for local development. Production deployments should apply versioned Alembic migrations and use a managed database before enabling scheduled sync.
- Syncing Cito does not create odds or serve statistics. Those remain separate provider integrations and required model inputs.


### Independent bookmaker odds ingestion

The optional `the_odds_api` adapter uses a separate `THE_ODDS_API_KEY` and does not reuse the Cito credential. Configure `TENNIS_ODDS_PROVIDER=the_odds_api`, then call:

- `POST /odds/sync?sport_key=tennis_atp_french_open`
- Optional query parameters: `regions=eu` and `markets=h2h,spreads,totals`

The adapter converts only bookmaker outcomes actually returned into timestamped decimal `OddsSnapshot` records. Market lines are retained in the market identifier, and repeated identical observations are deduplicated. `GET /providers` reports configured odds capabilities.

**Coverage limitation:** The Odds API currently lists selected Grand Slams, ATP 1000/500 and WTA 1000/500 tournaments. Tennis coverage is mainly match-winner odds, with limited game spreads and totals from selected bookmakers. This is not full ATP/WTA tour coverage or every tennis market. Odds event IDs are provider-specific and are never treated as Cito IDs. The ingestion service now creates an explicit cross-provider link only when both player names match exactly (accent/punctuation normalized), the scheduled start is within 24 hours, and there is exactly one candidate. Ambiguous or incomplete events remain unlinked. Retrieve linked quotes with `GET /matches/{cito_external_match_id}/odds`.


## Sprint 7: prediction journal

- `POST /predictions` records an auditable prediction snapshot: event, market, selection, model probability, decimal price, stake and timezone-aware placement time. Inputs are retained as placed; later odds changes do not overwrite the record.
- `GET /predictions?status=open` lists open records; use `status=settled` for completed records.
- `POST /predictions/{id}/settle?outcome=true` settles a record once. Duplicate settlement returns HTTP 409; unknown IDs return HTTP 404.
- `GET /predictions/metrics` reports settled-only hit rate, ROI, yield, net profit, Brier score and maximum drawdown. With no settled records, it explicitly reports that metrics are unavailable.
- Journal ROI and yield are both net profit divided by total recorded stake. Metrics exclude fees, taxes, slippage and cash-out. The journal stores a probability snapshot but does not yet auto-generate predictions or place bets.


## Sprint 8: live value-bet analysis

After syncing the Cito board and The Odds API snapshots, call `POST /matches/{cito_match_id}/analyze-live` with both players' serve-point-win rates. The service joins the canonical Cito match to a uniquely linked odds event, requires a complete same-bookmaker h2h pair, rejects odds older than 15 minutes, prices the exact current score with the Markov engine, and returns EV/value-bet flags. Positive-threshold candidates can be automatically written to the prediction journal; set `auto_record=false` for analysis only.

Example request body:

```json
{
  "player_1_serve_point_win": 0.68,
  "player_2_serve_point_win": 0.64,
  "min_ev": 0.03,
  "stake": 1.0,
  "auto_record": true
}
```

**Important safeguards and current limitations:**
- Serve-point-win rates are explicit request inputs. They are not fabricated from the Cito scoreboard; connect a verified player-statistics feed before automating these inputs.
- A tiebreak analysis requires Cito to supply both the tiebreak score and first server. If first-server identity is missing, analysis is rejected rather than guessed.
- Only linked, complete match-winner (h2h) odds from one bookmaker and one timestamp are used. Spread/total markets are not priced by this live match-winner model.
- Odds must be no older than 15 minutes. Refresh the odds feed before scanning again.
- Repeated scans do not create a duplicate open journal entry for the same match, selection and price within 15 minutes.
- This is an experimental, uncalibrated probability model. A value flag is a model estimate, not a guarantee or a betting instruction.


## Sprint 9: automatic live serve statistics

- `GET /matches/{cito_match_id}/serve-stats` requests the detailed Cito live match state with structured statistics and reports each player's service-point win rate when valid won/played counts are present.
- `POST /matches/{cito_match_id}/analyze-live/auto` fetches those statistics directly, validates both players' service-point counts, and passes the resulting rates into the Sprint 8 live Markov + fresh-odds analysis and prediction journal. Request options are `min_ev`, `bookmaker`, `stake`, and `auto_record`; no serve-rate values need to be supplied.
- Rates are computed only as service points won divided by service points played. Missing statistics, unavailable-stat codes, zero denominators and inconsistent counts remain unavailable; the endpoint never fills gaps with estimates.
- The endpoint exposes `ready_for_live_model`. Live analysis can use the returned rates as inputs to Sprint 8's Markov pricing endpoint.
- Cito's Free plan delays live endpoints by approximately 60 seconds and has a 500-request monthly allowance according to its current documentation. Account-specific quotas and exact response coverage must be monitored.
- The automatic route returns HTTP 422 with Cito's availability reason when the stats are missing or incomplete; it never falls back to guessed rates. Historical player priors, confidence/sample-size weighting, and automated periodic scanning remain follow-up work.


## Sprint 10: live serve sample confidence

- Each normalized live serve rate now includes its sample size and a 95% Wilson score interval. The observed rate is passed unchanged into the live Markov model; the interval is an uncertainty diagnostic, not a fabricated adjustment.
- Samples below 30 service points are labelled `low`; 30–59 are `moderate`; 60 or more are `stronger`. These are transparent operational labels, not validated betting thresholds.
- The automatic live-analysis response includes these diagnostics under `serve_statistics.serve_point_win_rates[].sample_confidence`.
- No historical player prior is currently blended into the live rate. A trustworthy player- and surface-specific historical service-points dataset is not yet persisted in this project; inventing a prior or treating match-win form as serve-point form would be statistically unsound.
- Historical form integration remains a separate next step: ingest timestamped player match/stat records, enforce a pre-match cutoff to prevent leakage, and validate any prior/shrinkage weights with chronological backtests before using them in live pricing.


## Sprint 11: historical player data access

- The Cito adapter now exposes a paginated historical archive request with tour, year, surface, player and include=stats filters. It validates query inputs and preserves the provider's items and pagination metadata.
- GET /players/{player_id}/history retrieves a player's newest-first match log, with optional surface/year filters and statistics.
- GET /players/{player_id}/form retrieves Cito's documented recent-form endpoint, including opponent-rank context when supplied by Cito.
- These endpoints return provider records without fabricating missing statistics. The history route is a retrieval layer; it does not yet persist archive rows or calculate a serve-point prior.
- Use canonical Cito player IDs (for example, IDs returned by player search), not display names. For chronological model evaluation, each feature must be cut off at the prediction timestamp; tournament-start dates for older archive eras are not exact match dates and must not be treated as exact-day evidence.
- Next work: controlled historical ingestion with resumable pagination, immutable match-stat snapshots, late-stat refresh, and walk-forward backtests with strict timestamp cutoffs.


## Sprint 12: historical storage and as-of player features

- `POST /history/sync?tour=ATP&year=2025&page=1&page_size=100` fetches one Cito archive page with `include=stats` and idempotently upserts it into `historical_match_snapshots`. Repeat with `page=2..total_pages`; the response includes provider pagination metadata and the count of rows reporting available stats.
- Re-fetching an existing match updates its raw archive payload so statistics published after the match can be filled in. Missing stats remain explicit in `stats_available` and `stats_unavailable_code`.
- `GET /players/{player_id}/historical-features?before=2026-06-01T12:00:00Z&surface=clay&limit=20` calculates recent match win/loss features using only archive rows strictly before the cutoff. Date-only rows are excluded for the same calendar day because their time is unknown.
- The historical store preserves raw Cito statistics verbatim. Match win rate is not substituted for serve-point win rate; structured serve/return aggregation and point-in-time odds features remain follow-up work.
- This is the storage and feature foundation for walk-forward evaluation, not yet an automatically trained/retrained historical prediction model. The existing `/backtest` still evaluates supplied pre-generated predictions.


## Sprint 13: historical serve and return aggregation

- `GET /players/{player_id}/historical-stats?before=2026-06-01T12:00:00Z&surface=clay&limit=200` aggregates the player's stored Cito match-stat count pairs strictly before the exclusive cutoff.
- Service-point and return-point rates are computed as summed numerators divided by summed denominators; match-level percentages and display strings are ignored.
- Player-side mapping uses the stored Cito player IDs. Surface filtering is optional; all-court aggregation is used when omitted.
- Each available service/return win metric includes a 95% Wilson interval and sample-size diagnostic. Missing or invalid count pairs and zero denominators remain unavailable.
- Date-only rows from the cutoff calendar day are excluded. Timestamped rows must be strictly earlier than the cutoff.
- This endpoint is descriptive historical evidence only. It does not silently blend historical rates into the live model; any shrinkage or weighting requires separate chronological validation.


## Sprint 14: historical Elo replay

- `GET /players/{player_id}/elo-vs/{opponent_id}?before=2026-06-01T12:00:00Z&surface=clay` replays stored completed Cito match outcomes chronologically and returns both players' as-of overall Elo, surface Elo and a baseline Elo matchup probability.
- All stored completed matches are replayed to preserve opponent-strength propagation; only information strictly before the exclusive cutoff is included.
- Date-only matches on the cutoff calendar day are excluded. Timestamped rows must be earlier than the cutoff.
- Ratings start at 1500 with K=32. Surface Elo is blended at 35% only after at least five recorded matches on that surface; otherwise effective Elo uses overall rating alone.
- This is an interpretable baseline, not a trained or calibrated betting probability. It currently has no inactivity decay, tournament-specific K factor, retirement adjustment, doubles model or walk-forward calibration. It is exposed separately and does not silently replace the existing `/predict` model.


## Sprint 15: walk-forward historical Elo validation

- `GET /backtest/historical-elo?min_training_matches=10&surface=clay` evaluates the stored Cito archive with an expanding chronological window. It does not call the provider or change the existing prediction endpoint.
- For each timestamped match, both players' pre-match ratings are used to generate the probability first; the match result is applied only afterward. Matches with identical timestamps are all predicted from the same frozen rating state before any result in that timestamp group is applied.
- The default warm-up requires at least 10 prior recorded matches for each player. Change `min_training_matches` to adjust this threshold. Surface Elo uses the existing 35% blend only after five prior matches on that surface.
- Reports out-of-sample match count, warm-up/date-only exclusions, accuracy, Brier score, log loss and calibration buckets. A sample of up to 100 individual predictions includes pre-match ratings and prior-match counts.
- Date-only archive rows are never scored because their exact start time is unknown; they can update ratings only at the end of their calendar day for subsequent dates. Invalid rows are skipped.
- ROI/Yield are intentionally not reported: this evaluation does not yet have a verified, timestamped pre-match odds join. It is a model-probability validation, not a profitability claim.
- The evaluation is limited by whatever historical matches are already stored locally. It is a transparent Elo baseline, not a trained model or evidence of a betting edge.


## Sprint 17: integrated live match analysis

The automatic live analysis response combines three independently sourced views: Cito live serve-point counts and confidence intervals; extended live metrics (aces, double faults, break points, return points, and game/point counts); and a fresh linked bookmaker h2h market priced by the live-score Markov model. It also returns an overall/surface historical Elo matchup signal replayed as-of the analysis timestamp when stable Cito player IDs and a supported surface are available.

These signals are intentionally reported side by side. Historical Elo is not blended into the live Markov probability or used to create EV flags until an ensemble passes leakage-safe walk-forward calibration. Missing identities, unsupported metrics, or absent history remain explicitly unavailable. This remains a research/paper-trading tool, not proof of an edge.


## Sprint 18: chronological probability calibration

The historical Elo walk-forward endpoint now fits a Platt logistic calibration only on the earlier 60% of its out-of-sample predictions and evaluates the calibrated probabilities on the later 40%, which remains untouched during fitting. The response reports raw Elo and calibrated holdout accuracy, Brier score and log loss, along with the calibration parameters and sample sizes. Calibration is withheld when the training sample is too small or contains only one outcome class.

This is calibration of the Elo baseline, not yet a validated Elo + live-stat ensemble. It does not create bookmaker value flags or ROI. The historical archive currently has no verified timestamped pre-match odds join to the Cito historical match IDs; bookmaker advantage and profitability cannot be measured until that identity/timestamp linkage exists. Do not use in-sample calibration plots as proof of edge.


## Smart Match Watchlist (opt-in)

The Smart Match Watchlist is disabled by default and does not change the existing live-board endpoints.

Enable it by setting:

```bash
TENNIS_SMART_MATCH_WATCHLIST_ENABLED=true
```

When enabled, a successful `POST /matches/{match_id}/analyze-live/auto` automatically adds that Cito match to the watchlist. A match can also be added explicitly with `POST /watchlist` and body `{"match_id":"CITO_MATCH_ID"}`.

- `GET /watchlist` lists active watched matches and makes no Cito requests.
- `DELETE /watchlist/{match_id}` stops watching a match while retaining its audit row.
- `POST /watchlist/poll` fetches one Cito detail request per active watched match (up to 25 per call); it does not fetch the full live board. Never-checked and least-recently-checked matches are polled first so large lists do not starve older entries.
- The dashboard exposes add/stop controls and a manual polling action. Polling is deliberately user-triggered in this release; no serverless background scheduler or recurring Cito quota spend is enabled.
- Matches with an explicit terminal status returned by Cito are deactivated automatically.

The watchlist is stored in the configured SQL database. This first version provides an explicit polling endpoint; it does not create a background scheduler. The feature flag is a rollback switch: set it to `false` or remove it to restore the prior application behavior. No production environment variable or deployment is changed by this PR.
