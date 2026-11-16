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

- **Historical development ֻǱ˞