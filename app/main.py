from typing import Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .analytics.prediction import predict_match
from .analytics.backtest import BacktestBet, run_backtest
from .analytics.probability import remove_margin, expected_value, value_bet
from .analytics.live_probability import calculate_live_probabilities
from .analytics.serve_stats import normalize_live_serve_rates
from .data.config import ProviderSettings, build_live_provider, build_odds_provider
from .data.models import Tour
from .data.normalization import normalize_cito_live_match
from .data.providers.http import ProviderHttpError
from .services.live_sync import persist_live_rows
from .services.odds_sync import persist_odds_events
from .services.live_value import analyze_stored_live_match
from .storage.database import build_engine, create_schema
from .storage.models import OddsObservation, PredictionRecord, ProviderMatchLink, TennisMatch
from .storage.repository import save_odds_snapshot


app = FastAPI(
    title="Tennis Betting Analyzer",
    description="Advanced tennis analytics and prediction API",
    version="0.2.0",
)


class BacktestRequest(BaseModel):
    bets: list[BacktestBet] = Field(min_length=1)
    initial_bankroll: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    calibration_bins: int = Field(default=10, ge=2, le=50)


class LiveProbabilityRequest(BaseModel):
    sets_player_1: int = Field(default=0, ge=0, le=2)
    sets_player_2: int = Field(default=0, ge=0, le=2)
    games_player_1: int = Field(default=0, ge=0, le=7)
    games_player_2: int = Field(default=0, ge=0, le=7)
    points_player_1: int = Field(default=0, ge=0, le=4)
    points_player_2: int = Field(default=0, ge=0, le=4)
    server: int = Field(ge=1, le=2)
    best_of: int = Field(default=3, description="Best-of-3 or best-of-5")
    player_1_serve_point_win: float = Field(gt=0, lt=1)
    player_2_serve_point_win: float = Field(gt=0, lt=1)
    tiebreak_points_player_1: int = Field(default=0, ge=0)
    tiebreak_points_player_2: int = Field(default=0, ge=0)
    tiebreak_first_server: int | None = Field(default=None, ge=1, le=2)
    player_1_odds: float | None = Field(default=None, gt=1)
    player_2_odds: float | None = Field(default=None, gt=1)
    min_ev: float = Field(default=0.03, ge=0, le=1)


class LiveValueAnalysisRequest(BaseModel):
    player_1_serve_point_win: float = Field(gt=0, lt=1)
    player_2_serve_point_win: float = Field(gt=0, lt=1)
    min_ev: float = Field(default=0.03, ge=0, le=1)
    bookmaker: str | None = Field(default=None, min_length=1, max_length=120)
    stake: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    auto_record: bool = True


class PredictionJournalRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=180)
    market: str = Field(default="match_winner", min_length=1, max_length=120)
    selection: str = Field(min_length=1, max_length=180)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    odds_decimal: float = Field(gt=1, allow_inf_nan=False)
    stake: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    placed_at: datetime | None = None


class PredictionRequest(BaseModel):
    player: str = Field(min_length=1)
    opponent: str = Field(min_length=1)
    player_rating: float
    opponent_rating: float
    player_matches: list[dict[str, Any]] = Field(default_factory=list)
    opponent_matches: list[dict[str, Any]] = Field(default_factory=list)
    h2h_matches: list[dict[str, Any]] = Field(default_factory=list)
    surface: str
    player_odds: float | None = Field(default=None, gt=1)
    opponent_odds: float | None = Field(default=None, gt=1)
    min_ev: float = Field(default=0.03, ge=0, le=1)


def _odds_provider():
    settings = ProviderSettings.from_env()
    if settings.odds_provider == "none":
        raise HTTPException(status_code=503, detail="Odds provider is not configured.")
    if settings.odds_provider != "the_odds_api":
        raise HTTPException(status_code=503, detail="Configured odds provider is not supported.")
    if not settings.odds_api_key:
        raise HTTPException(status_code=503, detail="Odds provider API key is not configured.")
    return build_odds_provider(settings)


def _live_provider():
    settings = ProviderSettings.from_env()
    if settings.live_provider == "none":
        raise HTTPException(status_code=503, detail="Live data provider is not configured.")
    if settings.live_provider != "cito":
        raise HTTPException(status_code=503, detail="Configured live provider is not supported.")
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="Live provider API key is not configured.")
    return build_live_provider(settings)


@app.get("/")
async def root():
    return {
        "project": "Tennis Betting Analyzer",
        "status": "running",
        "version": app.version,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tennis-betting-analyzer"}


@app.get("/providers")
async def provider_capabilities():
    settings = ProviderSettings.from_env()
    if settings.live_provider == "cito":
        return {
            "live_provider": "cito",
            "odds_provider": settings.odds_provider,
            "odds_provider_configured": bool(settings.odds_api_key),
            "configured": bool(settings.api_key),
            "capabilities": {
                "live_scores": True,
                "point_by_point": True,
                "historical_matches": True,
                "pre_match_odds": False,
                "live_odds": False,
            },
        }
    return {
        "live_provider": settings.live_provider,
        "odds_provider": settings.odds_provider,
        "odds_provider_configured": settings.odds_provider == "the_odds_api" and bool(settings.odds_api_key),
        "odds_capabilities": {
            "pre_match_odds": settings.odds_provider == "the_odds_api" and bool(settings.odds_api_key),
            "live_odds": settings.odds_provider == "the_odds_api" and bool(settings.odds_api_key),
            "markets": ["h2h", "spreads", "totals"],
            "coverage_note": "Selected tennis tournaments only; actual markets vary by bookmaker and event.",
        },
        "configured": False,
        "capabilities": {
            "live_scores": False,
            "point_by_point": False,
            "historical_matches": False,
            "pre_match_odds": False,
            "live_odds": False,
        },
    }


@app.get("/live")
async def live_matches(
    tour: Tour | None = Query(default=None),
    include_point_history: bool = Query(default=False),
):
    provider = _live_provider()
    try:
        rows = await provider.fetch_live_payload(
            tour=tour,
            include_point_history=include_point_history,
        )
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Live data provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Live data provider returned an invalid response.") from error

    normalized = [
        item for row in rows
        if (item := normalize_cito_live_match(row)) is not None
    ]
    return {
        "provider": provider.name,
        "tour": tour.value if tour else "ALL",
        "count": len(rows),
        "normalized_count": len(normalized),
        "unmapped_count": len(rows) - len(normalized),
        "delay": getattr(provider, "last_live_metadata", {}).get("delayed"),
        "matches": rows,
        "normalized_matches": normalized,
    }


@app.post("/live/sync")
async def sync_live_matches(
    tour: Tour | None = Query(default=None),
    include_point_history: bool = Query(default=False),
):
    """Fetch the live board from Cito and idempotently persist canonical matches."""
    provider = _live_provider()
    try:
        rows = await provider.fetch_live_payload(
            tour=tour,
            include_point_history=include_point_history,
        )
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Live data provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Live data provider returned an invalid response.") from error

    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            result = persist_live_rows(session, rows)
        result["delay"] = getattr(provider, "last_live_metadata", {}).get("delayed")
        result["tour"] = tour.value if tour else "ALL"
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail="Live matches could not be persisted.") from error
    finally:
        engine.dispose()


@app.post("/odds/sync")
async def sync_tennis_odds(
    sport_key: str = Query(min_length=8, max_length=100),
    regions: str | None = Query(default=None, min_length=2, max_length=40),
    markets: str | None = Query(default=None, min_length=3, max_length=40),
):
    """Fetch available tennis odds and persist timestamped bookmaker snapshots."""
    settings = ProviderSettings.from_env()
    provider = _odds_provider()
    try:
        events = await provider.fetch_tennis_odds(
            sport_key=sport_key,
            regions=regions or settings.odds_api_regions,
            markets=markets or settings.odds_api_markets,
        )
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Odds provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            result = persist_odds_events(session, events)
        result["sport_key"] = sport_key
        result["regions"] = regions or settings.odds_api_regions
        result["markets"] = markets or settings.odds_api_markets
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail="Odds snapshots could not be persisted.") from error
    finally:
        engine.dispose()


@app.get("/matches/{match_id}/odds")
async def stored_match_odds(match_id: str):
    """Return bookmaker snapshots linked to a canonical Cito match ID."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            links = session.scalars(
                select(ProviderMatchLink).where(
                    ProviderMatchLink.canonical_provider == "cito",
                    ProviderMatchLink.canonical_external_id == match_id,
                )
            ).all()
            event_ids = [link.source_external_id for link in links]
            if not event_ids:
                return {"match_id": match_id, "matched_odds_events": 0, "count": 0, "odds": []}
            observations = session.scalars(
                select(OddsObservation)
                .where(
                    OddsObservation.provider == "the_odds_api",
                    OddsObservation.external_match_id.in_(event_ids),
                )
                .order_by(OddsObservation.captured_at.desc())
            ).all()
            return {
                "match_id": match_id,
                "matched_odds_events": len(event_ids),
                "count": len(observations),
                "odds": [
                    {
                        "provider": item.provider,
                        "external_match_id": item.external_match_id,
                        "bookmaker": item.bookmaker,
                        "market": item.market,
                        "selection": item.selection,
                        "odds_decimal": item.odds_decimal,
                        "captured_at": item.captured_at.isoformat(),
                        "is_live": item.is_live,
                    }
                    for item in observations
                ],
            }
    finally:
        engine.dispose()


@app.get("/matches/{match_id}/serve-stats")
async def live_match_serve_stats(match_id: str):
    """Fetch and normalize Cito's structured live service-point statistics."""
    provider = _live_provider()
    try:
        payload = await provider.fetch_live_stats_detail(match_id)
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Live statistics provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Live statistics provider returned an invalid response.") from error
    result = normalize_live_serve_rates(payload)
    result["match_id"] = match_id
    return result


@app.post("/matches/{match_id}/analyze-live")
async def analyze_live_match_value(match_id: str, request: LiveValueAnalysisRequest):
    """Price a synced live score against a fresh, linked, complete bookmaker market."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            try:
                return analyze_stored_live_match(
                    session,
                    match_id=match_id,
                    player_1_serve_point_win=request.player_1_serve_point_win,
                    player_2_serve_point_win=request.player_2_serve_point_win,
                    min_ev=request.min_ev,
                    bookmaker=request.bookmaker,
                    stake=request.stake,
                    auto_record=request.auto_record,
                )
            except LookupError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()


@app.get("/matches")
async def stored_matches(
    tour: Tour | None = Query(default=None),
    status: str | None = Query(default=None, min_length=1, max_length=24),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Read the most recently updated canonical matches from local persistence."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            statement = select(TennisMatch).order_by(TennisMatch.updated_at.desc()).limit(limit)
            if tour is not None:
                statement = statement.where(TennisMatch.tour == tour.value)
            if status is not None:
                statement = statement.where(TennisMatch.status == status.lower())
            records = session.scalars(statement).all()
            return {
                "count": len(records),
                "matches": [
                    {
                        "id": row.id,
                        "provider": row.provider,
                        "external_id": row.external_id,
                        "tour": row.tour,
                        "status": row.status,
                        "scheduled_at": row.scheduled_at.isoformat() if row.scheduled_at else None,
                        "tournament_name": row.tournament_name,
                        "surface": row.surface,
                        "player_1_name": row.player_1_name,
                        "player_2_name": row.player_2_name,
                        "scores": row.scores,
                        "best_of": row.best_of,
                    }
                    for row in records
                ],
            }
    finally:
        engine.dispose()


@app.get("/live/{match_id}")
async def live_match_detail(match_id: str):
    provider = _live_provider()
    try:
        detail = await provider.fetch_live_match_detail(match_id)
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Live data provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Live data provider returned an invalid response.") from error
    return {"provider": provider.name, "match": detail}


@app.post("/live/probability")
async def live_probability(request: LiveProbabilityRequest):
    """Calculate in-play game, set and match probabilities from an explicit score state."""
    try:
        return calculate_live_probabilities(**request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict")
async def predict(request: PredictionRequest):
    if (request.player_odds is None) != (request.opponent_odds is None):
        raise HTTPException(
            status_code=422,
            detail="Both player odds and opponent odds are required.",
        )

    try:
        result = predict_match(
            player=request.player,
            opponent=request.opponent,
            player_rating=request.player_rating,
            opponent_rating=request.opponent_rating,
            player_matches=request.player_matches,
            opponent_matches=request.opponent_matches,
            h2h_matches=request.h2h_matches,
            surface=request.surface,
        )

        if request.player_odds is not None:
            market = remove_margin({
                "player": request.player_odds,
                "opponent": request.opponent_odds,
            })
            player_ev = expected_value(result["player_probability"], request.player_odds)
            opponent_ev = expected_value(result["opponent_probability"], request.opponent_odds)
            result["betting"] = {
                "market_probability": market,
                "player_ev": round(player_ev, 4),
                "opponent_ev": round(opponent_ev, 4),
                "player_value_bet": value_bet(
                    result["player_probability"], request.player_odds, request.min_ev
                ),
                "opponent_value_bet": value_bet(
                    result["opponent_probability"], request.opponent_odds, request.min_ev
                ),
            }
        return result
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error



@app.post("/predictions", status_code=201)
async def create_prediction_record(request: PredictionJournalRequest):
    placed_at = request.placed_at or datetime.now(timezone.utc)
    if placed_at.tzinfo is None or placed_at.utcoffset() is None:
        raise HTTPException(status_code=422, detail="placed_at must include a timezone.")
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            row = PredictionRecord(
                event_id=request.event_id, market=request.market,
                selection=request.selection, probability=request.probability,
                odds_decimal=request.odds_decimal, stake=request.stake,
                placed_at=placed_at.astimezone(timezone.utc), status="open",
            )
            session.add(row)
            session.flush()
            return {"id": row.id, "event_id": row.event_id, "market": row.market,
                    "selection": row.selection, "probability": row.probability,
                    "odds_decimal": row.odds_decimal, "stake": row.stake,
                    "placed_at": row.placed_at.isoformat(), "status": row.status}
    finally:
        engine.dispose()


@app.get("/predictions")
async def list_prediction_records(status: str | None = Query(default=None, pattern="^(open|settled)$"), limit: int = Query(default=100, ge=1, le=500)):
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            statement = select(PredictionRecord).order_by(PredictionRecord.placed_at.desc()).limit(limit)
            if status:
                statement = statement.where(PredictionRecord.status == status)
            rows = session.scalars(statement).all()
            return {"count": len(rows), "predictions": [
                {"id": row.id, "event_id": row.event_id, "market": row.market,
                 "selection": row.selection, "probability": row.probability,
                 "odds_decimal": row.odds_decimal, "stake": row.stake,
                 "placed_at": row.placed_at.isoformat(), "status": row.status,
                 "outcome": row.outcome, "settled_at": row.settled_at.isoformat() if row.settled_at else None,
                 "profit": row.profit} for row in rows
            ]}
    finally:
        engine.dispose()


@app.post("/predictions/{prediction_id}/settle")
async def settle_prediction_record(prediction_id: int, outcome: bool):
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            row = session.get(PredictionRecord, prediction_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Prediction not found.")
            if row.status == "settled":
                raise HTTPException(status_code=409, detail="Prediction is already settled.")
            row.outcome = outcome
            row.status = "settled"
            row.settled_at = datetime.now(timezone.utc)
            row.profit = round(row.stake * (row.odds_decimal - 1), 8) if outcome else -row.stake
            return {"id": row.id, "status": row.status, "outcome": row.outcome,
                    "settled_at": row.settled_at.isoformat(), "profit": row.profit}
    finally:
        engine.dispose()


@app.get("/predictions/metrics")
async def prediction_journal_metrics():
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            rows = session.scalars(
                select(PredictionRecord).where(PredictionRecord.status == "settled")
                .order_by(PredictionRecord.settled_at.asc(), PredictionRecord.id.asc())
            ).all()
            open_count = session.query(PredictionRecord).filter_by(status="open").count()
            if not rows:
                return {"settled": 0, "open": open_count,
                        "message": "No settled predictions yet; performance metrics are not available."}
            total_staked = sum(row.stake for row in rows)
            profit = sum(row.profit or 0.0 for row in rows)
            wins = sum(bool(row.outcome) for row in rows)
            brier = sum((row.probability - float(bool(row.outcome))) ** 2 for row in rows) / len(rows)
            equity = peak = max_drawdown = 0.0
            for row in rows:
                equity += row.profit or 0.0
                peak = max(peak, equity)
                max_drawdown = max(max_drawdown, peak - equity)
            return {
                "settled": len(rows), "open": open_count, "wins": wins, "losses": len(rows) - wins,
                "hit_rate": round(wins / len(rows), 6), "total_staked": round(total_staked, 6),
                "net_profit": round(profit, 6),
                "roi": round(profit / total_staked, 6) if total_staked else 0.0,
                "yield": round(profit / total_staked, 6) if total_staked else 0.0,
                "brier_score": round(brier, 6), "max_drawdown": round(max_drawdown, 6),
                "scope": "settled journal entries only; no fees, taxes or price slippage",
            }
    finally:
        engine.dispose()


@app.post("/backtest")
async def backtest(request: BacktestRequest):
    """Evaluate chronological, already-generated predictions against settled results."""
    try:
        return run_backtest(
            request.bets,
            initial_bankroll=request.initial_bankroll,
            calibration_bins=request.calibration_bins,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
