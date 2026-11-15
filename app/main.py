from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .analytics.prediction import predict_match
from .analytics.backtest import BacktestBet, run_backtest
from .analytics.probability import remove_margin, expected_value, value_bet
from .analytics.live_probability import calculate_live_probabilities
from .data.config import ProviderSettings, build_live_provider, build_odds_provider
from .data.models import Tour
from .data.normalization import normalize_cito_live_match
from .data.providers.http import ProviderHttpError
from .services.live_sync import persist_live_rows
from .services.odds_sync import persist_odds_events
from .storage.database import build_engine, create_schema
from .storage.models import TennisMatch
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
