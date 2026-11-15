from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .analytics.prediction import predict_match
from .analytics.probability import remove_margin, expected_value, value_bet
from .data.config import ProviderSettings, build_live_provider
from .data.models import Tour
from .data.providers.http import ProviderHttpError


app = FastAPI(
    title="Tennis Betting Analyzer",
    description="Advanced tennis analytics and prediction API",
    version="0.2.0",
)


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

    return {
        "provider": provider.name,
        "tour": tour.value if tour else "ALL",
        "count": len(rows),
        "matches": rows,
    }


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
