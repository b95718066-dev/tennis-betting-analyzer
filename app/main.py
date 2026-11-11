
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analytics.prediction import predict_match
from .analytics.probability import remove_margin, expected_value, value_bet


app = FastAPI(
    title="Tennis Betting Analyzer",
    description="Advanced tennis analytics and prediction API",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    player: str = Field(min_length=1)
    opponent: str = Field(min_length=1)

    player_rating: float
    opponent_rating: float

    player_matches: list[dict[str, Any]] = Field(
        default_factory=list
    )
    opponent_matches: list[dict[str, Any]] = Field(
        default_factory=list
    )
    h2h_matches: list[dict[str, Any]] = Field(
        default_factory=list
    )

    surface: str

    player_odds: float | None = Field(default=None, gt=1)
    opponent_odds: float | None = Field(default=None, gt=1)

    min_ev: float = 0.03


@app.get("/")
async def root():
    return {
        "project": "Tennis Betting Analyzer",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "tennis-betting-analyzer",
    }


@app.post("/predict")
async def predict(request: PredictionRequest):

    if (request.player_odds is None) != (
        request.opponent_odds is None
    ):
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

            player_ev = expected_value(
                result["player_probability"],
                request.player_odds,
            )

            opponent_ev = expected_value(
                result["opponent_probability"],
                request.opponent_odds,
            )

            result["betting"] = {
                "market_probability": market,
                "player_ev": round(player_ev, 4),
                "opponent_ev": round(opponent_ev, 4),
                "player_value_bet": value_bet(
                    result["player_probability"],
                    request.player_odds,
                    request.min_ev,
                ),
                "opponent_value_bet": value_bet(
                    result["opponent_probability"],
                    request.opponent_odds,
                    request.min_ev,
                ),
            }

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
