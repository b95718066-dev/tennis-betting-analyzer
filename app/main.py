from typing import Any
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .analytics.prediction import predict_match
from .analytics.backtest import BacktestBet, run_backtest
from .analytics.walk_forward import walk_forward_historical_elo
from .analytics.probability import remove_margin, expected_value, value_bet
from .analytics.live_probability import calculate_live_probabilities
from .analytics.serve_stats import normalize_live_match_stats, normalize_live_serve_rates
from .analytics.historical_elo import historical_elo_ratings
from .data.config import ProviderSettings, build_live_provider, build_odds_provider
from .data.models import Tour
from .data.normalization import normalize_cito_live_match
from .data.providers.http import ProviderHttpError
from .services.live_sync import persist_live_rows
from .services.historical_sync import persist_historical_rows, player_history_features
from .services.odds_sync import persist_odds_events
from .services.live_value import analyze_stored_live_match
from .services.pre_match_forecast import generate_pre_match_forecast, search_historical_players
from .storage.database import build_engine, create_schema
from .storage.models import OddsObservation, PredictionRecord, ProviderMatchLink, SmartMatchWatchlist, TennisMatch
from .services.smaֻǱ˞