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
from .services.odds_sync import persist_odds_events, persist_odds_snapshots
from .services.live_value import analyze_stored_live_match
from .services.pre_match_forecast import generate_pre_match_forecast, search_historical_players
from .storage.database import build_engine, create_schema
from .storage.models import OddsObservation, PredictionRecord, ProviderMatchLink, SmartMatchWatchlist, TennisMatch
from .services.smart_watchlist import (
    extract_match_status, is_terminal_match_status,
    list_watched_matches, unwatch_match, watch_match,
)


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


class AutoLiveValueAnalysisRequest(BaseModel):
    min_ev: float = Field(default=0.03, ge=0, le=1)
    bookmaker: str | None = Field(default=None, min_length=1, max_length=120)
    stake: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    auto_record: bool = True


class WatchlistRequest(BaseModel):
    match_id: str = Field(min_length=1, max_length=180)


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


class PreMatchForecastRequest(BaseModel):
    player_1_id: str = Field(min_length=1, max_length=180)
    player_2_id: str = Field(min_length=1, max_length=180)
    surface: str = Field(pattern="(?i)^(hard|clay|grass|carpet)$")
    before: datetime | None = None
    player_1_odds: float | None = Field(default=None, gt=1, allow_inf_nan=False)
    player_2_odds: float | None = Field(default=None, gt=1, allow_inf_nan=False)
    min_ev: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)


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
    if settings.live_provider not in {"cito", "tennis_api"}:
        raise HTTPException(status_code=503, detail="Configured live provider is not supported.")
    configured_key = settings.tennis_api_key if settings.live_provider == "tennis_api" else settings.api_key
    if not configured_key:
        raise HTTPException(status_code=503, detail="Live provider API key is not configured.")
    return build_live_provider(settings)


@app.get("/")
async def root():
    return FileResponse("app/static/index.html", media_type="text/html")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tennis-betting-analyzer"}


@app.get("/providers")
async def provider_capabilities():
    settings = ProviderSettings.from_env()
    odds_configured = settings.odds_provider == "the_odds_api" and bool(settings.odds_api_key)
    return {
        "live_provider": settings.live_provider,
        "odds_provider": settings.odds_provider,
        "odds_provider_configured": odds_configured,
        "configured": bool(settings.api_key) if settings.live_provider == "cito" else False,
        "capabilities": {
            "live_scores": settings.live_provider == "cito",
            "point_by_point": settings.live_provider == "cito",
            "historical_matches": settings.live_provider == "cito",
            "pre_match_odds": odds_configured,
            "live_odds": odds_configured,
            "odds_markets": ["h2h", "spreads", "totals"] if odds_configured else [],
            "odds_coverage_note": "Selected tennis tournaments only; actual markets vary by bookmaker and event.",
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

    tennis_odds_snapshots = []
    odds_fetch_errors = []
    if provider.name == "tennis_api":
        for row in rows:
            event_id = row.get("provider_event_id") or row.get("id")
            if not event_id:
                continue
            try:
                odds_payload = await provider.fetch_recent_odds(str(event_id), inplay=True)
                tennis_odds_snapshots.extend(provider.normalize_recent_odds(
                    odds_payload, event_id=str(event_id),
                    player_1_name=(row.get("player1") or {}).get("name", "player_1"),
                    player_2_name=(row.get("player2") or {}).get("name", "player_2"),
                ))
            except (ProviderHttpError, ValueError):
                odds_fetch_errors.append(str(event_id))

    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            result = persist_live_rows(session, rows)
            odds_result = persist_odds_snapshots(session, tennis_odds_snapshots, provider="tennis_api")
        if provider.name == "tennis_api":
            result["odds_sync"] = {
                **odds_result,
                "events_without_odds_response": len(odds_fetch_errors),
                "event_ids_without_odds_response": odds_fetch_errors[:20],
            }
        else:
            result["odds_sync"] = {"provider": "separate_odds_feed", "automatic_sync": False}
        result["delay"] = getattr(provider, "last_live_metadata", {}).get("delayed")
        result["tour"] = tour.value if tour else "ALL"
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail="Live matches could not be persisted.") from error
    finally:
        engine.dispose()


@app.get("/odds/sports")
async def available_tennis_odds_sports(all_sports: bool = Query(default=False)):
    """Discover currently available tennis sport keys without consuming odds quota."""
    provider = _odds_provider()
    try:
        sports = await provider.fetch_sports(all_sports=all_sports)
        return {"provider": "the_odds_api", "count": len(sports), "sports": sports}
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Odds provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Odds provider returned an invalid sports catalogue.") from error


@app.get("/odds/diagnostics")
async def odds_provider_diagnostics():
    """Compare provider catalogue breadth, live-board breadth and persisted odds coverage."""
    odds_provider = _odds_provider()
    live_provider = _live_provider()
    try:
        sports = await odds_provider.fetch_sports(all_sports=True)
        live_rows = await live_provider.fetch_live_payload()
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Provider diagnostics request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="A provider returned invalid diagnostics data.") from error

    active_sports = [row for row in sports if row.get("active") is True]
    cito_tours: dict[str, int] = {}
    cito_tournaments: dict[str, int] = {}
    for row in live_rows:
        tour = str(row.get("tour") or row.get("category") or "UNKNOWN").upper()
        tournament = str(row.get("tournament_name") or row.get("tournament") or "UNKNOWN")
        cito_tours[tour] = cito_tours.get(tour, 0) + 1
        cito_tournaments[tournament] = cito_tournaments.get(tournament, 0) + 1

    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            snapshot_count = session.query(OddsObservation).filter(
                OddsObservation.provider == "the_odds_api"
            ).count()
            link_count = session.query(ProviderMatchLink).filter(
                ProviderMatchLink.source_provider == "the_odds_api",
                ProviderMatchLink.canonical_provider == "cito",
            ).count()
    finally:
        engine.dispose()

    active_tours = sorted({
        "WTA" if "wta" in str(row.get("key", "")).lower()
        else "ATP" if "atp" in str(row.get("key", "")).lower()
        else "OTHER"
        for row in active_sports
    })
    return {
        "status": "coverage_gap" if not active_sports or not live_rows or link_count == 0 else "providers_active",
        "odds_provider": {
            "name": "the_odds_api",
            "catalogue_tournaments": len(sports),
            "active_tournaments": len(active_sports),
            "active_tours": active_tours,
            "active": [
                {"key": row.get("key"), "title": row.get("title"), "description": row.get("description")}
                for row in active_sports
            ],
        },
        "live_provider": {
            "name": "cito",
            "live_matches": len(live_rows),
            "matches_by_tour": dict(sorted(cito_tours.items())),
            "matches_by_tournament": dict(sorted(cito_tournaments.items())),
            "delayed": getattr(live_provider, "last_live_metadata", {}).get("delayed"),
        },
        "persistence": {
            "odds_snapshots": snapshot_count,
            "linked_events": link_count,
        },
        "limitations": [
            "The sports catalogue reports tournament availability, not match-level event overlap.",
            "This diagnostic does not request bookmaker event odds and does not consume odds-market quota.",
            "A provider event must still match a unique Cito participant pair and start time before linking.",
        ],
        "next_step": (
            "Wait for an active tournament to overlap with Cito's match board, then run odds sync."
            if link_count == 0 else
            "Review linked events and verify complete two-sided bookmaker markets before value analysis."
        ),
    }


@app.get("/odds/coverage")
async def odds_coverage():
    """Summarize persisted bookmaker coverage without exposing provider credentials."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            rows = session.scalars(
                select(OddsObservation).where(OddsObservation.provider == "the_odds_api")
            ).all()
            links = session.scalars(
                select(ProviderMatchLink).where(
                    ProviderMatchLink.source_provider == "the_odds_api",
                    ProviderMatchLink.canonical_provider == "cito",
                )
            ).all()
            latest = max((row.captured_at for row in rows), default=None)
            age_seconds = None
            if latest is not None:
                if latest.tzinfo is None or latest.utcoffset() is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                age_seconds = round((datetime.now(timezone.utc) - latest.astimezone(timezone.utc)).total_seconds(), 1)
            return {
                "provider": "the_odds_api",
                "snapshots": len(rows),
                "linked_events": len(links),
                "bookmakers": sorted({row.bookmaker for row in rows}),
                "markets": sorted({row.market for row in rows}),
                "latest_snapshot_at": latest.isoformat() if latest else None,
                "latest_snapshot_age_seconds": age_seconds,
                "fresh_within_15m": age_seconds is not None and -60 <= age_seconds <= 900,
                "note": "A linked event is not necessarily a complete two-sided market; live analysis checks that separately.",
            }
    finally:
        engine.dispose()


@app.get("/odds/events")
async def preview_tennis_odds(
    sport_key: str = Query(min_length=8, max_length=100),
    regions: str = Query(default="eu", min_length=2, max_length=40),
    markets: str = Query(default="h2h", min_length=3, max_length=40),
):
    """Read current tennis bookmaker odds without persisting them."""
    provider = _odds_provider()
    try:
        events = await provider.fetch_tennis_odds(
            sport_key=sport_key, regions=regions, markets=markets
        )
        return {
            "provider": "the_odds_api",
            "sport_key": sport_key,
            "regions": regions,
            "markets": markets,
            "events_count": len(events),
            "events": events,
            "note": "Read-only preview; call POST /odds/sync to persist timestamped odds.",
        }
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Odds provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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


@app.post("/matches/{match_id}/analyze-live/auto")
async def analyze_live_match_value_auto(
    match_id: str, request: AutoLiveValueAnalysisRequest
):
    """Fetch verified Cito serve stats and use them to price the stored live score."""
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

    stats = normalize_live_serve_rates(payload)
    if not stats["ready_for_live_model"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Cito has not supplied valid service-point won/played counts for both players.",
                "statistics": stats,
            },
        )

    rates = stats["serve_point_win_rates"]
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            try:
                result = analyze_stored_live_match(
                    session,
                    match_id=match_id,
                    player_1_serve_point_win=rates[0]["serve_point_win_rate"],
                    player_2_serve_point_win=rates[1]["serve_point_win_rate"],
                    min_ev=request.min_ev,
                    bookmaker=request.bookmaker,
                    stake=request.stake,
                    auto_record=request.auto_record,
                )
            except LookupError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        result["serve_statistics"] = stats
        result["extended_live_statistics"] = normalize_live_match_stats(payload)
        result["serve_point_win_rates_source"] = "cito_live_match_stats"

        # Keep historical Elo separate from the in-play Markov price until an
        # ensemble has passed leakage-safe walk-forward calibration.
        with factory() as history_session:
            stored = history_session.scalar(select(TennisMatch).where(
                TennisMatch.provider == "cito", TennisMatch.external_id == match_id,
            ))
            if (
                stored is not None and stored.player_1_id and stored.player_2_id
                and stored.surface in {"hard", "clay", "grass", "carpet"}
            ):
                result["historical_elo_signal"] = historical_elo_ratings(
                    history_session, player_id=stored.player_1_id,
                    opponent_id=stored.player_2_id,
                    before=datetime.now(timezone.utc).isoformat(),
                    surface=stored.surface,
                )
            else:
                result["historical_elo_signal"] = {
                    "available": False,
                    "reason": "Stable Cito player IDs or a supported surface are missing; no identity was guessed.",
                }
        result["probability_comparison"] = {
            "live_markov": (result.get("probability_model") or {}).get("player_1_match_probability"),
            "historical_elo": result["historical_elo_signal"].get("player_win_probability_elo"),
            "combined_probability": None,
            "reason": "Signals remain separate until walk-forward calibration validates an ensemble.",
        }
        if os.getenv("TENNIS_SMART_MATCH_WATCHLIST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}:
            with factory.begin() as watch_session:
                watch_match(watch_session, match_id, source="analyzed")
            result["watchlist"] = {"enabled": True, "watching": True, "match_id": match_id}
        return result
    finally:
        engine.dispose()


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



@app.get("/players/{player_id}/history")
async def player_historical_matches(
    player_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    surface: str | None = Query(default=None),
    year: int | None = Query(default=None, ge=1877, le=2100),
    include_stats: bool = Query(default=True),
):
    """Fetch a player's newest-first Cito archive log, optionally with match stats."""
    provider = _live_provider()
    try:
        payload = await provider.fetch_player_matches(
            player_id, limit=limit, surface=surface, year=year,
            include_stats=include_stats,
        )
        return {
            "provider": "cito",
            "player_id": player_id,
            "filters": {"limit": limit, "surface": surface, "year": year,
                        "include_stats": include_stats},
            "history": payload,
            "note": "Provider archive rows are returned as supplied; missing statistics remain missing.",
        }
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Historical data provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/players/{player_id}/form")
async def player_recent_form(
    player_id: str,
    limit: int = Query(default=10, ge=1, le=100),
):
    """Fetch Cito's documented recent-form results for one canonical player ID."""
    provider = _live_provider()
    try:
        payload = await provider.fetch_player_form(player_id, limit=limit)
        return {"provider": "cito", "player_id": player_id, "form": payload}
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Player form provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error



@app.post("/history/sync")
async def sync_historical_matches(
    tour: str = Query(pattern="^(ATP|WTA)$"),
    year: int = Query(ge=1877, le=2100),
    surface: str | None = Query(default=None, pattern="^(hard|clay|grass|carpet)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=100),
):
    """Fetch and upsert one historical archive page, including available match stats."""
    provider = _live_provider()
    try:
        payload = await provider.fetch_historical_matches(
            tour=tour, year=year, surface=surface, page=page,
            page_size=page_size, include_stats=True,
        )
    except ProviderHttpError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Historical provider request failed (HTTP {error.status_code or 'network error'}).",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            result = persist_historical_rows(session, payload["items"])
        result.update({
            "provider": "cito", "tour": tour, "year": year, "surface": surface,
            "page": payload.get("page", page),
            "page_size": payload.get("page_size", page_size),
            "total": payload.get("total"),
            "total_pages": payload.get("total_pages"),
            "has_next": payload.get("has_next"),
            "stats_rows": sum(bool(row.get("stats_available")) for row in payload["items"]),
        })
        return result
    finally:
        engine.dispose()


@app.get("/players/{player_id}/historical-features")
async def get_player_historical_features(
    player_id: str,
    before: str = Query(description="Exclusive ISO-8601 prediction timestamp"),
    surface: str | None = Query(default=None, pattern="^(hard|clay|grass|carpet)$"),
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return prior match-result features with strict timestamp leakage protection."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return player_history_features(
                    session, player_id=player_id, before=before,
                    surface=surface, limit=limit,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()




@app.get("/players/{player_id}/historical-stats")
async def get_player_historical_stats(
    player_id: str,
    before: str = Query(description="Exclusive ISO-8601 prediction timestamp"),
    surface: str | None = Query(default=None, pattern="^(hard|clay|grass|carpet)$"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Return prior Cito match-result features without inferring serve/return rates."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return player_history_features(
                    session, player_id=player_id, before=before,
                    surface=surface, limit=limit,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()




@app.get("/players/{player_id}/elo-vs/{opponent_id}")
async def get_historical_elo_matchup(
    player_id: str,
    opponent_id: str,
    before: str = Query(description="Exclusive ISO-8601 prediction timestamp"),
    surface: str = Query(pattern="^(hard|clay|grass|carpet)$"),
):
    """Return as-of overall/surface Elo ratings and baseline matchup probability."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return historical_elo_ratings(
                    session, player_id=player_id, opponent_id=opponent_id,
                    before=before, surface=surface,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()


@app.get("/backtest/historical-elo")
async def backtest_historical_elo(
    surface: str | None = Query(default=None, pattern="^(hard|clay|grass|carpet)$"),
    tour: str | None = Query(default=None, pattern="^(?i:ATP|WTA)$"),
    min_training_matches: int = Query(default=10, ge=1, le=1000),
    calibration_bins: int = Query(default=10, ge=2, le=50),
    min_ev: float = Query(default=0.03, ge=0, le=1),
):
    """Run expanding-window Elo evaluation using stored history only."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return walk_forward_historical_elo(
                    session, surface=surface, tour=tour,
                    min_training_matches=min_training_matches,
                    calibration_bins=calibration_bins,
                    min_ev=min_ev,
                )
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


@app.get("/players/search")
async def search_players_for_forecast(
    q: str = Query(min_length=2, max_length=100),
    tour: str | None = Query(default=None, pattern="^(?i:ATP|WTA)$"),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Search player names in the local historical archive for forecast selection."""
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return search_historical_players(
                    session, query=q, limit=limit, tour=tour,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()


@app.post("/predict/historical")
async def predict_historical_matchup(request: PreMatchForecastRequest):
    """Generate a pre-match forecast from stored history without manual Elo inputs."""
    before = request.before or datetime.now(timezone.utc)
    if before.tzinfo is None or before.utcoffset() is None:
        raise HTTPException(status_code=422, detail="before must include a timezone.")
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            try:
                return generate_pre_match_forecast(
                    session,
                    player_1_id=request.player_1_id,
                    player_2_id=request.player_2_id,
                    surface=request.surface,
                    before=before.astimezone(timezone.utc).isoformat(),
                    player_1_odds=request.player_1_odds,
                    player_2_odds=request.player_2_odds,
                    min_ev=request.min_ev,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        engine.dispose()


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



def _smart_watchlist_enabled() -> bool:
    return os.getenv("TENNIS_SMART_MATCH_WATCHLIST_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _require_smart_watchlist() -> None:
    if not _smart_watchlist_enabled():
        raise HTTPException(
            status_code=404,
            detail="Smart Match Watchlist is disabled. Set TENNIS_SMART_MATCH_WATCHLIST_ENABLED=true to enable it.",
        )


@app.get("/watchlist")
async def get_smart_match_watchlist():
    """List opted-in Cito matches without making any upstream provider requests."""
    _require_smart_watchlist()
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            rows = list_watched_matches(session)
            return {
                "enabled": True,
                "count": len(rows),
                "matches": [
                    {
                        "match_id": row.external_match_id,
                        "provider": row.provider,
                        "source": row.source,
                        "active": row.active,
                        "last_status": row.last_status,
                        "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
                    }
                    for row in rows
                ],
            }
    finally:
        engine.dispose()


@app.post("/watchlist", status_code=201)
async def add_smart_match_to_watchlist(request: WatchlistRequest):
    """Explicitly add one Cito match to the monitored set."""
    _require_smart_watchlist()
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            try:
                row = watch_match(session, request.match_id, source="manual")
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            return {
                "match_id": row.external_match_id,
                "active": row.active,
                "source": row.source,
            }
    finally:
        engine.dispose()


@app.delete("/watchlist/{match_id}")
async def remove_smart_match_from_watchlist(match_id: str):
    """Stop monitoring a match without deleting its audit history."""
    _require_smart_watchlist()
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory.begin() as session:
            removed = unwatch_match(session, match_id)
            return {"match_id": match_id, "removed": removed}
    finally:
        engine.dispose()


@app.post("/watchlist/poll")
async def poll_smart_match_watchlist():
    """Fetch details only for active watched matches; finished matches are retired."""
    _require_smart_watchlist()
    engine = build_engine()
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as session:
            rows = list_watched_matches(session, poll_priority=True)
            match_ids = [row.external_match_id for row in rows[:25]]
            total_active = len(rows)
        if not match_ids:
            return {
                "checked": 0,
                "matches": [],
                "remaining_unpolled": 0,
                "note": "No active watched matches; no Cito requests made.",
            }

        provider = _live_provider()
        checked = []
        for match_id in match_ids:
            try:
                payload = await provider.fetch_live_match_detail(match_id)
                raw_status = extract_match_status(payload)
                terminal = is_terminal_match_status(raw_status)
                with factory.begin() as session:
                    row = session.scalar(select(SmartMatchWatchlist).where(
                        SmartMatchWatchlist.provider == "cito",
                        SmartMatchWatchlist.external_match_id == match_id,
                    ))
                    if row is not None:
                        row.last_status = raw_status or row.last_status
                        row.last_checked_at = datetime.now(timezone.utc)
                        if terminal:
                            row.active = False
                        still_active = row.active
                    else:
                        still_active = False
                checked.append({
                    "match_id": match_id,
                    "status": raw_status or "unknown",
                    "active": still_active,
                    "finished": terminal,
                })
            except ProviderHttpError as error:
                # Missing/retired provider IDs should not consume quota forever.
                # Transient failures (429, timeout, 5xx) stay active and are retried.
                not_found = error.status_code in {404, 410}
                with factory.begin() as session:
                    row = session.scalar(select(SmartMatchWatchlist).where(
                        SmartMatchWatchlist.provider == "cito",
                        SmartMatchWatchlist.external_match_id == match_id,
                    ))
                    if row is not None:
                        row.last_checked_at = datetime.now(timezone.utc)
                        if not_found:
                            row.last_status = "not_found"
                            row.active = False
                        still_active = row.active
                    else:
                        still_active = False
                checked.append({
                    "match_id": match_id,
                    "error": "match_not_found" if not_found else "provider_unavailable",
                    "active": still_active,
                })
            except ValueError:
                # A malformed response is not evidence that the match has ended.
                with factory.begin() as session:
                    row = session.scalar(select(SmartMatchWatchlist).where(
                        SmartMatchWatchlist.provider == "cito",
                        SmartMatchWatchlist.external_match_id == match_id,
                    ))
                    if row is not None:
                        row.last_checked_at = datetime.now(timezone.utc)
                checked.append({
                    "match_id": match_id,
                    "error": "invalid_provider_response",
                })
        return {
            "checked": len(checked),
            "matches": checked,
            "remaining_unpolled": max(0, total_active - len(match_ids)),
            "note": "One detail request per watched match; no full live-board request is made.",
        }
    finally:
        engine.dispose()
