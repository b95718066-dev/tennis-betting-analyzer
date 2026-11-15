from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.services.live_value import analyze_stored_live_match
from app.storage.database import create_schema
from app.storage.models import OddsObservation, PredictionRecord, ProviderMatchLink, TennisMatch


def seed_live_market(session, *, age_minutes=1):
    captured = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import timedelta
    captured -= timedelta(minutes=age_minutes)
    match = TennisMatch(
        provider="cito", external_id="live-42", tour="ATP", status="live",
        scheduled_at=datetime.now(timezone.utc), surface="hard",
        player_1_name="Player One", player_2_name="Player Two", best_of=3,
        scores=[{"player_1_games": 2, "player_2_games": 1}],
        raw={
            "current_set": 1, "game_score": "15-0", "server": 1,
            "player1": {"name": "Player One", "current_game_score": "15", "is_serving": True},
            "player2": {"name": "Player Two", "current_game_score": "0", "is_serving": False},
        },
    )
    session.add(match)
    session.flush()
    session.add(ProviderMatchLink(
        source_provider="the_odds_api", source_external_id="odds-event-42",
        canonical_provider="cito", canonical_external_id="live-42",
        match_method="exact_player_pair_and_start_time", confidence=1.0,
    ))
    for name, price in (("Player One", 2.0), ("Player Two", 2.0)):
        session.add(OddsObservation(
            provider="the_odds_api", external_match_id="odds-event-42",
            bookmaker="book-a", market="h2h", selection=name,
            odds_decimal=price, captured_at=captured, is_live=True,
        ))


def test_live_analysis_joins_fresh_odds_and_records_value_prediction():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        seed_live_market(session)
        result = analyze_stored_live_match(
            session, match_id="live-42",
            player_1_serve_point_win=0.72,
            player_2_serve_point_win=0.60,
            min_ev=0.01, stake=5.0,
        )
        assert result["bookmaker"] == "book-a"
        assert result["odds_age_seconds"] < 180
        assert result["probability_model"]["market"]["overround"] == pytest.approx(0.0)
        assert result["journal_prediction_ids"]
        rows = session.scalars(select(PredictionRecord)).all()
        assert len(rows) == len(result["value_bets"])
        assert all(row.status == "open" and row.stake == 5.0 for row in rows)
        repeated = analyze_stored_live_match(
            session, match_id="live-42",
            player_1_serve_point_win=0.72,
            player_2_serve_point_win=0.60,
            min_ev=0.01, stake=5.0,
        )
        assert repeated["journal_prediction_ids"] == result["journal_prediction_ids"]
        assert len(session.scalars(select(PredictionRecord)).all()) == len(result["value_bets"])


def test_live_analysis_rejects_stale_odds():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        seed_live_market(session, age_minutes=30)
        with pytest.raises(LookupError, match="older than 15 minutes"):
            analyze_stored_live_match(
                session, match_id="live-42",
                player_1_serve_point_win=0.70,
                player_2_serve_point_win=0.60,
            )


def test_live_analysis_refuses_tiebreak_without_first_server_identity():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        seed_live_market(session)
        match = session.scalar(select(TennisMatch).where(TennisMatch.external_id == "live-42"))
        match.scores = [{"player_1_games": 6, "player_2_games": 6,
                         "player_1_tiebreak": 4, "player_2_tiebreak": 3}]
        with pytest.raises(ValueError, match="first server"):
            analyze_stored_live_match(
                session, match_id="live-42",
                player_1_serve_point_win=0.70,
                player_2_serve_point_win=0.60,
            )
