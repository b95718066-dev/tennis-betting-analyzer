from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.data.models import MatchRecord, MatchStatus, OddsSnapshot, Surface, Tour
from app.storage.database import create_schema
from app.storage.models import OddsObservation, ProviderSyncState, TennisMatch
from app.storage.repository import save_match, save_odds_snapshot, update_sync_state


def make_match():
    return MatchRecord(
        provider="fixture", external_id="atp-2026-001", tour=Tour.ATP,
        status=MatchStatus.FINISHED, scheduled_at=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        surface=Surface.CLAY, player_1_id="p1", player_1_name="Player One",
        player_2_id="p2", player_2_name="Player Two", winner_id="p1",
    )


def test_match_upsert_is_idempotent_and_updates_status():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        first = save_match(session, make_match())
        first_id = first.id
        updated = make_match().model_copy(update={"status": MatchStatus.RETIRED})
        second = save_match(session, updated)
        assert second.id == first_id
        assert second.status == "retired"
        assert len(session.scalars(select(TennisMatch)).all()) == 1


def test_odds_snapshots_are_append_only_and_deduplicated():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    record = OddsSnapshot(
        provider="odds", external_match_id="m1", bookmaker="book",
        market="match_winner", selection="Player One", odds_decimal=1.9,
        captured_at=datetime(2026, 6, 1, 12, tzinfo=timezone.utc), is_live=True,
    )
    with Session(engine) as session, session.begin():
        _, inserted = save_odds_snapshot(session, record)
        _, duplicate_inserted = save_odds_snapshot(session, record)
        assert inserted is True
        assert duplicate_inserted is False
        assert len(session.scalars(select(OddsObservation)).all()) == 1


def test_provider_sync_state_is_upserted():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        update_sync_state(session, provider="feed", resource="results", cursor="2026-06-01")
        state = update_sync_state(
            session, provider="feed", resource="results", cursor="2026-06-02",
            last_success_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        )
        assert state.cursor == "2026-06-02"
        assert len(session.scalars(select(ProviderSyncState)).all()) == 1
