from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.services.smart_watchlist import list_watched_matches, unwatch_match, watch_match
from app.storage.database import create_schema
from app.storage.models import SmartMatchWatchlist


def make_session():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    return engine


def test_watch_match_is_idempotent_and_reactivates():
    engine = make_session()
    with Session(engine) as session, session.begin():
        first = watch_match(session, "cito-123", source="analyzed")
        second = watch_match(session, "cito-123", source="analyzed")
        rows = session.scalars(select(SmartMatchWatchlist)).all()

        assert first.id == second.id
        assert len(rows) == 1
        assert rows[0].active is True
        assert rows[0].source == "analyzed"

        assert unwatch_match(session, "cito-123") is True
        assert list_watched_matches(session) == []

        reactivated = watch_match(session, "cito-123", source="manual")
        assert reactivated.id == first.id
        assert reactivated.active is True
        assert reactivated.source == "manual"
    engine.dispose()


def test_unwatch_unknown_match_is_safe():
    engine = make_session()
    with Session(engine) as session, session.begin():
        assert unwatch_match(session, "missing") is False
    engine.dispose()


def test_invalid_match_ids_are_rejected():
    engine = make_session()
    with Session(engine) as session, session.begin():
        try:
            watch_match(session, "../secret")
        except ValueError as error:
            assert "valid Cito match ID" in str(error)
        else:
            raise AssertionError("invalid match ID should be rejected")
    engine.dispose()
