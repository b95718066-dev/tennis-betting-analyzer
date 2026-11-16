from datetime import datetime, timezone

import pytest
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


@pytest.mark.parametrize("match_id", ["../secret", r"..\\secret", "a/b", "a?x=1", "white space", ""])
def test_invalid_match_ids_are_rejected_for_all_path_syntax(match_id):
    engine = make_session()
    with Session(engine) as session, session.begin():
        try:
            watch_match(session, match_id)
        except ValueError as error:
            assert "valid Cito match ID" in str(error)
        else:
            raise AssertionError("invalid match ID should be rejected")
    engine.dispose()


@pytest.mark.asyncio
async def test_watchlist_api_lifecycle(monkeypatch, tmp_path):
    import httpx
    import app.main as api

    monkeypatch.setenv("TENNIS_SMART_MATCH_WATCHLIST_ENABLED", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'watchlist-api.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/watchlist", json={"match_id": "cito-api-123"})
        listed = await client.get("/watchlist")
        removed = await client.delete("/watchlist/cito-api-123")
        empty = await client.get("/watchlist")

    assert created.status_code == 201
    assert created.json() == {
        "match_id": "cito-api-123", "active": True, "source": "manual",
    }
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["matches"][0]["match_id"] == "cito-api-123"
    assert removed.status_code == 200
    assert removed.json()["removed"] is True
    assert empty.json()["count"] == 0


@pytest.mark.asyncio
async def test_watchlist_poll_prioritizes_never_checked_and_retires_nested_terminal_status(
    monkeypatch, tmp_path,
):
    import httpx
    import app.main as api

    db_path = tmp_path / "watchlist-poll.db"
    monkeypatch.setenv("TENNIS_SMART_MATCH_WATCHLIST_ENABLED", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        recently_checked = watch_match(session, "cito-old-checked")
        watch_match(session, "cito-never-checked")
        recently_checked.last_checked_at = datetime.now(timezone.utc)
    engine.dispose()

    class FakeProvider:
        def __init__(self):
            self.calls = []

        async def fetch_live_match_detail(self, match_id):
            self.calls.append(match_id)
            if match_id == "cito-never-checked":
                return {"success": True, "data": {"match": {"status": "FINISHED"}}}
            return {"id": match_id, "status": "IN_PROGRESS"}

        async def fetch_live_payload(self, **kwargs):
            raise AssertionError("watchlist polling must not fetch the full live board")

    provider = FakeProvider()
    monkeypatch.setattr(api, "_live_provider", lambda: provider)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        polled = await client.post("/watchlist/poll")
        listed = await client.get("/watchlist")

    assert polled.status_code == 200
    body = polled.json()
    assert provider.calls == ["cito-never-checked", "cito-old-checked"]
    assert body["matches"][0]["status"] == "finished"
    assert body["matches"][0]["finished"] is True
    assert body["matches"][0]["active"] is False
    assert body["matches"][1]["active"] is True
    assert listed.json()["count"] == 1
    assert listed.json()["matches"][0]["match_id"] == "cito-old-checked"


@pytest.mark.asyncio
async def test_watchlist_is_hidden_when_feature_flag_is_disabled(monkeypatch):
    import httpx
    import app.main as api

    monkeypatch.setenv("TENNIS_SMART_MATCH_WATCHLIST_ENABLED", "false")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/watchlist")

    assert response.status_code == 404
