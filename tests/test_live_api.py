"""API tests for the Cito live-data integration."""

import httpx
import pytest

import app.main as api


class FakeLiveProvider:
    name = "cito"

    async def fetch_live_payload(self, *, tour=None, include_point_history=False):
        return [{"id": "match-1", "tour": tour.value if tour else "ATP", "points": include_point_history}]

    async def fetch_live_match_detail(self, match_id):
        return {"id": match_id, "status": "live", "score": "6-4 2-3"}


@pytest.mark.asyncio
async def test_live_route_returns_provider_matches(monkeypatch):
    monkeypatch.setattr(api, "_live_provider", lambda: FakeLiveProvider())
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/live?tour=ATP&include_point_history=true")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "cito"
    assert body["tour"] == "ATP"
    assert body["count"] == 1
    assert body["matches"][0]["points"] is True


@pytest.mark.asyncio
async def test_live_detail_route_uses_live_match_id(monkeypatch):
    monkeypatch.setattr(api, "_live_provider", lambda: FakeLiveProvider())
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/live/match-123")

    assert response.status_code == 200
    assert response.json()["match"]["id"] == "match-123"
    assert response.json()["match"]["status"] == "live"


@pytest.mark.asyncio
async def test_live_route_reports_upstream_failure_as_bad_gateway(monkeypatch):
    class BrokenProvider(FakeLiveProvider):
        async def fetch_live_payload(self, *, tour=None, include_point_history=False):
            from app.data.providers.http import ProviderHttpError
            raise ProviderHttpError("cito", 429, "cito: upstream returned an HTTP error")

    monkeypatch.setattr(api, "_live_provider", lambda: BrokenProvider())
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/live")

    assert response.status_code == 502
    assert "429" in response.json()["detail"]

@pytest.mark.asyncio
async def test_live_probability_endpoint_returns_model_and_market_value():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/live/probability", json={
            "sets_player_1": 0,
            "sets_player_2": 0,
            "games_player_1": 3,
            "games_player_2": 2,
            "points_player_1": 2,
            "points_player_2": 1,
            "server": 1,
            "best_of": 3,
            "player_1_serve_point_win": 0.64,
            "player_2_serve_point_win": 0.62,
            "player_1_odds": 1.85,
            "player_2_odds": 2.05,
        })

    assert response.status_code == 200
    body = response.json()
    assert body["player_1_match_probability"] + body["player_2_match_probability"] == pytest.approx(1.0)
    assert "player_1_current_set_probability" in body
    assert body["market"]["overround"] > 0


@pytest.mark.asyncio
async def test_live_probability_endpoint_rejects_invalid_serve_rate():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/live/probability", json={
            "server": 1,
            "player_1_serve_point_win": 1.0,
            "player_2_serve_point_win": 0.62,
        })

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_live_sync_persists_matches_and_stored_matches_can_be_read(monkeypatch, tmp_path):
    class SyncProvider(FakeLiveProvider):
        async def fetch_live_payload(self, *, tour=None, include_point_history=False):
            return [{
                "id": "sync-match-1",
                "canonical_id": "sync-match-1",
                "tour": "ATP",
                "status": "IN_PROGRESS",
                "scheduled_at": "2026-11-15T18:00:00+00:00",
                "tournament_name": "Test Open",
                "surface": "Hard",
                "best_of": 3,
                "player1": {"id": "p1", "name": "Player One", "current_game_score": "15", "is_serving": True},
                "player2": {"id": "p2", "name": "Player Two", "current_game_score": "0", "is_serving": False},
                "sets": [{"player1_games": 2, "player2_games": 1}],
                "current_set": 1,
                "game_score": "15-0",
                "server": 1,
            }]

    monkeypatch.setattr(api, "_live_provider", lambda: SyncProvider())
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'live-sync.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        sync_response = await client.post("/live/sync?tour=ATP")
        stored_response = await client.get("/matches?tour=ATP&status=live")

    assert sync_response.status_code == 200
    assert sync_response.json()["fetched_count"] == 1
    assert sync_response.json()["persisted_count"] == 1
    assert stored_response.status_code == 200
    assert stored_response.json()["count"] == 1
    assert stored_response.json()["matches"][0]["external_id"] == "sync-match-1"


@pytest.mark.asyncio
async def test_odds_sync_persists_and_links_bookmaker_snapshots(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    class FakeLiveProvider:
        name = "cito"

        async def fetch_live_payload(self, *, tour=None, include_point_history=False):
            updated = datetime.now(timezone.utc).isoformat()
            return [{
                "id": "sync-match-1",
                "canonical_id": "sync-match-1",
                "tour": "ATP",
                "status": "IN_PROGRESS",
                "scheduled_at": updated,
                "tournament_name": "Test Open",
                "surface": "Hard",
                "best_of": 3,
                "player1": {"id": "p1", "name": "Player One", "current_game_score": "15", "is_serving": True},
                "player2": {"id": "p2", "name": "Player Two", "current_game_score": "0", "is_serving": False},
                "sets": [{"player1_games": 2, "player2_games": 1}],
                "current_set": 1,
                "game_score": "15-0",
                "server": 1,
            }]

    class FakeOddsProvider:
        async def fetch_tennis_odds(self, *, sport_key, regions="eu", markets="h2h,spreads,totals"):
            updated = datetime.now(timezone.utc).isoformat()
            return [{
                "id": "odds-event-api-1",
                "home_team": "Player Two",
                "away_team": "Player One",
                "commence_time": updated,
                "bookmakers": [{
                    "key": "book-a",
                    "last_update": updated,
                    "markets": [{
                        "key": "h2h",
                        "last_update": updated,
                        "outcomes": [
                            {"name": "Player One", "price": 1.85},
                            {"name": "Player Two", "price": 2.05},
                        ],
                    }],
                }],
            }]

    monkeypatch.setattr(api, "_live_provider", lambda: FakeLiveProvider())
    monkeypatch.setattr(api, "_odds_provider", lambda: FakeOddsProvider())
    monkeypatch.setenv("TENNIS_ODDS_PROVIDER", "the_odds_api")
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'odds-sync.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        live_response = await client.post("/live/sync?tour=ATP")
        odds_response = await client.post("/odds/sync?sport_key=tennis_atp_french_open")
        match_odds_response = await client.get("/matches/sync-match-1/odds")

    assert live_response.status_code == 200
    assert odds_response.status_code == 200
    body = odds_response.json()
    assert body["events_fetched"] == 1
    assert body["snapshots_inserted"] == 2
    assert body["events_matched_to_cito"] == 1
    assert body["events_unmatched_to_cito"] == 0
    assert match_odds_response.status_code == 200
    assert match_odds_response.json()["count"] == 2



@pytest.mark.asyncio
async def test_auto_live_analysis_uses_cito_serve_rates(monkeypatch, tmp_path):
    class StatsProvider(FakeLiveProvider):
        async def fetch_live_stats_detail(self, match_id):
            assert match_id == "auto-match"
            return {
                "stats_available": True,
                "stats": {
                    "player1": {"service_points_won": 24, "service_points_played": 35},
                    "player2": {"service_points_won": 18, "service_points_played": 30},
                },
            }

    captured = {}

    def fake_analyze(session, **kwargs):
        captured.update(kwargs)
        return {"event_id": kwargs["match_id"], "value_bets": []}

    monkeypatch.setattr(api, "_live_provider", lambda: StatsProvider())
    monkeypatch.setattr(api, "analyze_stored_live_match", fake_analyze)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'auto-live.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/matches/auto-match/analyze-live/auto", json={
            "min_ev": 0.04, "stake": 3.0, "auto_record": False,
        })

    assert response.status_code == 200
    assert captured["player_1_serve_point_win"] == pytest.approx(24 / 35)
    assert captured["player_2_serve_point_win"] == pytest.approx(18 / 30)
    assert captured["stake"] == 3.0
    assert response.json()["serve_point_win_rates_source"] == "cito_live_match_stats"


@pytest.mark.asyncio
async def test_auto_live_analysis_refuses_missing_cito_serve_stats(monkeypatch, tmp_path):
    class StatsProvider(FakeLiveProvider):
        async def fetch_live_stats_detail(self, match_id):
            return {"stats_available": False, "stats_unavailable_code": "stats_pending", "stats": []}

    monkeypatch.setattr(api, "_live_provider", lambda: StatsProvider())
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'auto-live-missing.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/matches/auto-match/analyze-live/auto", json={})

    assert response.status_code == 422
    assert response.json()["detail"]["statistics"]["stats_unavailable_code"] == "stats_pending"
