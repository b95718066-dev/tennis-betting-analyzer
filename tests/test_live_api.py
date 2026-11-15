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
