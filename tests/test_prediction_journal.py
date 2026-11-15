import httpx
import pytest

import app.main as api


@pytest.mark.asyncio
async def test_prediction_journal_create_settle_and_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'predictions.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/predictions", json={
            "event_id": "atp-match-42",
            "market": "match_winner",
            "selection": "Player One",
            "probability": 0.62,
            "odds_decimal": 1.9,
            "stake": 10,
        })
        assert created.status_code == 201
        prediction_id = created.json()["id"]

        listed = await client.get("/predictions?status=open")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1
        assert listed.json()["predictions"][0]["probability"] == pytest.approx(0.62)

        settled = await client.post(
            f"/predictions/{prediction_id}/settle?outcome=true"
        )
        assert settled.status_code == 200
        assert settled.json()["profit"] == pytest.approx(9.0)

        duplicate = await client.post(
            f"/predictions/{prediction_id}/settle?outcome=false"
        )
        assert duplicate.status_code == 409

        metrics = await client.get("/predictions/metrics")
        assert metrics.status_code == 200
        body = metrics.json()
        assert body["settled"] == 1
        assert body["wins"] == 1
        assert body["net_profit"] == pytest.approx(9.0)
        assert body["roi"] == pytest.approx(0.9)
        assert body["yield"] == pytest.approx(0.9)
        assert body["brier_score"] == pytest.approx((0.62 - 1) ** 2)


@pytest.mark.asyncio
async def test_prediction_journal_rejects_naive_timestamp(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'predictions-invalid.db'}")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predictions", json={
            "event_id": "atp-match-43",
            "selection": "Player Two",
            "probability": 0.4,
            "odds_decimal": 2.4,
            "placed_at": "2026-11-15T18:00:00",
        })
    assert response.status_code == 422
