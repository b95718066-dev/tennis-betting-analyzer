from datetime import datetime, timezone

import httpx
import pytest

from app.data.providers.tennis_api import TennisApiProvider


@pytest.mark.asyncio
async def test_live_feed_uses_one_provider_and_preserves_live_event_id():
    seen = {}

    async def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("X-RapidAPI-Key")
        seen["host"] = request.headers.get("X-RapidAPI-Host")
        return httpx.Response(200, json={
            "success": True,
            "results": [{
                "id": 123456, "status": "InPlay", "tourType": "atp",
                "participant1": "Player One", "participant2": "Player Two",
                "score": "6-4,3-2", "points": "30-15", "indicator": "1,0",
                "startTimestamp": 1794852000,
                "tournament": {"id": 77, "name": "Challenger Test", "surface": "Hard"},
            }],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TennisApiProvider(api_key="test-key", client=client)
        rows = await provider.fetch_live_events()

    assert seen["key"] == "test-key"
    assert seen["host"] == provider.DEFAULT_HOST
    assert seen["url"].endswith("/tennis/v2/extend/api/events/live")
    normalized = provider.normalize_live_event(rows[0])
    assert normalized["provider_event_id"] == "123456"
    assert normalized["player1"]["name"] == "Player One"
    assert normalized["player2"]["name"] == "Player Two"
    assert normalized["server"] == 1
    assert normalized["status"] == "IN_PROGRESS"
    assert normalized["sets"] == [
        {"player1_games": 6, "player2_games": 4},
        {"player1_games": 3, "player2_games": 2},
    ]


@pytest.mark.asyncio
async def test_recent_odds_are_normalized_against_same_live_event_id():
    async def handler(request):
        assert request.url.path.endswith("/event/recent-odds/get/123456")
        assert request.url.params.get("inplay") == "true"
        return httpx.Response(200, json={
            "success": True,
            "result": {
                "Full Time Result": {
                    "Bookmaker A": {
                        "od1": "1.85", "od2": "2.05",
                        "addTime": 1794852000, "oddType": "inplay",
                    }
                }
            },
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TennisApiProvider(api_key="test-key", client=client)
        payload = await provider.fetch_recent_odds("123456")
    snapshots = provider.normalize_recent_odds(payload, event_id="123456")
    assert len(snapshots) == 2
    assert {item.provider for item in snapshots} == {"tennis_api"}
    assert {item.external_match_id for item in snapshots} == {"123456"}
    assert {item.bookmaker for item in snapshots} == {"Bookmaker A"}
    assert all(item.is_live for item in snapshots)
    assert {item.selection for item in snapshots} == {"player_1", "player_2"}


def test_invalid_live_event_and_invalid_odds_are_not_fabricated():
    assert TennisApiProvider.normalize_live_event({"id": 1}) is None
    snapshots = TennisApiProvider.normalize_recent_odds(
        {"success": True, "result": {"market": {"book": {"od1": "bad", "od2": "2.0"}}}},
        event_id="1",
    )
    assert snapshots == []


@pytest.mark.asyncio
async def test_live_stats_uses_documented_service_points_won_and_played_counts():
    async def handler(request):
        assert request.url.path.endswith("/event/get/statistics/123456")
        return httpx.Response(200, json={
            "success": True,
            "statistics": {
                "ALL": {
                    "points": {
                        "servicePointsScored": {
                            "player1": {"value": 28, "total": 40, "percentage": 70},
                            "player2": {"value": 21, "total": 35, "percentage": 60},
                        }
                    }
                }
            }
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TennisApiProvider(api_key="test-key", client=client)
        payload = await provider.fetch_live_stats_detail("123456")

    assert payload["provider"] == "tennis_api"
    assert payload["stats"]["player1"] == {
        "service_points_won": 28, "service_points_played": 40,
    }
    assert payload["stats"]["player2"] == {
        "service_points_won": 21, "service_points_played": 35,
    }


def test_tennis_api_serve_counts_feed_existing_live_probability_contract():
    from app.analytics.serve_stats import normalize_live_serve_rates

    result = normalize_live_serve_rates({
        "provider": "tennis_api",
        "stats_available": True,
        "stats": {
            "player1": {"service_points_won": 28, "service_points_played": 40},
            "player2": {"service_points_won": 21, "service_points_played": 35},
        },
    })
    assert result["ready_for_live_model"] is True
    assert result["provider"] == "tennis_api"
    assert result["serve_point_win_rates"][0]["serve_point_win_rate"] == 0.7
    assert result["serve_point_win_rates"][1]["serve_point_win_rate"] == 0.6


def test_tennis_api_event_odds_are_persisted_without_cross_provider_linking():
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.services.odds_sync import persist_odds_snapshots
    from app.storage.database import create_schema
    from app.storage.models import OddsObservation, ProviderMatchLink

    captured = datetime.now(timezone.utc)
    from app.data.models import OddsSnapshot
    snapshots = [
        OddsSnapshot(provider="tennis_api", external_match_id="123456",
                     bookmaker="Bookmaker A", market="h2h", selection="Player One",
                     odds_decimal=1.85, captured_at=captured, is_live=True),
        OddsSnapshot(provider="tennis_api", external_match_id="123456",
                     bookmaker="Bookmaker A", market="h2h", selection="Player Two",
                     odds_decimal=2.05, captured_at=captured, is_live=True),
    ]
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        first = persist_odds_snapshots(session, snapshots, provider="tennis_api")
        second = persist_odds_snapshots(session, snapshots, provider="tennis_api")
        assert first["snapshots_inserted"] == 2
        assert second["snapshots_deduplicated"] == 2
        assert len(session.scalars(select(OddsObservation)).all()) == 2
        assert session.scalars(select(ProviderMatchLink)).all() == []
