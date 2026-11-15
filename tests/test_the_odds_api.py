from datetime import datetime, timezone

import httpx
import pytest

from app.data.providers.the_odds_api import TheOddsApiProvider


@pytest.mark.asyncio
async def test_odds_api_fetch_uses_v4_tennis_route_and_api_key_query_parameter():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=[{"id": "event-1", "bookmakers": []}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = TheOddsApiProvider(api_key="test-secret", client=client)
        rows = await provider.fetch_tennis_odds(
            sport_key="tennis_atp_french_open",
            regions="eu,uk",
            markets="h2h,spreads",
        )

    assert rows[0]["id"] == "event-1"
    assert "/sports/tennis_atp_french_open/odds?" in seen["url"]
    assert "apiKey=test-secret" in seen["url"]
    assert "regions=eu%2Cuk" in seen["url"]


@pytest.mark.asyncio
async def test_odds_api_rejects_non_tennis_sport_keys_and_unknown_markets():
    provider = TheOddsApiProvider(api_key="test-secret")
    with pytest.raises(ValueError, match="tennis_"):
        await provider.fetch_tennis_odds(sport_key="soccer_epl")
    with pytest.raises(ValueError, match="only h2h"):
        await provider.fetch_tennis_odds(sport_key="tennis_atp_french_open", markets="player_props")


def test_odds_api_normalizes_bookmaker_outcomes_and_preserves_market_lines():
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "id": "event-1",
        "commence_time": now,
        "bookmakers": [{
            "key": "bookmaker-a",
            "last_update": now,
            "markets": [
                {"key": "h2h", "last_update": now, "outcomes": [
                    {"name": "Player One", "price": 1.8},
                    {"name": "Player Two", "price": 2.1},
                ]},
                {"key": "spreads", "last_update": now, "outcomes": [
                    {"name": "Player One", "price": 1.9, "point": -2.5},
                ]},
                {"key": "totals", "last_update": now, "outcomes": [
                    {"name": "Over", "price": 1.91, "point": 22.5},
                ]},
            ],
        }],
    }

    snapshots = TheOddsApiProvider.normalize_event_odds(event)

    assert len(snapshots) == 4
    assert {item.market for item in snapshots} == {
        "h2h", "spreads:line=-2.5", "totals:line=22.5"
    }
    assert all(item.is_live for item in snapshots)
    assert all(item.provider == "the_odds_api" for item in snapshots)
    assert all(item.captured_at.tzinfo is not None for item in snapshots)


def test_odds_api_skips_invalid_event_or_invalid_price():
    assert TheOddsApiProvider.normalize_event_odds({"bookmakers": []}) == []
    rows = TheOddsApiProvider.normalize_event_odds({
        "id": "event-2",
        "bookmakers": [{
            "key": "book-a",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Player One", "price": 1.0},
                {"name": "Player Two", "price": 2.0},
            ]}],
        }],
    })
    assert len(rows) == 1
    assert rows[0].selection == "Player Two"
