"""Contract tests for the Cito live-score adapter."""

import asyncio

import httpx
import pytest

from app.data.models import Tour
from app.data.providers.cito import CitoTennisProvider
from app.data.providers.base import ProviderNotImplementedError


def test_cito_live_payload_uses_documented_items_envelope_and_auth_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"success": True, "items": [{"id": "match-1"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CitoTennisProvider(api_key="test-secret", client=client)

    rows = asyncio.run(provider.fetch_live_payload(tour=Tour.ATP))

    assert rows == [{"id": "match-1"}]
    assert seen["url"].endswith("/matches/live?tour=ATP")
    assert seen["key"] == "test-secret"
    assert "test-secret" not in str(provider.__dict__)
    asyncio.run(client.aclose())


def test_cito_rejects_unexpected_live_envelope():
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"success": True, "data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CitoTennisProvider(client=client)
    with pytest.raises(ValueError, match="items"):
        asyncio.run(provider.fetch_live_payload())
    asyncio.run(client.aclose())


def test_cito_does_not_claim_odds_support():
    provider = CitoTennisProvider()
    assert provider.capabilities.pre_match_odds is False
    assert provider.capabilities.live_odds is False
    with pytest.raises(ProviderNotImplementedError, match="does not currently offer"):
        asyncio.run(provider.get_odds(external_match_id="match-1"))
