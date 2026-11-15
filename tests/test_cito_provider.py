"""Contract tests for the Cito live-score adapter."""

import httpx
import pytest

from app.data.models import Tour
from app.data.providers.cito import CitoTennisProvider
from app.data.providers.base import ProviderNotImplementedError


@pytest.mark.asyncio
async def test_cito_live_payload_uses_documented_items_envelope_and_auth_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"success": True, "items": [{"id": "match-1"}], "meta": {"count": 1}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CitoTennisProvider(api_key="test-secret", client=client)
        rows = await provider.fetch_live_payload(tour=Tour.ATP, include_point_history=True)

    assert rows == [{"id": "match-1"}]
    assert seen["url"].endswith("/matches/live?tour=ATP&include=point_history")
    assert seen["key"] == "test-secret"


@pytest.mark.asyncio
async def test_cito_accepts_empty_live_items_list():
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"success": True, "items": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CitoTennisProvider(client=client)
        rows = await provider.fetch_live_payload()

    assert rows == []


@pytest.mark.asyncio
async def test_cito_does_not_claim_odds_support():
    provider = CitoTennisProvider()
    assert provider.capabilities.pre_match_odds is False
    assert provider.capabilities.live_odds is False
    with pytest.raises(ProviderNotImplementedError, match="does not currently offer"):
        await provider.get_odds(external_match_id="match-1")



@pytest.mark.asyncio
async def test_cito_http_403_preserves_only_machine_readable_error_code():
    from app.data.providers.http import ProviderHttpError

    def handler(request: httpx.Request):
        return httpx.Response(
            403,
            json={
                "error": {
                    "type": "authorization_error",
                    "message": "Email verification required",
                    "code": "EMAIL_NOT_VERIFIED",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CitoTennisProvider(api_key="test-secret", client=client)
        with pytest.raises(ProviderHttpError) as error:
            await provider.fetch_live_payload()

    assert error.value.status_code == 403
    assert error.value.error_code == "EMAIL_NOT_VERIFIED"
    assert "test-secret" not in str(error.value)
    assert "Email verification required" not in str(error.value)
