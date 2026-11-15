from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.data.config import ProviderSettings
from app.data.models import MatchRecord, MatchStatus, OddsSnapshot, Surface, Tour
from app.data.providers.http import HttpJsonProvider, ProviderHttpError


def test_match_record_normalizes_timestamp_to_utc():
    match = MatchRecord(
        provider="fixture", external_id="m-1", tour=Tour.ATP,
        status=MatchStatus.SCHEDULED,
        scheduled_at=datetime.fromisoformat("2026-05-01T12:00:00+02:00"),
        surface=Surface.CLAY, player_1_name="Player A", player_2_name="Player B",
    )
    assert match.scheduled_at == datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def test_match_record_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        MatchRecord(
            provider="fixture", external_id="m-1", tour=Tour.WTA,
            scheduled_at=datetime(2026, 5, 1, 12),
            player_1_name="Player A", player_2_name="Player B",
        )


def test_odds_snapshot_requires_timezone_and_valid_decimal_odds():
    base = dict(provider="odds", external_match_id="m-2", bookmaker="book",
                market="match_winner", selection="Player A",
                captured_at=datetime.now(timezone.utc))
    assert OddsSnapshot(**base, odds_decimal=1.91).odds_decimal == 1.91
    with pytest.raises(ValidationError):
        OddsSnapshot(**base, odds_decimal=1.0)


@pytest.mark.asyncio
async def test_http_provider_sends_configured_auth_header():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Api-Key"] == "secret-test-key"
        return httpx.Response(200, json={"matches": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = HttpJsonProvider(
            name="test", base_url="https://example.test/api", api_key="secret-test-key",
            api_key_header="X-Api-Key", api_key_prefix="", client=client,
        )
        assert await provider.get_json("matches", {"tour": "ATP"}) == {"matches": []}


@pytest.mark.asyncio
async def test_http_provider_sanitizes_upstream_errors():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "secret details"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = HttpJsonProvider(name="provider-x", base_url="https://example.test", client=client)
        with pytest.raises(ProviderHttpError) as error:
            await provider.get_json("matches")
        assert error.value.status_code == 503
        assert "secret details" not in str(error.value)


def test_http_provider_rejects_non_https_and_path_traversal():
    with pytest.raises(ValueError):
        HttpJsonProvider(name="bad", base_url="http://example.test")
    provider = HttpJsonProvider(name="ok", base_url="https://example.test")
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(provider.get_json("../admin"))


def test_provider_settings_default_to_no_external_provider(monkeypatch):
    for name in (
        "TENNIS_HISTORICAL_PROVIDER", "TENNIS_LIVE_PROVIDER", "TENNIS_ODDS_PROVIDER",
        "TENNIS_DATA_API_BASE_URL", "TENNIS_DATA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = ProviderSettings.from_env()
    assert settings.historical_provider == "none"
    assert settings.live_provider == "none"
    assert settings.odds_provider == "none"
