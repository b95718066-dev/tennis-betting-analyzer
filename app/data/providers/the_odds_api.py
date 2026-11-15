"""The Odds API v4 adapter for tennis match odds.

The provider's tennis coverage is tournament- and market-limited. This adapter
only normalizes markets actually present in the returned response.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.data.models import OddsSnapshot
from app.data.providers.http import HttpJsonProvider


class TheOddsApiProvider(HttpJsonProvider):
    PROVIDER_NAME = "the_odds_api"
    DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4/"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float = 15.0,
        client=None,
    ) -> None:
        if not api_key:
            raise ValueError("The Odds API key is required.")
        super().__init__(
            name=self.PROVIDER_NAME,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout_seconds=timeout_seconds,
            client=client,
        )
        self.api_key = api_key

    async def fetch_tennis_odds(
        self,
        *,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
    ) -> list[dict[str, Any]]:
        if not sport_key.startswith("tennis_") or not sport_key.replace("_", "").isalnum():
            raise ValueError("sport_key must be a valid tennis_* sport key.")
        if not regions or any(not part.strip().isalpha() for part in regions.split(",")):
            raise ValueError("regions must be a comma-separated list of region codes.")
        allowed_markets = {"h2h", "spreads", "totals"}
        requested = [item.strip() for item in markets.split(",") if item.strip()]
        if not requested or any(item not in allowed_markets for item in requested):
            raise ValueError("markets may contain only h2h, spreads and totals.")

        payload = await self.get_json(
            f"sports/{quote(sport_key, safe='_')}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": ",".join(requested),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("The Odds API response must be a list of event objects.")
        return payload

    @staticmethod
    def normalize_event_odds(event: dict[str, Any]) -> list[OddsSnapshot]:
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            return []
        commence_time = _parse_timestamp(event.get("commence_time"))
        is_live = bool(commence_time and commence_time <= datetime.now(timezone.utc))
        snapshots: list[OddsSnapshot] = []
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            return snapshots

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            bookmaker_key = bookmaker.get("key")
            if not isinstance(bookmaker_key, str) or not bookmaker_key.strip():
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_key = market.get("key")
                outcomes = market.get("outcomes")
                if market_key not in {"h2h", "spreads", "totals"} or not isinstance(outcomes, list):
                    continue
                updated = _parse_timestamp(market.get("last_update")) or _parse_timestamp(bookmaker.get("last_update"))
                captured_at = updated or datetime.now(timezone.utc)
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    selection = outcome.get("name")
                    price = outcome.get("price")
                    if not isinstance(selection, str) or isinstance(price, bool):
                        continue
                    try:
                        decimal_price = float(price)
                        point = outcome.get("point")
                        market_name = market_key if point is None else f"{market_key}:line={point}"
                        snapshots.append(OddsSnapshot(
                            provider="the_odds_api",
                            external_match_id=event_id,
                            bookmaker=bookmaker_key,
                            market=market_name,
                            selection=selection,
                            odds_decimal=decimal_price,
                            captured_at=captured_at,
                            is_live=is_live,
                        ))
                    except (TypeError, ValueError):
                        continue
        return snapshots


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)
