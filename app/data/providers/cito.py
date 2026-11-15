"""Cito Tennis API adapter.

This adapter preserves the raw provider payload and exposes the documented
live-feed envelope and metadata. Canonical MatchRecord normalization is handled
by app.data.normalization and covered by contract tests. Cito does not currently
provide bookmaker odds; odds must come from a separate provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data.models import Tour
from .http import HttpJsonProvider


@dataclass(frozen=True)
class CitoCapabilities:
    live_scores: bool = True
    historical_matches: bool = True
    point_by_point: bool = True
    pre_match_odds: bool = False
    live_odds: bool = False


class CitoTennisProvider(HttpJsonProvider):
    """Read-only client for Cito's Tennis API (server-side API key only)."""

    PROVIDER_NAME = "cito"
    DEFAULT_BASE_URL = "https://api.citoapi.com/api/v1/tennis"
    capabilities = CitoCapabilities()

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 15.0,
        client=None,
    ) -> None:
        super().__init__(
            name=self.PROVIDER_NAME,
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key,
            api_key_header="x-api-key",
            api_key_prefix="",
            timeout_seconds=timeout_seconds,
            client=client,
        )
        self.last_live_metadata: dict[str, Any] = {}

    async def fetch_live_payload(
        self,
        *,
        tour: Tour | None = None,
        include_point_history: bool = False,
    ) -> list[dict[str, Any]]:
        """Return raw live rows; optional point history follows the Cito API contract."""
        params: dict[str, Any] = {}
        if tour in (Tour.ATP, Tour.WTA):
            params["tour"] = tour.value
        if include_point_history:
            params["include"] = "point_history"
        payload = await self.get_json("matches/live", params=params or None)
        if not isinstance(payload, dict):
            raise ValueError("Cito live response must be a JSON object.")
        self.last_live_metadata = {
            "delayed": payload.get("delayed"),
            "total": payload.get("total"),
            "page": payload.get("page"),
            "page_size": payload.get("page_size"),
            "total_pages": payload.get("total_pages"),
        }
        # Cito Tennis list endpoints use {success, items, ...}; accept the legacy
        # data key as a compatibility fallback for older upstream responses.
        rows = payload.get("items")
        if rows is None:
            rows = payload.get("data")
        if payload.get("success") is not True or not isinstance(rows, list):
            raise ValueError("Cito live response must contain success=true and an 'items' list.")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Cito live response contains a non-object match row.")
        return rows

    async def fetch_live_match_detail(self, match_id: str) -> dict[str, Any]:
        """Fetch detailed live state (scoreboard, serve, momentum and stats)."""
        from urllib.parse import quote

        if not match_id or "/" in match_id or match_id in {".", ".."}:
            raise ValueError("match_id must be a non-empty provider ID.")
        payload = await self.get_json(f"matches/live/{quote(match_id, safe='')}")
        if not isinstance(payload, dict):
            raise ValueError("Cito live match detail must be a JSON object.")
        return payload

    async def fetch_live_stats_detail(self, match_id: str) -> dict[str, Any]:
        """Fetch the documented detailed live state, including structured stats when available."""
        from urllib.parse import quote

        if not match_id or "/" in match_id or match_id in {".", ".."}:
            raise ValueError("match_id must be a non-empty provider ID.")
        payload = await self.get_json(f"matches/live/{quote(match_id, safe='')}")
        if not isinstance(payload, dict):
            raise ValueError("Cito live match detail must be a JSON object.")
        return payload

    async def fetch_match_detail(self, match_id: str) -> dict[str, Any]:
        """Fetch one provider match detail; match_id is URL-encoded as one path segment."""
        from urllib.parse import quote

        if not match_id or "/" in match_id or match_id in {".", ".."}:
            raise ValueError("match_id must be a non-empty provider ID.")
        payload = await self.get_json(f"matches/{quote(match_id, safe='')}")
        if not isinstance(payload, dict):
            raise ValueError("Cito match detail must be a JSON object.")
        return payload


    async def fetch_historical_matches(
        self,
        *,
        tour: str | None = None,
        year: int | None = None,
        surface: str | None = None,
        player_id: str | None = None,
        page: int = 1,
        page_size: int = 100,
        include_stats: bool = True,
    ) -> dict[str, Any]:
        """Fetch one page from Cito's historical match archive.

        The caller controls pagination; raw rows and provider metadata are retained.
        """
        if tour is not None and tour.upper() not in {"ATP", "WTA"}:
            raise ValueError("tour must be ATP or WTA.")
        if year is not None and not 1877 <= year <= 2100:
            raise ValueError("year is outside the supported range.")
        if surface is not None and surface.lower() not in {"hard", "clay", "grass", "carpet"}:
            raise ValueError("surface must be hard, clay, grass or carpet.")
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("page must be positive and page_size must be between 1 and 100.")
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if tour:
            params["tour"] = tour.upper()
        if year is not None:
            params["year"] = year
        if surface:
            params["surface"] = surface.title()
        if player_id:
            params["player_id"] = player_id
        if include_stats:
            params["include"] = "stats"
        payload = await self.get_json("matches", params=params)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Cito historical response must contain success=true.")
        rows = payload.get("items")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Cito historical response must contain an items list of objects.")
        return payload

    async def fetch_player_matches(
        self,
        player_id: str,
        *,
        limit: int = 10,
        surface: str | None = None,
        year: int | None = None,
        include_stats: bool = True,
    ) -> dict[str, Any]:
        """Fetch a player's newest-first archive log with optional surface/year filters."""
        from urllib.parse import quote

        if not player_id or "/" in player_id or player_id in {".", ".."}:
            raise ValueError("player_id must be a non-empty provider ID.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        if surface is not None and surface.lower() not in {"hard", "clay", "grass", "carpet"}:
            raise ValueError("surface must be hard, clay, grass or carpet.")
        params: dict[str, Any] = {"limit": limit, "view": "summary"}
        if surface:
            params["surface"] = surface.title()
        if year is not None:
            if not 1877 <= year <= 2100:
                raise ValueError("year is outside the supported range.")
            params["year"] = year
        if include_stats:
            params["include"] = "stats"
        payload = await self.get_json(f"players/{quote(player_id, safe='')}/matches", params=params)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Cito player-match response must contain success=true.")
        rows = payload.get("items")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Cito player-match response must contain an items list of objects.")
        return payload

    async def fetch_player_form(self, player_id: str, *, limit: int = 10) -> dict[str, Any]:
        """Fetch Cito's documented recent-form endpoint."""
        from urllib.parse import quote

        if not player_id or "/" in player_id or player_id in {".", ".."}:
            raise ValueError("player_id must be a non-empty provider ID.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        payload = await self.get_json(
            f"players/{quote(player_id, safe='')}/form", params={"limit": limit}
        )
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Cito player-form response must contain success=true.")
        return payload

    async def get_odds(self, *, external_match_id: str, live_only: bool = False):
        """Cito currently documents no bookmaker odds endpoint."""
        from app.data.providers.base import ProviderNotImplementedError

        raise ProviderNotImplementedError(
            "Cito does not currently offer pre-match or live bookmaker odds."
        )
