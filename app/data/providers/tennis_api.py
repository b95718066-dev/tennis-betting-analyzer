"""Tennis API (RapidAPI) adapter for unified live scores and bookmaker odds.

The provider's live event ID is the canonical join key for its live score and
odds routes. Core fixture IDs and point-by-point match keys are intentionally
kept separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.data.models import OddsSnapshot
from app.data.providers.http import HttpJsonProvider


@dataclass(frozen=True)
class TennisApiCapabilities:
    live_scores: bool = True
    historical_matches: bool = True
    point_by_point: bool = True
    pre_match_odds: bool = True
    live_odds: bool = True


class TennisApiProvider(HttpJsonProvider):
    PROVIDER_NAME = "tennis_api"
    DEFAULT_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
    DEFAULT_BASE_URL = f"https://{DEFAULT_HOST}/"
    LIVE_PREFIX = "tennis/v2/extend/api/"
    capabilities = TennisApiCapabilities()

    def __init__(
        self, *, api_key: str | None, host: str | None = None,
        base_url: str | None = None, timeout_seconds: float = 15.0, client=None,
    ) -> None:
        if not api_key:
            raise ValueError("Tennis API RapidAPI key is required.")
        self.host = (host or self.DEFAULT_HOST).strip()
        if not self.host or "/" in self.host:
            raise ValueError("Tennis API host must be a hostname.")
        super().__init__(
            name=self.PROVIDER_NAME, base_url=base_url or f"https://{self.host}/",
            timeout_seconds=timeout_seconds, client=client,
        )
        self._headers["X-RapidAPI-Key"] = api_key
        self._headers["X-RapidAPI-Host"] = self.host

    async def fetch_live_events(self) -> list[dict[str, Any]]:
        payload = await self.get_json(f"{self.LIVE_PREFIX}events/live")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Tennis API live response must contain success=true.")
        rows = payload.get("results")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Tennis API live response must contain a results list.")
        return rows

    async def fetch_live_stats_detail(self, event_id: str) -> dict[str, Any]:
        """Fetch detailed statistics and adapt exact service-point counts to the model contract."""
        if not str(event_id).isdigit():
            raise ValueError("Tennis API live event ID must be numeric.")
        payload = await self.get_json(f"{self.LIVE_PREFIX}event/get/statistics/{quote(str(event_id), safe='')}")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Tennis API statistics response must contain success=true.")
        statistics = payload.get("statistics") or {}
        all_stats = statistics.get("ALL") if isinstance(statistics, dict) else None
        points = all_stats.get("points") if isinstance(all_stats, dict) else None
        metric = points.get("servicePointsScored") if isinstance(points, dict) else None
        adapted = {}
        for side in (1, 2):
            raw = metric.get(f"player{side}") if isinstance(metric, dict) else None
            if isinstance(raw, dict):
                adapted[f"player{side}"] = {
                    "service_points_won": raw.get("value"),
                    "service_points_played": raw.get("total"),
                }
        return {
            "provider": self.PROVIDER_NAME, "stats": adapted,
            "player1_name": (payload.get("player1") or {}).get("name") if isinstance(payload.get("player1"), dict) else None,
            "player2_name": (payload.get("player2") or {}).get("name") if isinstance(payload.get("player2"), dict) else None,
            "stats_available": len(adapted) == 2,
            "stats_unavailable_code": None if len(adapted) == 2 else "service_points_not_available",
            "raw_statistics": payload,
        }

    async def fetch_recent_odds(self, event_id: str, *, inplay: bool = True) -> dict[str, Any]:
        if not str(event_id).isdigit():
            raise ValueError("Tennis API live event ID must be numeric.")
        payload = await self.get_json(
            f"{self.LIVE_PREFIX}event/recent-odds/get/{quote(str(event_id), safe='')}",
            params={"inplay": "true"} if inplay else None,
        )
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Tennis API odds response must contain success=true.")
        return payload

    async def fetch_event_by_players(
        self, player_1: str, player_2: str, date_only: str | None = None,
        *, include_odds: bool = True,
    ) -> dict[str, Any]:
        from datetime import date
        if not player_1.strip() or not player_2.strip():
            raise ValueError("Both player names are required.")
        if date_only is not None:
            date.fromisoformat(date_only)
        path = f"{self.LIVE_PREFIX}event/get/{quote(player_1, safe='')}/{quote(player_2, safe='')}"
        if date_only:
            path += f"/{date_only}"
        payload = await self.get_json(path, params={"odds": "yes"} if include_odds else None)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Tennis API event lookup did not return success=true.")
        return payload

    @staticmethod
    def normalize_live_event(row: dict[str, Any]) -> dict[str, Any] | None:
        event_id = row.get("id")
        p1 = row.get("player1") or {}
        p2 = row.get("player2") or {}
        name1 = p1.get("name") if isinstance(p1, dict) else row.get("participant1")
        name2 = p2.get("name") if isinstance(p2, dict) else row.get("participant2")
        if not event_id or not name1 or not name2:
            return None
        status = str(row.get("status") or "").casefold()
        status = "live" if status in {"inplay", "in_play", "live", "in play"} else (
            "finished" if status in {"ended", "finished"} else "scheduled"
        )
        start = row.get("startTimestamp")
        scheduled = None
        if isinstance(start, (int, float)):
            scheduled = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
        tournament = row.get("tournament") if isinstance(row.get("tournament"), dict) else {}
        score_rows = []
        raw_score = row.get("score")
        if isinstance(raw_score, str):
            for part in raw_score.split(","):
                try:
                    a, b = part.strip().split("-", 1)
                    score_rows.append({"player1_games": int(a), "player2_games": int(b)})
                except (ValueError, TypeError):
                    continue
        indicator = str(row.get("indicator") or "")
        server = 1 if indicator == "1,0" else 2 if indicator == "0,1" else None
        return {
            "id": str(event_id), "canonical_id": str(event_id), "provider": "tennis_api", "tour": str(row.get("tourType") or "ATP").upper(),
            "status": "IN_PROGRESS" if status == "live" else "COMPLETED" if status == "finished" else "SCHEDULED",
            "scheduled_at": scheduled, "tournament_name": tournament.get("name") or row.get("league"),
            "tournament_id": str(tournament.get("id")) if tournament.get("id") is not None else None,
            "surface": tournament.get("surface") or "unknown", "best_of": row.get("bestOf") or 3,
            "player1": {"id": str(p1.get("id")) if p1.get("id") is not None else None,
                        "canonical_id": str(p1.get("id")) if p1.get("id") is not None else None,
                        "name": name1, "current_game_score": None,
                        "is_serving": server == 1 if server else None},
            "player2": {"id": str(p2.get("id")) if p2.get("id") is not None else None,
                        "canonical_id": str(p2.get("id")) if p2.get("id") is not None else None,
                        "name": name2, "current_game_score": None,
                        "is_serving": server == 2 if server else None},
            "sets": score_rows, "current_set": len(score_rows) if score_rows else None,
            "game_score": row.get("points"), "server": server,
            "provider_event_id": str(event_id), "provider_match_key": row.get("matchKey"),
            "raw_provider": row,
        }

    @staticmethod
    def normalize_recent_odds(
        payload: dict[str, Any], *, event_id: str,
        player_1_name: str = "player_1", player_2_name: str = "player_2",
    ) -> list[OddsSnapshot]:
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        snapshots: list[OddsSnapshot] = []
        for market_name, bookmakers in result.items():
            if not isinstance(bookmakers, dict):
                continue
            for bookmaker, quote_data in bookmakers.items():
                if not isinstance(quote_data, dict):
                    continue
                try:
                    odds_1 = float(quote_data.get("od1"))
                    odds_2 = float(quote_data.get("od2"))
                    timestamp = float(quote_data.get("addTime"))
                    captured = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                except (TypeError, ValueError, OverflowError):
                    continue
                if odds_1 <= 1 or odds_2 <= 1 or not isinstance(bookmaker, str):
                    continue
                odd_type = str(quote_data.get("oddType") or "").casefold()
                is_live = odd_type == "inplay"
                normalized_market = "h2h" if "full time result" in str(market_name).casefold() else str(market_name)
                for selection, price in (
                    (player_1_name, odds_1), (player_2_name, odds_2)
                ):
                    snapshots.append(OddsSnapshot(
                        provider="tennis_api", external_match_id=str(event_id),
                        bookmaker=bookmaker, market=normalized_market,
                        selection=selection, odds_decimal=price,
                        captured_at=captured, is_live=is_live,
                    ))
        return snapshots
