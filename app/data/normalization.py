"""Provider payload normalization for live tennis matches."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.data.models import MatchRecord, MatchStatus, SetScore, Surface, Tour


def _aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _tour(value: Any) -> Tour | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    aliases = {"CHALLENGER": "CHALLENGER", "ITF": "ITF", "ATP": "ATP", "WTA": "WTA"}
    try:
        return Tour(aliases.get(normalized, normalized))
    except ValueError:
        return None


def _status(value: Any) -> MatchStatus:
    normalized = str(value or "").strip().upper()
    if normalized in {"IN_PROGRESS", "LIVE", "IN PLAY"}:
        return MatchStatus.LIVE
    if normalized in {"COMPLETED", "FINISHED"}:
        return MatchStatus.FINISHED
    if normalized in {"RETIRED", "RETIREMENT"}:
        return MatchStatus.RETIRED
    if normalized == "WALKOVER":
        return MatchStatus.WALKOVER
    if normalized in {"CANCELLED", "CANCELED"}:
        return MatchStatus.CANCELLED
    if normalized == "POSTPONED":
        return MatchStatus.POSTPONED
    if normalized in {"SCHEDULED", "NOT_STARTED"}:
        return MatchStatus.SCHEDULED
    return MatchStatus.UNKNOWN


def _surface(value: Any) -> Surface:
    normalized = str(value or "").strip().lower()
    mapping = {
        "hard": Surface.HARD,
        "clay": Surface.CLAY,
        "grass": Surface.GRASS,
        "carpet": Surface.CARPET,
    }
    return mapping.get(normalized, Surface.UNKNOWN)


def normalize_cito_live_match(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a documented Cito live row into the provider-neutral match contract.

    Returns None when required identity fields are absent; it never fabricates
    player names, tour values, or match identifiers.
    """
    player1 = row.get("player1")
    player2 = row.get("player2")
    if not isinstance(player1, dict) or not isinstance(player2, dict):
        return None

    player1_name = player1.get("name")
    player2_name = player2.get("name")
    external_id = row.get("canonical_id") or row.get("id") or row.get("match_id")
    tour = _tour(row.get("tour"))
    if not all(isinstance(value, str) and value.strip() for value in (player1_name, player2_name, external_id)):
        return None
    if tour is None:
        return None

    scores: list[SetScore] = []
    sets = row.get("sets")
    if isinstance(sets, list):
        for item in sets:
            if not isinstance(item, dict):
                continue
            p1_games = item.get("player1_games")
            p2_games = item.get("player2_games")
            if not isinstance(p1_games, int) or not isinstance(p2_games, int):
                continue
            tb = item.get("tiebreak_points")
            tb_p1 = tb.get("player1") if isinstance(tb, dict) else None
            tb_p2 = tb.get("player2") if isinstance(tb, dict) else None
            scores.append(
                SetScore(
                    player_1_games=p1_games,
                    player_2_games=p2_games,
                    player_1_tiebreak=tb_p1 if isinstance(tb_p1, int) else None,
                    player_2_tiebreak=tb_p2 if isinstance(tb_p2, int) else None,
                )
            )

    winner = row.get("winner")
    winner_id = winner.get("canonical_id") or winner.get("id") if isinstance(winner, dict) else None
    match = MatchRecord(
        provider="cito",
        external_id=external_id,
        tour=tour,
        status=_status(row.get("status")),
        scheduled_at=_aware_datetime(row.get("scheduled_at")),
        tournament_id=row.get("canonical_tournament_id") or row.get("tournament_id"),
        tournament_name=row.get("tournament_display_name") or row.get("tournament_name"),
        round_name=row.get("round_name") or row.get("round"),
        surface=_surface(row.get("surface")),
        player_1_id=player1.get("canonical_id") or player1.get("id"),
        player_1_name=player1_name,
        player_2_id=player2.get("canonical_id") or player2.get("id"),
        player_2_name=player2_name,
        winner_id=winner_id,
        scores=scores,
        best_of=row.get("best_of") if isinstance(row.get("best_of"), int) else None,
        raw=row,
    )
    normalized = match.model_dump(mode="json", exclude={"raw"})
    # Live-only fields are retained alongside the canonical contract.
    normalized.update(
        {
            "live": {
                "current_set": row.get("current_set"),
                "game_score": row.get("game_score"),
                "server": row.get("server"),
                "tiebreak_score": row.get("tiebreak_score"),
                "player_1_current_game_score": player1.get("current_game_score"),
                "player_2_current_game_score": player2.get("current_game_score"),
                "player_1_is_serving": player1.get("is_serving"),
                "player_2_is_serving": player2.get("is_serving"),
                "court": row.get("court"),
                "last_updated_at": row.get("last_updated_at"),
                "is_simulated": row.get("is_simulated", False),
                "stats_available": row.get("stats_available"),
                "stats_unavailable_code": row.get("stats_unavailable_code"),
            }
        }
    )
    return normalized
