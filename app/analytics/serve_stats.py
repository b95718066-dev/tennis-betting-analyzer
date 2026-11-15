"""Normalize verified Cito live serve statistics without fabricating missing values."""

from __future__ import annotations

from typing import Any

from app.analytics.sample_confidence import assess_serve_sample


SERVE_WON_KEYS = {
    "service_points_won", "serve_points_won", "service_points_won_count",
}
SERVE_PLAYED_KEYS = {
    "service_points_played", "serve_points", "service_points", "serve_points_played",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def _metric_pair(node: Any) -> tuple[float, float] | None:
    """Read a count pair from a metric object or a direct count mapping."""
    if not isinstance(node, dict):
        return None
    normalized = {str(key).strip().lower(): value for key, value in node.items()}
    won = next((_number(normalized[key]) for key in SERVE_WON_KEYS if key in normalized), None)
    played = next((_number(normalized[key]) for key in SERVE_PLAYED_KEYS if key in normalized), None)
    if won is not None and played is not None and played > 0 and won <= played:
        return won, played
    numerator = _number(normalized.get("numerator"))
    denominator = _number(normalized.get("denominator"))
    if numerator is not None and denominator is not None and denominator > 0 and numerator <= denominator:
        return numerator, denominator
    return None


def _side_stats(stats: Any, side: int) -> tuple[float, float] | None:
    side_names = (f"player{side}", f"player_{side}", f"side{side}", f"side_{side}")
    if isinstance(stats, dict):
        for name in side_names:
            if name in stats:
                # Do not fall back to another player's valid metric when this side
                # is missing or internally inconsistent.
                return _find_serve_pair(stats[name])
        return None
    if isinstance(stats, list):
        for row in stats:
            if not isinstance(row, dict):
                continue
            label = str(row.get("key") or row.get("name") or row.get("stat") or "").lower()
            label = label.replace("-", "_").replace(" ", "_")
            # Cito's live catalog represents each statistic as one row with
            # per-player numerator/denominator objects. Only accept the
            # service-points-won row; percentage-only rows are not sufficient.
            if label in SERVE_WON_KEYS:
                # The live-board response can flatten each metric into one row:
                # {"key":"service_points_won","side":1,"numerator":18,"denominator":29}.
                # Accept only the explicitly tagged side; never infer the other side.
                row_side = row.get("side")
                if row_side == side:
                    found = _metric_pair(row)
                    if found:
                        return found
                for name in side_names:
                    if name in row:
                        found = _metric_pair(row[name])
                        if found:
                            return found
    return None


def _find_serve_pair(node: Any) -> tuple[float, float] | None:
    direct = _metric_pair(node)
    if direct:
        return direct
    if isinstance(node, dict):
        # Only inspect nested metric objects; no percentage is guessed from unrelated stats.
        for value in node.values():
            found = _find_serve_pair(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_serve_pair(value)
            if found:
                return found
    return None


def normalize_live_serve_rates(payload: dict[str, Any]) -> dict[str, Any]:
    """Return per-player service-point win rates when Cito supplies valid counts.

    Rates are computed only as service points won / service points played.
    Missing, unavailable, or zero-denominator statistics remain unavailable.
    """
    stats = payload.get("stats")
    available = payload.get("stats_available")
    unavailable_code = payload.get("stats_unavailable_code")
    players = []
    for side in (1, 2):
        pair = _side_stats(stats, side)
        if pair is None:
            players.append({
                "side": side, "serve_point_win_rate": None,
                "service_points_won": None, "service_points_played": None,
                "available": False,
            })
            continue
        won, played = pair
        players.append({
            "side": side, "serve_point_win_rate": won / played,
            "service_points_won": int(won), "service_points_played": int(played),
            "sample_confidence": assess_serve_sample(int(won), int(played)),
            "available": True,
        })
    complete = all(player["available"] for player in players)
    return {
        "provider": "cito",
        "stats_available": bool(available) if available is not None else complete,
        "stats_unavailable_code": unavailable_code,
        "serve_point_win_rates": players,
        "ready_for_live_model": complete,
        "source": "cito_live_match_stats",
        "limitations": [] if complete else ["Cito has not supplied valid service-point won/played counts for both players."],
    }
