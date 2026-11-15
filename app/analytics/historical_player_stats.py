"""Leakage-safe historical serve/return aggregation from stored Cito match snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.analytics.sample_confidence import assess_serve_sample
from app.storage.models import HistoricalMatchSnapshot

# Cito match-stat catalog keys are explicit; percentages and display strings are never used.
METRICS = {
    "service_points_won": {"service_points_won", "serve_points_won"},
    "service_points_played": {"service_points_played", "serve_points", "service_points"},
    "return_points_won": {"return_points_won"},
    "return_points_played": {"return_points_played", "return_points"},
}


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _metric_row(stats: Any, aliases: set[str], side: int) -> tuple[int, int] | None:
    if not isinstance(stats, list):
        return None
    side_keys = (f"player{side}", f"player_{side}", f"side{side}", f"side_{side}")
    for item in stats:
        if not isinstance(item, dict):
            continue
        label = str(item.get("key") or item.get("name") or item.get("stat") or "").strip().lower()
        label = label.replace("-", "_").replace(" ", "_")
        if label not in aliases:
            continue
        node = next((item[k] for k in side_keys if k in item), None)
        if not isinstance(node, dict):
            return None
        won = _count(node.get("numerator"))
        played = _count(node.get("denominator"))
        if won is None or played is None or played <= 0 or won > played:
            return None
        return won, played
    return None


def _as_of_rows(session: Session, player_id: str, before: str, surface: str | None, limit: int):
    try:
        cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    query = select(HistoricalMatchSnapshot).where(
        HistoricalMatchSnapshot.provider == "cito",
        or_(HistoricalMatchSnapshot.player_1_id == player_id,
            HistoricalMatchSnapshot.player_2_id == player_id),
        HistoricalMatchSnapshot.match_date.is_not(None),
    )
    if surface:
        query = query.where(HistoricalMatchSnapshot.surface == surface.lower())
    candidates = session.scalars(query.order_by(HistoricalMatchSnapshot.match_date.desc())).all()
    rows = []
    utc_cutoff = cutoff.astimezone(timezone.utc)
    for row in candidates:
        try:
            match_time = datetime.fromisoformat(row.match_date.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
        if match_time.tzinfo is None or match_time.utcoffset() is None:
            eligible = match_time.date() < utc_cutoff.date()
        else:
            eligible = match_time.astimezone(timezone.utc) < utc_cutoff
        if eligible and row.stats_available is True and isinstance(row.raw, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows, utc_cutoff


def historical_player_stats(
    session: Session, *, player_id: str, before: str,
    surface: str | None = None, limit: int = 200,
) -> dict[str, Any]:
    """Aggregate only Cito count pairs from completed historical snapshots before cutoff."""
    if not player_id.strip():
        raise ValueError("player_id is required.")
    rows, cutoff = _as_of_rows(session, player_id, before, surface, limit)
    side_totals = {name: [0, 0] for name in METRICS}
    matches_with_stats = 0
    for row in rows:
        side = 1 if row.player_1_id == player_id else 2 if row.player_2_id == player_id else None
        if side is None:
            continue
        match_used = False
        stats = row.raw.get("stats")
        for metric, aliases in METRICS.items():
            pair = _metric_row(stats, aliases, side)
            if pair:
                side_totals[metric][0] += pair[0]
                side_totals[metric][1] += pair[1]
                match_used = True
        matches_with_stats += int(match_used)

    result = {}
    for metric, (won, played) in side_totals.items():
        result[metric] = {
            "numerator": won, "denominator": played,
            "rate": won / played if played else None,
            "sample_confidence": assess_serve_sample(won, played) if played and metric in {
                "service_points_won", "return_points_won"
            } and won <= played else None,
        }
    return {
        "provider": "cito",
        "player_id": player_id,
        "cutoff_exclusive": cutoff.isoformat(),
        "surface": surface.lower() if surface else None,
        "matches_scanned": len(rows),
        "matches_with_usable_stats": matches_with_stats,
        "metrics": result,
        "source": "cito_historical_match_snapshots.raw.stats",
        "limitations": [
            "Only explicit numerator/denominator count pairs are aggregated; percentages and display values are ignored.",
            "Date-only matches on the cutoff calendar day are excluded to prevent temporal leakage.",
            "Only rows with stats_available=true are considered; Cito statistics may be missing or published late.",
            "Historical rates are diagnostics and are not automatically blended into live model inputs.",
        ],
    }
