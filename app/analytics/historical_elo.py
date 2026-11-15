"""Chronological as-of Elo ratings replayed from stored historical match outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.elo import DEFAULT_K_FACTOR, DEFAULT_RATING, expected_probability, update_elo
from app.storage.models import HistoricalMatchSnapshot

SURFACE_WEIGHT = 0.35
MIN_SURFACE_MATCHES = 5
VALID_SURFACES = {"hard", "clay", "grass", "carpet"}


def _parse_cutoff(before: str) -> datetime:
    try:
        cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    return cutoff.astimezone(timezone.utc)


def _eligible_date(value: str | None, cutoff: datetime) -> bool:
    if not value:
        return False
    try:
        match_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if match_time.tzinfo is None or match_time.utcoffset() is None:
        return match_time.date() < cutoff.date()
    return match_time.astimezone(timezone.utc) < cutoff


def historical_elo_ratings(
    session: Session, *, player_id: str, opponent_id: str, before: str,
    surface: str, k_factor: float = DEFAULT_K_FACTOR,
) -> dict[str, Any]:
    """Replay completed matches strictly before cutoff; no future results enter ratings."""
    if not player_id.strip() or not opponent_id.strip() or player_id == opponent_id:
        raise ValueError("Two distinct non-empty player IDs are required.")
    if surface.lower() not in VALID_SURFACES:
        raise ValueError("surface must be hard, clay, grass or carpet.")
    cutoff = _parse_cutoff(before)
    rows = session.scalars(
        select(HistoricalMatchSnapshot).where(
            HistoricalMatchSnapshot.provider == "cito",
            HistoricalMatchSnapshot.match_date.is_not(None),
            HistoricalMatchSnapshot.winner_id.is_not(None),
            HistoricalMatchSnapshot.player_1_id.is_not(None),
            HistoricalMatchSnapshot.player_2_id.is_not(None),
        ).order_by(HistoricalMatchSnapshot.match_date.asc(), HistoricalMatchSnapshot.id.asc())
    ).all()

    overall: dict[str, float] = {}
    surface_ratings: dict[str, dict[str, float]] = {}
    overall_matches: dict[str, int] = {}
    surface_matches: dict[str, dict[str, int]] = {}
    used_matches = 0
    for row in rows:
        if not _eligible_date(row.match_date, cutoff):
            continue
        p1, p2, winner = row.player_1_id, row.player_2_id, row.winner_id
        if winner not in {p1, p2} or p1 == p2:
            continue
        r1 = overall.get(p1, DEFAULT_RATING)
        r2 = overall.get(p2, DEFAULT_RATING)
        n1, n2 = update_elo(r1, r2, 1.0 if winner == p1 else 0.0, k_factor)
        overall[p1], overall[p2] = n1, n2
        overall_matches[p1] = overall_matches.get(p1, 0) + 1
        overall_matches[p2] = overall_matches.get(p2, 0) + 1
        normalized_surface = (row.surface or "").lower()
        if normalized_surface in VALID_SURFACES:
            ratings = surface_ratings.setdefault(normalized_surface, {})
            counts = surface_matches.setdefault(normalized_surface, {})
            s1, s2 = ratings.get(p1, DEFAULT_RATING), ratings.get(p2, DEFAULT_RATING)
            sn1, sn2 = update_elo(s1, s2, 1.0 if winner == p1 else 0.0, k_factor)
            ratings[p1], ratings[p2] = sn1, sn2
            counts[p1] = counts.get(p1, 0) + 1
            counts[p2] = counts.get(p2, 0) + 1
        if player_id in {p1, p2} or opponent_id in {p1, p2}:
            used_matches += 1

    def player_result(pid: str) -> dict[str, Any]:
        base = overall.get(pid, DEFAULT_RATING)
        surface_rating = surface_ratings.get(surface.lower(), {}).get(pid, DEFAULT_RATING)
        surface_count = surface_matches.get(surface.lower(), {}).get(pid, 0)
        use_surface = surface_count >= MIN_SURFACE_MATCHES
        adjusted = base * (1 - SURFACE_WEIGHT) + surface_rating * SURFACE_WEIGHT if use_surface else base
        return {
            "player_id": pid,
            "overall_elo": round(base, 2),
            "overall_matches": overall_matches.get(pid, 0),
            "surface": surface.lower(),
            "surface_elo": round(surface_rating, 2) if surface_count else None,
            "surface_matches": surface_count,
            "surface_adjustment_used": use_surface,
            "effective_elo": round(adjusted, 2),
        }

    p = player_result(player_id)
    o = player_result(opponent_id)
    probability = expected_probability(p["effective_elo"], o["effective_elo"])
    return {
        "provider": "cito_historical_match_snapshots",
        "cutoff_exclusive": cutoff.isoformat(),
        "surface": surface.lower(),
        "player": p,
        "opponent": o,
        "elo_difference": round(p["effective_elo"] - o["effective_elo"], 2),
        "player_win_probability_elo": round(probability, 6),
        "replay_matches_for_target_players": used_matches,
        "method": {
            "initial_rating": DEFAULT_RATING,
            "k_factor": k_factor,
            "surface_weight": SURFACE_WEIGHT,
            "minimum_surface_matches": MIN_SURFACE_MATCHES,
            "ordering": "match_date ascending; same-timestamp rows use stored ID order",
        },
        "limitations": [
            "This is a transparent baseline Elo replay, not a trained or calibrated betting model.",
            "Only stored Cito match outcomes with resolved player IDs and winner IDs are used.",
            "Date-only matches on the cutoff calendar day are excluded.",
            "No inactivity decay, tournament-level K factor, retirement adjustment or doubles modeling is applied.",
        ],
    }
