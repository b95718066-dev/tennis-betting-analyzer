"""Generate pre-match forecasts directly from stored, as-of historical results."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.analytics.historical_elo import historical_elo_ratings
from app.analytics.prediction import predict_match
from app.analytics.probability import expected_value, remove_margin, value_bet
from app.storage.models import HistoricalMatchSnapshot


VALID_SURFACES = {"hard", "clay", "grass", "carpet"}


def _cutoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("before must be an ISO-8601 timestamp with timezone.")
    return parsed.astimezone(timezone.utc)


def _row_time(value: str | None) -> tuple[datetime | None, bool]:
    if not value:
        return None, False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(value[:10])
        except ValueError:
            return None, False
        return datetime.combine(parsed_date, time.max, tzinfo=timezone.utc), True
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return datetime.combine(parsed.date(), time.max, tzinfo=timezone.utc), True
    return parsed.astimezone(timezone.utc), False


def _is_before(value: str | None, cutoff: datetime) -> bool:
    parsed, date_only = _row_time(value)
    if parsed is None:
        return False
    if date_only:
        return parsed.date() < cutoff.date()
    return parsed < cutoff


def generate_pre_match_forecast(
    session: Session,
    *,
    player_1_id: str,
    player_2_id: str,
    surface: str,
    before: str,
    player_1_odds: float | None = None,
    player_2_odds: float | None = None,
    min_ev: float = 0.03,
) -> dict[str, Any]:
    """Build a forecast using stored Cito and Valuebetennis matches strictly before the cutoff."""
    if not player_1_id.strip() or not player_2_id.strip() or player_1_id == player_2_id:
        raise ValueError("Two distinct, non-empty player IDs are required.")
    normalized_surface = surface.strip().lower()
    if normalized_surface not in VALID_SURFACES:
        raise ValueError("surface must be hard, clay, grass or carpet.")
    cutoff = _cutoff(before)
    if (player_1_odds is None) != (player_2_odds is None):
        raise ValueError("Both player odds are required when supplying market prices.")

    ratings = historical_elo_ratings(
        session,
        player_id=player_1_id,
        opponent_id=player_2_id,
        before=cutoff.isoformat(),
        surface=normalized_surface,
    )
    rows = session.scalars(
        select(HistoricalMatchSnapshot).where(
            HistoricalMatchSnapshot.provider.in_(["cito", "valuebetennis"]),
            or_(
                HistoricalMatchSnapshot.player_1_id.in_([player_1_id, player_2_id]),
                HistoricalMatchSnapshot.player_2_id.in_([player_1_id, player_2_id]),
            ),
            HistoricalMatchSnapshot.match_date.is_not(None),
            HistoricalMatchSnapshot.winner_id.is_not(None),
        )
    ).all()

    eligible = [
        row for row in rows
        if _is_before(row.match_date, cutoff)
        and row.player_1_id != row.player_2_id
        and row.winner_id in {row.player_1_id, row.player_2_id}
    ]
    eligible.sort(key=lambda row: (_row_time(row.match_date)[0], row.id))

    histories: dict[str, list[dict[str, Any]]] = {
        player_1_id: [], player_2_id: [],
    }
    h2h_matches = []
    for row in eligible:
        p1, p2 = row.player_1_id, row.player_2_id
        normalized_match_surface = (row.surface or "").strip().lower()
        if p1 in histories:
            histories[p1].append({
                "result": "W" if row.winner_id == p1 else "L",
                "surface": normalized_match_surface,
            })
        if p2 in histories:
            histories[p2].append({
                "result": "W" if row.winner_id == p2 else "L",
                "surface": normalized_match_surface,
            })
        if {p1, p2} == {player_1_id, player_2_id}:
            h2h_matches.append({
                "player": p1, "opponent": p2,
                "winner": row.winner_id, "surface": normalized_match_surface,
            })

    prediction = predict_match(
        player=player_1_id,
        opponent=player_2_id,
        player_rating=ratings["player"]["effective_elo"],
        opponent_rating=ratings["opponent"]["effective_elo"],
        player_matches=histories[player_1_id],
        opponent_matches=histories[player_2_id],
        h2h_matches=h2h_matches,
        surface=normalized_surface,
    )
    result: dict[str, Any] = {
        **prediction,
        "forecast_type": "pre_match_historical",
        "as_of": cutoff.isoformat(),
        "historical_elo": ratings,
        "history_coverage": {
            "player_1_matches": len(histories[player_1_id]),
            "player_2_matches": len(histories[player_2_id]),
            "head_to_head_matches": len(h2h_matches),
            "stored_rows_used": len(eligible),
            "source": "stored Cito and Valuebetennis historical match snapshots",
            "date_only_policy": "date-only rows on the as-of calendar day are excluded",
        },
    }
    if player_1_odds is not None and player_2_odds is not None:
        fair_market = remove_margin({
            player_1_id: player_1_odds,
            player_2_id: player_2_odds,
        })
        p1_ev = expected_value(prediction["player_probability"], player_1_odds)
        p2_ev = expected_value(prediction["opponent_probability"], player_2_odds)
        result["betting"] = {
            "market_probability_without_margin": fair_market,
            "player_1_ev": round(p1_ev, 4),
            "player_2_ev": round(p2_ev, 4),
            "player_1_value_bet": value_bet(prediction["player_probability"], player_1_odds, min_ev),
            "player_2_value_bet": value_bet(prediction["opponent_probability"], player_2_odds, min_ev),
            "minimum_ev_threshold": min_ev,
            "warning": "Experimental model probability; value flags are not validated betting recommendations.",
        }
    return result


def search_historical_players(
    session: Session, *, query: str, limit: int = 20, tour: str | None = None,
) -> dict[str, Any]:
    """Search the player identities already present in the historical archive."""
    normalized_query = query.strip().casefold()
    if len(normalized_query) < 2:
        raise ValueError("Player search query must contain at least two characters.")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50.")
    if tour is not None and tour.upper() not in {"ATP", "WTA"}:
        raise ValueError("tour must be ATP or WTA.")

    statement = select(HistoricalMatchSnapshot).where(
        HistoricalMatchSnapshot.provider.in_(["cito", "valuebetennis"]),
        HistoricalMatchSnapshot.player_1_id.is_not(None),
        HistoricalMatchSnapshot.player_2_id.is_not(None),
    ).order_by(HistoricalMatchSnapshot.match_date.desc())
    if tour:
        statement = statement.where(HistoricalMatchSnapshot.tour == tour.upper())

    players: dict[str, dict[str, Any]] = {}
    for row in session.scalars(statement).all():
        raw = row.raw if isinstance(row.raw, dict) else {}
        for side, player_id in ((1, row.player_1_id), (2, row.player_2_id)):
            key = f"player{side}"
            node = raw.get(key) or raw.get(f"player_{side}") or {}
            if not isinstance(node, dict) or not player_id:
                continue
            name = (node.get("name") or node.get("full_name") or node.get("display_name") or raw.get(f"player_{side}_name"))
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if normalized_query not in name.casefold():
                continue
            entry = players.setdefault(player_id, {
                "player_id": player_id,
                "name": name,
                "tour": row.tour,
                "matches_in_archive": 0,
            })
            entry["matches_in_archive"] += 1

    results = sorted(
        players.values(),
        key=lambda item: (-item["matches_in_archive"], item["name"].casefold()),
    )[:limit]
    return {
        "count": len(results),
        "query": query,
        "tour": tour.upper() if tour else "ALL",
        "players": results,
        "source": "stored Cito and Valuebetennis historical match snapshots",
        "archive_note": "Search results combine Cito and Valuebetennis players already present in the local archive; source-prefixed IDs keep identities separate.",
    }
