"""Join a Cito live score with linked bookmaker snapshots and price value bets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.live_probability import calculate_live_probabilities
from app.storage.models import OddsObservation, PredictionRecord, ProviderMatchLink, TennisMatch


POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4, "ADV": 4}


def _score_point(value: Any) -> int:
    if isinstance(value, int) and value in range(5):
        return value
    normalized = str(value or "").strip().upper()
    if normalized not in POINTS:
        raise ValueError(f"Unsupported live point score: {value!r}")
    return POINTS[normalized]


def _latest_complete_h2h(
    session: Session, match_id: str, player_1: str, player_2: str,
    bookmaker: str | None = None,
) -> list[dict[str, Any]]:
    if session.get(TennisMatch, -1) is not None:
        pass
    # The unified Tennis API uses one live event ID for both score and odds.
    # This avoids a fuzzy cross-vendor event-link step entirely.
    direct = session.scalars(select(OddsObservation).where(
        OddsObservation.provider == "tennis_api",
        OddsObservation.external_match_id == match_id,
        OddsObservation.market == "h2h",
    ).order_by(OddsObservation.captured_at.desc())).all()
    if direct:
        rows = direct
    else:
        links = session.scalars(select(ProviderMatchLink).where(
            ProviderMatchLink.canonical_provider == "cito",
            ProviderMatchLink.canonical_external_id == match_id,
            ProviderMatchLink.source_provider == "the_odds_api",
        )).all()
        event_ids = [link.source_external_id for link in links]
        if not event_ids:
            return []
        rows = session.scalars(select(OddsObservation).where(
            OddsObservation.provider == "the_odds_api",
            OddsObservation.external_match_id.in_(event_ids),
            OddsObservation.market == "h2h",
        ).order_by(OddsObservation.captured_at.desc())).all()
    grouped: dict[tuple[str, datetime], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if bookmaker and row.bookmaker != bookmaker:
            continue
        key = (row.bookmaker, row.captured_at)
        grouped[key][row.selection] = row
    candidates = []
    for (book, captured), selections in grouped.items():
        if player_1 in selections and player_2 in selections:
            candidates.append((captured, book, selections))
    if not candidates:
        return []
    captured, book, selections = max(candidates, key=lambda item: item[0])
    return [
        {"bookmaker": book, "captured_at": captured, "selection": player_1,
         "odds": selections[player_1].odds_decimal},
        {"bookmaker": book, "captured_at": captured, "selection": player_2,
         "odds": selections[player_2].odds_decimal},
    ]


def analyze_stored_live_match(
    session: Session, *, match_id: str, player_1_serve_point_win: float,
    player_2_serve_point_win: float, min_ev: float = 0.03,
    bookmaker: str | None = None, stake: float = 1.0,
    auto_record: bool = True,
) -> dict[str, Any]:
    match = session.scalar(select(TennisMatch).where(
        TennisMatch.external_id == match_id,
        TennisMatch.provider.in_({"cito", "tennis_api"}),
    ))
    if match is None:
        raise LookupError("Live match not found. Sync the Cito live board first.")
    if match.status != "live":
        raise ValueError("Only matches with live status can be analyzed.")
    raw = match.raw or {}
    p1 = raw.get("player1") or {}
    p2 = raw.get("player2") or {}
    live = raw
    current_set = live.get("current_set")
    if not isinstance(current_set, int) or current_set < 1 or current_set > len(match.scores):
        raise ValueError("Current set score is missing or inconsistent.")
    score = match.scores[current_set - 1]
    games_1, games_2 = score.get("player_1_games"), score.get("player_2_games")
    if not isinstance(games_1, int) or not isinstance(games_2, int):
        raise ValueError("Current set game score is unavailable.")
    server = live.get("server")
    if server not in (1, 2):
        if p1.get("is_serving") is True:
            server = 1
        elif p2.get("is_serving") is True:
            server = 2
        else:
            raise ValueError("Current server is not supplied by Cito.")
    in_tiebreak = games_1 == 6 and games_2 == 6
    tb_score = live.get("tiebreak_score")
    if in_tiebreak:
        if not isinstance(tb_score, dict):
            tb_score = {
                "player1": score.get("player_1_tiebreak"),
                "player2": score.get("player_2_tiebreak"),
            }
        if not isinstance(tb_score.get("player1"), int) or not isinstance(tb_score.get("player2"), int):
            raise ValueError("Tiebreak score is missing; refusing to infer it.")
        points_1 = points_2 = 0
        tb1, tb2 = tb_score["player1"], tb_score["player2"]
        # Cito's current server is known, but first-server identity is not
        # derivable from a single scoreboard snapshot. Require explicit field.
        tb_first = live.get("tiebreak_first_server")
        if tb_first not in (1, 2):
            raise ValueError("Tiebreak first server is unavailable; cannot price this state safely.")
    else:
        raw_game_score = live.get("game_score")
        if not isinstance(raw_game_score, str) or "-" not in raw_game_score:
            raw_game_score = f"{p1.get('current_game_score')}-{p2.get('current_game_score')}"
        left, right = raw_game_score.split("-", 1)
        points_1, points_2 = _score_point(left), _score_point(right)
        tb1 = tb2 = 0
        tb_first = None

    odds = _latest_complete_h2h(session, match_id, match.player_1_name,
                                match.player_2_name, bookmaker)
    if not odds:
        raise LookupError("No complete, linked, same-bookmaker h2h odds snapshot is available.")
    captured_at = odds[0]["captured_at"]
    # SQLite drops timezone metadata; persisted timestamps are normalized UTC.
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - captured_at.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -60 or age_seconds > 15 * 60:
        raise LookupError("Latest complete odds snapshot is older than 15 minutes; refresh odds before analysis.")

    result = calculate_live_probabilities(
        sets_player_1=sum(1 for item in match.scores if item["player_1_games"] > item["player_2_games"]),
        sets_player_2=sum(1 for item in match.scores if item["player_2_games"] > item["player_1_games"]),
        games_player_1=games_1, games_player_2=games_2,
        points_player_1=points_1, points_player_2=points_2,
        server=server, best_of=match.best_of or 3,
        player_1_serve_point_win=player_1_serve_point_win,
        player_2_serve_point_win=player_2_serve_point_win,
        tiebreak_points_player_1=tb1, tiebreak_points_player_2=tb2,
        tiebreak_first_server=tb_first,
        player_1_odds=odds[0]["odds"], player_2_odds=odds[1]["odds"],
        min_ev=min_ev,
    )
    market = result.get("market", {})
    candidates = []
    for idx, name in ((1, match.player_1_name), (2, match.player_2_name)):
        ev = market.get(f"player_{idx}_ev", -1)
        if market.get(f"player_{idx}_value_bet"):
            candidates.append({
                "selection": name, "probability": result[f"player_{idx}_match_probability"],
                "odds_decimal": odds[idx - 1]["odds"], "expected_value": ev,
            })
    created = []
    if auto_record:
        now = datetime.now(timezone.utc)
        for candidate in candidates:
            existing = session.scalar(select(PredictionRecord).where(
                PredictionRecord.event_id == match.external_id,
                PredictionRecord.market == "match_winner",
                PredictionRecord.selection == candidate["selection"],
                PredictionRecord.odds_decimal == candidate["odds_decimal"],
                PredictionRecord.status == "open",
                PredictionRecord.placed_at >= now.replace(microsecond=0) - timedelta(minutes=15),
            ).limit(1))
            if existing is not None:
                created.append(existing.id)
                continue
            row = PredictionRecord(
                event_id=match.external_id, market="match_winner",
                selection=candidate["selection"], probability=candidate["probability"],
                odds_decimal=candidate["odds_decimal"], stake=stake,
                placed_at=datetime.now(timezone.utc), status="open",
            )
            session.add(row)
            session.flush()
            created.append(row.id)
    return {
        "event_id": match.external_id,
        "players": [match.player_1_name, match.player_2_name],
        "score": {"sets": [result["player_1_match_probability"], result["player_2_match_probability"]],
                  "current_set_games": [games_1, games_2],
                  "current_game_points": [points_1, points_2],
                  "server": server, "tiebreak": in_tiebreak},
        "bookmaker": odds[0]["bookmaker"],
        "odds_captured_at": captured_at.isoformat(),
        "odds_age_seconds": round(age_seconds, 1),
        "probability_model": result,
        "value_bets": candidates,
        "journal_prediction_ids": created,
        "serve_point_win_rates_source": "explicit_request_inputs; not inferred from score feed",
    }
