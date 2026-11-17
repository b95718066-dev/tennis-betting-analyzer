"""Read-only, chronological experiments against stored historical tennis data.

No rows are written or modified. Predictions are generated from features available
before the calendar day of each match; same-day results/statistics are applied only
after every match for that day has been scored.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from itertools import groupby
from math import isfinite, log
from statistics import median
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.elo import DEFAULT_RATING, expected_probability
from app.analytics.live_probability import calculate_live_probabilities
from app.analytics.probability import remove_margin
from app.storage.models import HistoricalMatchSnapshot
from experiments.model_01_market_anchored.model import MarketAnchoredLogit
from experiments.model_03_clv_lab.clv import calculate_clv

K_FACTOR = 32.0
SURFACE_WEIGHT = 0.35
MIN_SURFACE_MATCHES = 5
MIN_PRIOR_SERVE_POINTS = 50
SERVE_PRIOR_STRENGTH = 100.0


def _day(value: str | None) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


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


def _number(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            continue
        if isfinite(parsed) and parsed > 1.0:
            return parsed
    return None


def _surface(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    aliases = {
        "hard court": "hard", "hardcourt": "hard", "indoor hard": "hard",
        "clay court": "clay", "red clay": "clay",
        "grass court": "grass",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"hard", "clay", "grass", "carpet"} else "unknown"


def _score_metrics(rows: list[tuple[float, bool]], bins: int = 10) -> dict[str, Any] | None:
    if not rows:
        return None
    n = len(rows)
    eps = 1e-15
    brier = sum((p - float(y)) ** 2 for p, y in rows) / n
    log_loss = -sum(
        float(y) * log(max(p, eps)) + (1.0 - float(y)) * log(max(1.0 - p, eps))
        for p, y in rows
    ) / n
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, y in rows:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    calibration = []
    for index, items in sorted(buckets.items()):
        predicted = sum(p for p, _ in items) / len(items)
        observed = sum(float(y) for _, y in items) / len(items)
        calibration.append({
            "bin_lower": index / bins,
            "bin_upper": (index + 1) / bins,
            "count": len(items),
            "mean_predicted_probability": predicted,
            "observed_win_rate": observed,
            "absolute_error": abs(predicted - observed),
        })
    return {
        "matches": n,
        "accuracy": sum((p >= 0.5) == y for p, y in rows) / n,
        "brier_score": brier,
        "brier_skill_vs_50pct": 1.0 - brier / 0.25,
        "log_loss": log_loss,
        "calibration": calibration,
    }


def _valuebetennis_rows(session: Session) -> Iterable[Any]:
    statement = select(
        HistoricalMatchSnapshot.id,
        HistoricalMatchSnapshot.match_date,
        HistoricalMatchSnapshot.tour,
        HistoricalMatchSnapshot.surface,
        HistoricalMatchSnapshot.player_1_id,
        HistoricalMatchSnapshot.player_2_id,
        HistoricalMatchSnapshot.winner_id,
        HistoricalMatchSnapshot.raw,
    ).where(
        HistoricalMatchSnapshot.provider == "valuebetennis",
        HistoricalMatchSnapshot.match_date.is_not(None),
        HistoricalMatchSnapshot.player_1_id.is_not(None),
        HistoricalMatchSnapshot.player_2_id.is_not(None),
        HistoricalMatchSnapshot.winner_id.is_not(None),
    ).order_by(HistoricalMatchSnapshot.match_date.asc(), HistoricalMatchSnapshot.id.asc())
    return session.execute(statement.execution_options(yield_per=5000))


def run_market_anchored_and_clv(session: Session) -> dict[str, Any]:
    """Build daily as-of Elo features, train on earlier dates, score later dates."""
    overall: dict[str, float] = {}
    overall_count: dict[str, int] = defaultdict(int)
    surface_rating: dict[tuple[str, str], float] = {}
    surface_count: dict[tuple[str, str], int] = defaultdict(int)
    forecast_rows: list[dict[str, Any]] = []
    clv_rows: list[dict[str, float]] = []
    total_rows = 0
    date_only_rows = 0
    valid_outcomes = 0

    rows = _valuebetennis_rows(session)
    for day, day_iter in groupby(rows, key=lambda row: _day(row.match_date)):
        batch = list(day_iter)
        if day is None:
            continue
        overall_delta: dict[str, float] = defaultdict(float)
        surface_delta: dict[tuple[str, str], float] = defaultdict(float)

        # All forecasts for a date use a frozen rating state from prior dates.
        for row in batch:
            total_rows += 1
            p1, p2, winner = row.player_1_id, row.player_2_id, row.winner_id
            if not p1 or not p2 or p1 == p2 or winner not in {p1, p2}:
                continue
            valid_outcomes += 1
            surface = _surface(row.surface)
            r1 = overall.get(p1, DEFAULT_RATING)
            r2 = overall.get(p2, DEFAULT_RATING)
            sr1 = surface_rating.get((surface, p1), DEFAULT_RATING)
            sr2 = surface_rating.get((surface, p2), DEFAULT_RATING)
            use_surface = (
                surface != "unknown"
                and surface_count[(surface, p1)] >= MIN_SURFACE_MATCHES
                and surface_count[(surface, p2)] >= MIN_SURFACE_MATCHES
            )
            effective_1 = r1 * (1 - SURFACE_WEIGHT) + sr1 * SURFACE_WEIGHT if use_surface else r1
            effective_2 = r2 * (1 - SURFACE_WEIGHT) + sr2 * SURFACE_WEIGHT if use_surface else r2

            # Batch all same-day Elo updates from the same frozen pre-day ratings.
            expected = expected_probability(r1, r2)
            actual = 1.0 if winner == p1 else 0.0
            delta = K_FACTOR * (actual - expected)
            overall_delta[p1] += delta
            overall_delta[p2] -= delta

            if surface != "unknown":
                se1 = surface_rating.get((surface, p1), DEFAULT_RATING)
                se2 = surface_rating.get((surface, p2), DEFAULT_RATING)
                surface_expected = expected_probability(se1, se2)
                surface_delta[(surface, p1)] += K_FACTOR * (actual - surface_expected)
                surface_delta[(surface, p2)] -= K_FACTOR * (actual - surface_expected)

            raw = row.raw if isinstance(row.raw, dict) else {}
            opening_1 = _number(raw, "opening_odds_player_1", "cote1_ouverture")
            opening_2 = _number(raw, "opening_odds_player_2", "cote2_ouverture")
            closing_1 = _number(raw, "closing_odds_player_1", "cote1_cloture")
            closing_2 = _number(raw, "closing_odds_player_2", "cote2_cloture")

            if opening_1 and opening_2:
                # Source archive timestamps are not verified; exclude date-only rows.
                if isinstance(row.match_date, str) and len(row.match_date.strip()) > 10:
                    market = remove_margin({"p1": opening_1, "p2": opening_2})
                    forecast_rows.append({
                        "date": day.isoformat(),
                        "market_probability": market["p1"],
                        "features": {
                            "overall_elo_diff_scaled": (effective_1 - effective_2) / 400.0,
                            "surface_elo_diff_scaled": (sr1 - sr2) / 400.0 if use_surface else 0.0,
                        },
                        "outcome": winner == p1,
                    })
                else:
                    date_only_rows += 1

            if opening_1 and opening_2 and closing_1 and closing_2:
                try:
                    clv_rows.append(calculate_clv(
                        entry_odds={"p1": opening_1, "p2": opening_2},
                        closing_odds={"p1": closing_1, "p2": closing_2},
                        selection="p1" if winner == p1 else "p2",
                    ))
                except ValueError:
                    continue

        # Apply outcomes only after every match for the date has been scored.
        for player_id, delta in overall_delta.items():
            overall[player_id] = overall.get(player_id, DEFAULT_RATING) + delta
        for key, delta in surface_delta.items():
            surface_rating[key] = surface_rating.get(key, DEFAULT_RATING) + delta
        for row in batch:
            p1, p2 = row.player_1_id, row.player_2_id
            if not p1 or not p2 or p1 == p2 or row.winner_id not in {p1, p2}:
                continue
            overall_count[p1] += 1
            overall_count[p2] += 1
            surface = _surface(row.surface)
            if surface != "unknown":
                surface_count[(surface, p1)] += 1
                surface_count[(surface, p2)] += 1

    forecast_rows.sort(key=lambda item: item["date"])
    unique_days = sorted({row["date"] for row in forecast_rows})
    if len(unique_days) < 2:
        return {
            "status": "insufficient_chronological_coverage",
            "archive_rows": total_rows,
            "rows_with_valid_outcome": valid_outcomes,
            "market_forecast_rows": len(forecast_rows),
            "clv": _summarize_clv(clv_rows),
        }

    split_index = min(max(int(len(unique_days) * 0.60), 1), len(unique_days) - 1)
    cutoff_day = unique_days[split_index]
    train = [row for row in forecast_rows if row["date"] < cutoff_day]
    test = [row for row in forecast_rows if row["date"] >= cutoff_day]
    if len(train) < 30 or not test:
        return {
            "status": "insufficient_train_or_test_rows",
            "archive_rows": total_rows,
            "market_forecast_rows": len(forecast_rows),
            "training_rows": len(train),
            "test_rows": len(test),
            "clv": _summarize_clv(clv_rows),
        }

    model = MarketAnchoredLogit.fit(
        train, learning_rate=0.03, epochs=3000, l2=1.0
    )
    market_predictions = [(row["market_probability"], row["outcome"]) for row in test]
    model_predictions = [
        (model.predict_probability(row), row["outcome"]) for row in test
    ]
    return {
        "status": "evaluated",
        "data": {
            "provider": "valuebetennis",
            "archive_rows": total_rows,
            "rows_with_valid_outcome": valid_outcomes,
            "rows_with_opening_market_and_timestamped_match_time": len(forecast_rows),
            "date_only_rows_excluded_from_forecast": date_only_rows,
            "chronological_split": {
                "training_rows": len(train),
                "test_rows": len(test),
                "training_last_date": train[-1]["date"],
                "test_first_date": test[0]["date"],
                "train_fraction_by_unique_dates": 0.60,
            },
        },
        "out_of_sample": {
            "de_vigged_opening_market_baseline": _score_metrics(market_predictions),
            "market_anchored_logistic": _score_metrics(model_predictions),
        },
        "fitted_model": model.to_dict(),
        "clv_archive_comparison": _summarize_clv(clv_rows),
        "limitations": [
            "Valuebetennis opening/closing odds have no verified capture timestamps or bookmaker-level quote identity.",
            "Opening odds are a research market baseline, not verified executable prices at a known placement time.",
            "CLV is an archive opening-versus-closing comparison, not proof of beatable closing prices.",
            "Elo features use only completed match outcomes from earlier calendar dates; all same-day outcomes are applied after that day's forecasts.",
            "This is a retrospective single chronological holdout, not proof of profitability or a calibrated deployment model.",
        ],
    }


def _summarize_clv(rows: list[dict[str, float]]) -> dict[str, Any] | None:
    if not rows:
        return None
    probability = [float(row["probability_clv"]) for row in rows]
    odds = [float(row["odds_clv"]) for row in rows]
    beat = sum(bool(row["beat_closing_price"]) for row in rows)
    return {
        "matches": len(rows),
        "mean_probability_clv": sum(probability) / len(probability),
        "median_probability_clv": median(probability),
        "mean_odds_clv": sum(odds) / len(odds),
        "median_odds_clv": median(odds),
        "beat_closing_price_rate": beat / len(rows),
        "positive_probability_clv_count": sum(value > 0 for value in probability),
        "method": "de-vigged opening and closing prices from the same archived match row",
        "timestamp_validation": "not_verified",
    }


def _cito_rows(session: Session) -> Iterable[Any]:
    statement = select(
        HistoricalMatchSnapshot.id,
        HistoricalMatchSnapshot.match_date,
        HistoricalMatchSnapshot.tour,
        HistoricalMatchSnapshot.player_1_id,
        HistoricalMatchSnapshot.player_2_id,
        HistoricalMatchSnapshot.winner_id,
        HistoricalMatchSnapshot.stats_available,
        HistoricalMatchSnapshot.raw,
    ).where(
        HistoricalMatchSnapshot.provider == "cito",
        HistoricalMatchSnapshot.match_date.is_not(None),
        HistoricalMatchSnapshot.player_1_id.is_not(None),
        HistoricalMatchSnapshot.player_2_id.is_not(None),
        HistoricalMatchSnapshot.winner_id.is_not(None),
    ).order_by(HistoricalMatchSnapshot.match_date.asc(), HistoricalMatchSnapshot.id.asc())
    return session.execute(statement.execution_options(yield_per=2000))


def run_historical_markov(session: Session) -> dict[str, Any]:
    """Evaluate pre-match Markov forecasts using only earlier-day serve statistics."""
    player_serve: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    tour_serve: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions: list[tuple[float, bool]] = []
    rows_seen = 0
    stats_updates = 0
    skipped_unknown_time = 0
    skipped_insufficient_prior = 0

    rows = _cito_rows(session)
    for day, day_iter in groupby(rows, key=lambda row: _day(row.match_date)):
        batch = list(day_iter)
        if day is None:
            continue

        # No statistics from this calendar day are visible to any forecast on it.
        for row in batch:
            rows_seen += 1
            raw = row.raw if isinstance(row.raw, dict) else {}
            scheduled = _aware_datetime(raw.get("scheduled_at"))
            if (
                scheduled is None
                or raw.get("start_time_known") is not True
                or raw.get("date_is_tournament_start") is True
            ):
                skipped_unknown_time += 1
                continue
            best_of = raw.get("best_of")
            if best_of not in (3, 5):
                continue
            p1, p2 = row.player_1_id, row.player_2_id
            if not p1 or not p2 or p1 == p2 or row.winner_id not in {p1, p2}:
                continue
            tour = (row.tour or "").upper()
            a = player_serve[(tour, p1)]
            b = player_serve[(tour, p2)]
            if min(a[1], b[1]) < MIN_PRIOR_SERVE_POINTS:
                skipped_insufficient_prior += 1
                continue
            baseline = tour_serve[tour]
            if baseline[1] <= 0:
                skipped_insufficient_prior += 1
                continue
            prior_rate = baseline[0] / baseline[1]
            p1_rate = (a[0] + SERVE_PRIOR_STRENGTH * prior_rate) / (a[1] + SERVE_PRIOR_STRENGTH)
            p2_rate = (b[0] + SERVE_PRIOR_STRENGTH * prior_rate) / (b[1] + SERVE_PRIOR_STRENGTH)
            if not (0.0 < p1_rate < 1.0 and 0.0 < p2_rate < 1.0):
                continue

            # The opening server is unknown pre-match. Average both legal first-server states.
            try:
                first_server_1 = calculate_live_probabilities(
                    sets_player_1=0, sets_player_2=0,
                    games_player_1=0, games_player_2=0,
                    points_player_1=0, points_player_2=0,
                    server=1, best_of=best_of,
                    player_1_serve_point_win=p1_rate,
                    player_2_serve_point_win=p2_rate,
                )["player_1_match_probability"]
                first_server_2 = calculate_live_probabilities(
                    sets_player_1=0, sets_player_2=0,
                    games_player_1=0, games_player_2=0,
                    points_player_1=0, points_player_2=0,
                    server=2, best_of=best_of,
                    player_1_serve_point_win=p1_rate,
                    player_2_serve_point_win=p2_rate,
                )["player_1_match_probability"]
            except ValueError:
                continue
            predictions.append(((first_server_1 + first_server_2) / 2.0, row.winner_id == p1))

        # After the whole day has been forecast, ingest that day's completed match stats.
        for row in batch:
            if row.stats_available is not True:
                continue
            raw = row.raw if isinstance(row.raw, dict) else {}
            scheduled = _aware_datetime(raw.get("scheduled_at"))
            if (
                scheduled is None
                or raw.get("start_time_known") is not True
                or raw.get("date_is_tournament_start") is True
            ):
                continue
            stats = raw.get("stats")
            if not isinstance(stats, dict):
                continue
            tour = (row.tour or "").upper()
            for side, player_id in (("player1", row.player_1_id), ("player2", row.player_2_id)):
                item = stats.get(side)
                if not player_id or not isinstance(item, dict):
                    continue
                won = item.get("service_points_won")
                played = item.get("service_points_played")
                if not isinstance(won, (int, float)) or not isinstance(played, (int, float)):
                    continue
                if played <= 0 or won < 0 or won > played:
                    continue
                player_serve[(tour, player_id)][0] += int(won)
                player_serve[(tour, player_id)][1] += int(played)
                tour_serve[tour][0] += int(won)
                tour_serve[tour][1] += int(played)
                stats_updates += 1

    return {
        "status": "evaluated" if predictions else "insufficient_prior_statistics",
        "data": {
            "provider": "cito",
            "rows_seen": rows_seen,
            "usable_pre_match_predictions": len(predictions),
            "player_side_stat_updates": stats_updates,
            "minimum_prior_service_points_per_player": MIN_PRIOR_SERVE_POINTS,
            "prior_shrinkage_pseudo_points": SERVE_PRIOR_STRENGTH,
            "skipped_unknown_or_tournament_start_time": skipped_unknown_time,
            "skipped_insufficient_prior_serve_sample": skipped_insufficient_prior,
            "same_day_policy": "forecasts first; update serve statistics only after every match that calendar day",
        },
        "metrics": _score_metrics(predictions),
        "limitations": [
            "Only Cito matches with a verified scheduled_at and start_time_known flag are scored.",
            "Only earlier-calendar-day service-point counts are used; same-day match statistics are withheld.",
            "The model assumes constant serve-point rates, standard advantage scoring and standard 7-point tiebreaks.",
            "The first server is unknown before the match, so both starting-server states are averaged.",
            "This is a small, recent, experimental sample; it is not a calibrated or proven betting model.",
        ],
    }


def run_all_historical_experiments(session: Session) -> dict[str, Any]:
    return {
        "market_anchored_and_clv": run_market_anchored_and_clv(session),
        "pre_match_markov": run_historical_markov(session),
        "read_only": True,
        "database_writes": False,
    }
