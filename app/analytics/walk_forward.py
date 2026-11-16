"""Leakage-safe expanding-window evaluation for historical Elo.

Date-only archive rows can update ratings only after their calendar day. They
are never scored out-of-sample because their true pre-match time is unknown.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timezone
from math import log
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.elo import DEFAULT_K_FACTOR, DEFAULT_RATING, expected_probability, update_elo
from app.analytics.historical_elo import MIN_SURFACE_MATCHES, SURFACE_WEIGHT, VALID_SURFACES
from app.storage.models import HistoricalMatchSnapshot


def _parse_match_date(
    value: str | None, provider: str,
) -> tuple[datetime | None, bool]:
    """Parse source timestamps; Valuebetennis documents naive CSV datetimes as UTC."""
    if not value:
        return None, False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            day = date.fromisoformat(value[:10])
        except ValueError:
            return None, False
        return datetime.combine(day, time.max, tzinfo=timezone.utc), True
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if provider == "valuebetennis" and len(value.strip()) > 10:
            return parsed.replace(tzinfo=timezone.utc), False
        return datetime.combine(parsed.date(), time.max, tzinfo=timezone.utc), True
    return parsed.astimezone(timezone.utc), False


def _parse_decimal_odds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        odds = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return odds if 1.0 < odds < 1000.0 else None


def _effective_rating(pid: str, surface: str, overall: dict[str, float],
                      surface_ratings: dict[str, dict[str, float]],
                      surface_counts: dict[str, dict[str, int]]) -> float:
    base = overall.get(pid, DEFAULT_RATING)
    count = surface_counts.get(surface, {}).get(pid, 0)
    if count < MIN_SURFACE_MATCHES:
        return base
    specific = surface_ratings.get(surface, {}).get(pid, DEFAULT_RATING)
    return base * (1.0 - SURFACE_WEIGHT) + specific * SURFACE_WEIGHT



def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return log(clipped / (1.0 - clipped))


def _fit_platt(train: list[tuple[float, bool]]) -> dict[str, float] | None:
    """Fit logistic calibration on earlier out-of-sample predictions only."""
    from math import exp
    if len(train) < 30 or len({outcome for _, outcome in train}) < 2:
        return None
    xs = [_logit(probability) for probability, _ in train]
    ys = [1.0 if outcome else 0.0 for _, outcome in train]
    slope, intercept = 1.0, 0.0
    for _ in range(100):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            z = max(min(intercept + slope * x, 30.0), -30.0)
            p = 1.0 / (1.0 + exp(-z))
            w = max(p * (1.0 - p), 1e-9)
            error = p - y
            g0 += error
            g1 += error * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        h00 += 1e-6
        h11 += 1e-6
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-12:
            return None
        d0 = (h11 * g0 - h01 * g1) / determinant
        d1 = (-h01 * g0 + h00 * g1) / determinant
        intercept -= d0
        slope -= d1
        if abs(d0) + abs(d1) < 1e-7:
            break
    if not all(x == x and abs(x) < 100 for x in (slope, intercept)):
        return None
    return {"slope": slope, "intercept": intercept}


def _apply_platt(probability: float, parameters: dict[str, float]) -> float:
    from math import exp
    z = parameters["intercept"] + parameters["slope"] * _logit(probability)
    return 1.0 / (1.0 + exp(-max(min(z, 30.0), -30.0)))


def _score_probabilities(rows: list[tuple[float, bool]]) -> dict[str, float] | None:
    if not rows:
        return None
    n = len(rows)
    eps = 1e-15
    return {
        "matches": n,
        "accuracy": sum((p >= 0.5) == y for p, y in rows) / n,
        "brier_score": sum((p - float(y)) ** 2 for p, y in rows) / n,
        "log_loss": -sum(float(y) * log(max(p, eps)) + (1.0 - float(y)) * log(max(1.0 - p, eps)) for p, y in rows) / n,
    }


def walk_forward_historical_elo(
    session: Session, *, surface: str | None = None,
    min_training_matches: int = 10, calibration_bins: int = 10,
    k_factor: float = DEFAULT_K_FACTOR, min_ev: float = 0.03,
) -> dict[str, Any]:
    """Predict timestamped historical matches before updating ratings with outcomes."""
    if surface is not None and surface.lower() not in VALID_SURFACES:
        raise ValueError("surface must be hard, clay, grass or carpet.")
    if not isinstance(min_training_matches, int) or min_training_matches < 1:
        raise ValueError("min_training_matches must be a positive integer.")
    if not isinstance(calibration_bins, int) or not 2 <= calibration_bins <= 50:
        raise ValueError("calibration_bins must be between 2 and 50.")
    if not isinstance(min_ev, (int, float)) or not 0 <= min_ev <= 1:
        raise ValueError("min_ev must be between 0 and 1.")

    statement = select(
        HistoricalMatchSnapshot.id,
        HistoricalMatchSnapshot.provider,
        HistoricalMatchSnapshot.external_id,
        HistoricalMatchSnapshot.match_date,
        HistoricalMatchSnapshot.winner_id,
        HistoricalMatchSnapshot.player_1_id,
        HistoricalMatchSnapshot.player_2_id,
        HistoricalMatchSnapshot.surface,
        HistoricalMatchSnapshot.raw,
    ).where(
        HistoricalMatchSnapshot.provider.in_(["cito", "valuebetennis"]),
        HistoricalMatchSnapshot.match_date.is_not(None),
        HistoricalMatchSnapshot.winner_id.is_not(None),
        HistoricalMatchSnapshot.player_1_id.is_not(None),
        HistoricalMatchSnapshot.player_2_id.is_not(None),
    ).order_by(
        HistoricalMatchSnapshot.match_date.asc(),
        HistoricalMatchSnapshot.id.asc(),
    )
    rows = session.execute(statement.execution_options(yield_per=5000))

    usable = []
    invalid = 0
    source_rows: dict[str, int] = defaultdict(int)
    for row in rows:
        source_rows[row.provider] += 1
        parsed, date_only = _parse_match_date(row.match_date, row.provider)
        if (
            parsed is None
            or row.winner_id not in {row.player_1_id, row.player_2_id}
            or row.player_1_id == row.player_2_id
        ):
            invalid += 1
            continue
        row_surface = (row.surface or "").strip().lower()
        if row_surface not in VALID_SURFACES:
            row_surface = "unknown"
        raw = row.raw if isinstance(row.raw, dict) else {}
        closing_1 = (
            _parse_decimal_odds(raw.get("closing_odds_player_1") or raw.get("cote1_cloture"))
            if row.provider == "valuebetennis" else None
        )
        closing_2 = (
            _parse_decimal_odds(raw.get("closing_odds_player_2") or raw.get("cote2_cloture"))
            if row.provider == "valuebetennis" else None
        )
        usable.append((
            parsed, date_only, row.id, row.provider, row.external_id,
            row.player_1_id, row.player_2_id, row.winner_id, row_surface,
            closing_1, closing_2,
        ))
    usable.sort(key=lambda item: (item[0], item[1], item[2]))

    overall: dict[str, float] = {}
    overall_counts: dict[str, int] = defaultdict(int)
    surface_ratings: dict[str, dict[str, float]] = defaultdict(dict)
    surface_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    predictions: list[tuple[float, bool]] = []
    scored = 0
    warmup_skipped = 0
    date_only_skipped = 0
    prediction_rows = []

    # Equal timestamps are predicted from the same frozen state, then updated.
    index = 0
    while index < len(usable):
        timestamp = usable[index][0]
        group = []
        while index < len(usable) and usable[index][0] == timestamp:
            group.append(usable[index])
            index += 1

        pending_updates = []
        for _, is_date_only, _, row, row_surface in group:
            p1, p2, winner = row.player_1_id, row.player_2_id, row.winner_id
            if is_date_only:
                date_only_skipped += 1
            else:
                trained = min(overall_counts[p1], overall_counts[p2])
                if trained < min_training_matches:
                    warmup_skipped += 1
                else:
                    active_surface = row_surface if row_surface in VALID_SURFACES else "hard"
                    r1 = _effective_rating(p1, active_surface, overall, surface_ratings, surface_counts)
                    r2 = _effective_rating(p2, active_surface, overall, surface_ratings, surface_counts)
                    probability = expected_probability(r1, r2)
                    outcome = winner == p1
                    predictions.append((probability, outcome))
                    scored += 1
                    prediction_rows.append({
                        "match_id": row.external_id,
                        "match_date": row.match_date,
                        "player_1_id": p1,
                        "player_2_id": p2,
                        "player_1_win_probability": round(probability, 6),
                        "outcome_player_1_won": outcome,
                        "player_1_pre_match_elo": round(r1, 2),
                        "player_2_pre_match_elo": round(r2, 2),
                        "prior_matches_player_1": overall_counts[p1],
                        "prior_matches_player_2": overall_counts[p2],
                    })
            pending_updates.append((p1, p2, winner, row_surface))

        for p1, p2, winner, row_surface in pending_updates:
            r1, r2 = overall.get(p1, DEFAULT_RATING), overall.get(p2, DEFAULT_RATING)
            n1, n2 = update_elo(r1, r2, 1.0 if winner == p1 else 0.0, k_factor)
            overall[p1], overall[p2] = n1, n2
            overall_counts[p1] += 1
            overall_counts[p2] += 1
            if row_surface in VALID_SURFACES:
                ratings = surface_ratings[row_surface]
                counts = surface_counts[row_surface]
                s1, s2 = ratings.get(p1, DEFAULT_RATING), ratings.get(p2, DEFAULT_RATING)
                sn1, sn2 = update_elo(s1, s2, 1.0 if winner == p1 else 0.0, k_factor)
                ratings[p1], ratings[p2] = sn1, sn2
                counts[p1] += 1
                counts[p2] += 1

    # Fit on the earlier 60% of out-of-sample predictions; score only the later 40%.
    split = int(len(predictions) * 0.60)
    train_predictions, test_predictions = predictions[:split], predictions[split:]
    platt = _fit_platt(train_predictions)
    calibrated = [(_apply_platt(p, platt), y) for p, y in test_predictions] if platt else []
    calibration_evaluation = {
        "method": "Platt logistic calibration",
        "chronological_split": "first 60% fit; final 40% untouched evaluation",
        "training_matches": len(train_predictions),
        "evaluation_matches": len(test_predictions),
        "parameters": (
            {"slope": round(platt["slope"], 6), "intercept": round(platt["intercept"], 6)}
            if platt else None
        ),
        "raw_elo_holdout_metrics": _score_probabilities(test_predictions),
        "calibrated_elo_holdout_metrics": _score_probabilities(calibrated),
        "status": "evaluated" if platt is not None and len(test_predictions) >= 10 else
                  "insufficient_chronological_sample_for_reliable_calibration",
    }
    if not predictions:
        metrics = None
        calibration = []
    else:
        n = len(predictions)
        brier = sum((p - float(y)) ** 2 for p, y in predictions) / n
        eps = 1e-15
        log_loss = -sum(float(y) * log(max(p, eps)) + (1.0 - float(y)) * log(max(1.0 - p, eps)) for p, y in predictions) / n
        wins = sum(y for _, y in predictions)
        metrics = {
            "matches": n,
            "accuracy": sum((p >= 0.5) == y for p, y in predictions) / n,
            "hit_rate_player_1": wins / n,
            "brier_score": brier,
            "log_loss": log_loss,
        }
        buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for p, y in predictions:
            buckets[min(int(p * calibration_bins), calibration_bins - 1)].append((p, y))
        calibration = [{
            "bin_lower": i / calibration_bins,
            "bin_upper": (i + 1) / calibration_bins,
            "count": len(items),
            "mean_predicted_probability": sum(p for p, _ in items) / len(items),
            "observed_player_1_win_rate": sum(y for _, y in items) / len(items),
        } for i, items in sorted(buckets.items())]

    return {
        "model": "expanding_window_historical_elo",
        "surface_filter": surface.lower() if surface else None,
        "metrics": metrics,
        "calibration": calibration,
        "chronological_probability_calibration": calibration_evaluation,
        "coverage": {
            "stored_rows_considered": len(rows),
            "valid_rows": len(usable),
            "out_of_sample_matches": scored,
            "skipped_warmup": warmup_skipped,
            "skipped_date_only": date_only_skipped,
            "skipped_invalid": invalid,
            "min_training_matches_per_player": min_training_matches,
        },
        "methodology": {
            "rating_initial": DEFAULT_RATING,
            "k_factor": k_factor,
            "surface_weight": SURFACE_WEIGHT,
            "minimum_surface_matches": MIN_SURFACE_MATCHES,
            "walk_forward": "expanding window; prediction before outcome update",
            "same_timestamp": "all matches at an identical timestamp are predicted before any result at that timestamp is applied",
            "date_only_policy": "date-only rows are never scored and update ratings only at end of their calendar day",
            "odds_roi": "not calculated: no verified timestamped pre-match odds join is included",
            "probability_calibration": "Platt calibration is fitted on the earlier 60% of out-of-sample Elo predictions and scored on the later 40%; this is not an Elo+live-stat ensemble",
        },
        "sample_predictions": prediction_rows[:100],
        "limitations": [
            "This evaluates a transparent Elo baseline, not a trained or calibrated betting model.",
            "Results depend on completeness and selection bias of the locally stored Cito archive.",
            "No verified historical bookmaker baseline or odds ROI; no live-stat ensemble has been validated.",
            "Platt calibration may be unstable on small archives; evaluate only on the untouched later chronological holdout.",
            "No fees, retirement adjustment, inactivity decay or doubles-specific model.",
            "Date-only rows are excluded from scoring because their exact pre-match timestamp is unknown.",
        ],
    }
