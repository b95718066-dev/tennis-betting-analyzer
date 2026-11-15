"""Chronological backtesting and model calibration metrics.

This module evaluates already-generated, out-of-sample predictions. It does not
train a model or create historical predictions from future information.
"""

from __future__ import annotations

from datetime import datetime
from math import isfinite, log
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BacktestBet(BaseModel):
    """A prediction and executable price recorded before its result was known."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=180)
    market: str = Field(default="match_winner", min_length=1, max_length=120)
    selection: str = Field(min_length=1, max_length=180)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    odds_decimal: float = Field(gt=1.0, allow_inf_nan=False)
    outcome: bool
    stake: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    placed_at: datetime
    settled_at: datetime

    @field_validator("placed_at", "settled_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Backtest timestamps must include a timezone.")
        return value


def chronological_split(
    records: Sequence[dict],
    *,
    timestamp_field: str = "placed_at",
    train_fraction: float = 0.7,
) -> tuple[list[dict], list[dict]]:
    """Split records by time, never randomly, to avoid future-to-past leakage."""
    if not isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    ordered = sorted(records, key=lambda row: row[timestamp_field])
    if len(ordered) < 2:
        raise ValueError("At least two records are required for a chronological split.")
    split_at = int(len(ordered) * train_fraction)
    split_at = min(max(split_at, 1), len(ordered) - 1)
    return ordered[:split_at], ordered[split_at:]


def _calibration(
    observations: list[tuple[float, bool]],
    bins: int,
) -> list[dict[str, float | int]]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, outcome in observations:
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, outcome))

    result = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_predicted = sum(item[0] for item in bucket) / len(bucket)
        observed_rate = sum(1.0 if item[1] else 0.0 for item in bucket) / len(bucket)
        result.append({
            "bin_lower": index / bins,
            "bin_upper": (index + 1) / bins,
            "count": len(bucket),
            "mean_predicted_probability": mean_predicted,
            "observed_win_rate": observed_rate,
            "absolute_calibration_error": abs(mean_predicted - observed_rate),
        })
    return result


def run_backtest(
    bets: Sequence[BacktestBet | dict],
    *,
    initial_bankroll: float = 1000.0,
    calibration_bins: int = 10,
) -> dict:
    """Evaluate settled bets in placement order with flat, recorded stakes.

    ROI is net profit / initial bankroll. Yield is net profit / total stake
    actually placed. Fees, taxes, cash-out and price slippage are not modelled.
    """
    if not isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError("initial_bankroll must be finite and positive.")
    if not isinstance(calibration_bins, int) or not 2 <= calibration_bins <= 50:
        raise ValueError("calibration_bins must be an integer between 2 and 50.")

    validated = [item if isinstance(item, BacktestBet) else BacktestBet.model_validate(item) for item in bets]
    if not validated:
        raise ValueError("At least one settled bet is required.")

    for bet in validated:
        if bet.settled_at <= bet.placed_at:
            raise ValueError("Each bet must settle after it was placed.")

    ordered = sorted(validated, key=lambda item: item.placed_at)
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    total_staked = 0.0
    total_returned = 0.0
    net_profit = 0.0
    wins = 0
    losses = 0
    skipped = 0
    observations: list[tuple[float, bool]] = []
    equity_curve = [{
        "placed_at": ordered[0].placed_at.isoformat(),
        "bankroll": initial_bankroll,
        "drawdown": 0.0,
    }]

    for bet in ordered:
        if bet.stake > bankroll:
            skipped += 1
            continue

        profit = bet.stake * (bet.odds_decimal - 1.0) if bet.outcome else -bet.stake
        returned = bet.stake * bet.odds_decimal if bet.outcome else 0.0
        bankroll += profit
        net_profit += profit
        total_staked += bet.stake
        total_returned += returned
        wins += int(bet.outcome)
        losses += int(not bet.outcome)
        observations.append((bet.probability, bet.outcome))

        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = peak_bankroll - bankroll
        drawdown_pct = drawdown / peak_bankroll if peak_bankroll else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        equity_curve.append({
            "placed_at": bet.placed_at.isoformat(),
            "bankroll": bankroll,
            "drawdown": drawdown,
        })

    settled = wins + losses
    if settled == 0:
        raise ValueError("No bets could be placed with the supplied initial bankroll and stakes.")

    brier = sum((probability - float(outcome)) ** 2 for probability, outcome in observations) / settled
    epsilon = 1e-15
    log_loss = -sum(
        float(outcome) * log(max(probability, epsilon))
        + (1.0 - float(outcome)) * log(max(1.0 - probability, epsilon))
        for probability, outcome in observations
    ) / settled

    return {
        "summary": {
            "bets": settled,
            "skipped_insufficient_bankroll": skipped,
            "wins": wins,
            "losses": losses,
            "hit_rate": wins / settled,
            "initial_bankroll": initial_bankroll,
            "final_bankroll": bankroll,
            "total_staked": total_staked,
            "total_returned": total_returned,
            "net_profit": net_profit,
            "roi": net_profit / initial_bankroll,
            "yield": net_profit / total_staked if total_staked else 0.0,
            "average_odds": sum(bet.odds_decimal for bet in ordered if bet.stake <= initial_bankroll) / settled,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "brier_score": brier,
            "log_loss": log_loss,
        },
        "calibration": _calibration(observations, calibration_bins),
        "equity_curve": equity_curve,
        "methodology": {
            "ordering": "placed_at ascending",
            "prediction_timing": "placed_at must precede settled_at",
            "stake_policy": "fixed recorded stake; bets exceeding available bankroll are skipped",
            "fees_included": False,
            "slippage_included": False,
            "metrics_scope": "executed bets only",
        },
    }
