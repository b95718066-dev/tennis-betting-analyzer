"""Provider-neutral data contracts for tennis matches and betting markets."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tour(StrEnum):
    ATP = "ATP"
    WTA = "WTA"
    ITF = "ITF"
    CHALLENGER = "CHALLENGER"
    OTHER = "OTHER"


class MatchStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    RETIRED = "retired"
    WALKOVER = "walkover"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"
    UNKNOWN = "unknown"


class Surface(StrEnum):
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"
    UNKNOWN = "unknown"


class SetScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_1_games: int = Field(ge=0)
    player_2_games: int = Field(ge=0)
    player_1_tiebreak: int | None = Field(default=None, ge=0)
    player_2_tiebreak: int | None = Field(default=None, ge=0)


class MatchRecord(BaseModel):
    """Canonical match shape; provider IDs are namespaced by provider."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    tour: Tour
    status: MatchStatus = MatchStatus.UNKNOWN
    scheduled_at: datetime | None = None
    tournament_id: str | None = None
    tournament_name: str | None = None
    round_name: str | None = None
    surface: Surface = Surface.UNKNOWN
    player_1_id: str | None = None
    player_1_name: str = Field(min_length=1)
    player_2_id: str | None = None
    player_2_name: str = Field(min_length=1)
    winner_id: str | None = None
    scores: list[SetScore] = Field(default_factory=list)
    best_of: int | None = Field(default=None, ge=1, le=5)
    is_doubles: bool = False
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("scheduled_at")
    @classmethod
    def require_timezone_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("scheduled_at must include a timezone (prefer UTC).")
        return value.astimezone(timezone.utc) if value is not None else None


class OddsSnapshot(BaseModel):
    """Immutable odds observation; each update is stored as a new snapshot."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    external_match_id: str = Field(min_length=1)
    bookmaker: str = Field(min_length=1)
    market: str = Field(min_length=1)
    selection: str = Field(min_length=1)
    odds_decimal: float = Field(gt=1.0, allow_inf_nan=False)
    captured_at: datetime
    is_live: bool = False
    currency: str | None = None

    @field_validator("captured_at")
    @classmethod
    def require_timezone_aware_capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must include a timezone (prefer UTC).")
        return value.astimezone(timezone.utc)
