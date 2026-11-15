"""SQLAlchemy persistence models for matches, odds and sync cursors."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, Index, Integer, JSON, String, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TennisMatch(Base):
    __tablename__ = "tennis_matches"
    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_match_provider_external_id"),
        Index("ix_matches_tour_status_start", "tour", "status", "scheduled_at"),
        Index("ix_matches_player_1", "player_1_id"),
        Index("ix_matches_player_2", "player_2_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    external_id: Mapped[str] = mapped_column(String(180), nullable=False)
    tour: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tournament_id: Mapped[str | None] = mapped_column(String(180))
    tournament_name: Mapped[str | None] = mapped_column(String(240))
    round_name: Mapped[str | None] = mapped_column(String(80))
    surface: Mapped[str] = mapped_column(String(24), nullable=False)
    player_1_id: Mapped[str | None] = mapped_column(String(180))
    player_1_name: Mapped[str] = mapped_column(String(240), nullable=False)
    player_2_id: Mapped[str | None] = mapped_column(String(180))
    player_2_name: Mapped[str] = mapped_column(String(240), nullable=False)
    winner_id: Mapped[str | None] = mapped_column(String(180))
    scores: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    best_of: Mapped[int | None] = mapped_column(Integer)
    is_doubles: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class OddsObservation(Base):
    __tablename__ = "odds_observations"
    __table_args__ = (
        UniqueConstraint(
            "provider", "external_match_id", "bookmaker", "market", "selection", "captured_at",
            name="uq_odds_observation_identity",
        ),
        Index("ix_odds_match_market_time", "external_match_id", "market", "captured_at"),
        Index("ix_odds_live_capture", "is_live", "captured_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    external_match_id: Mapped[str] = mapped_column(String(180), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(120), nullable=False)
    market: Mapped[str] = mapped_column(String(120), nullable=False)
    selection: Mapped[str] = mapped_column(String(180), nullable=False)
    odds_decimal: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_live: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    currency: Mapped[str | None] = mapped_column(String(8))


class ProviderSyncState(Base):
    __tablename__ = "provider_sync_state"
    __table_args__ = (
        UniqueConstraint("provider", "resource", "scope", name="uq_provider_sync_scope"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    resource: Mapped[str] = mapped_column(String(80), nullable=False)
    scope: Mapped[str] = mapped_column(String(120), nullable=False, default="all")
    cursor: Mapped[str | None] = mapped_column(String(500))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(500))


class ProviderMatchLink(Base):
    """Explicit cross-provider identity link; never infer identity from provider IDs."""

    __tablename__ = "provider_match_links"
    __table_args__ = (
        UniqueConstraint("source_provider", "source_external_id", name="uq_source_match_link"),
        Index("ix_canonical_match_link", "canonical_provider", "canonical_external_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    source_external_id: Mapped[str] = mapped_column(String(180), nullable=False)
    canonical_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    canonical_external_id: Mapped[str] = mapped_column(String(180), nullable=False)
    match_method: Mapped[str] = mapped_column(String(80), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    matched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
