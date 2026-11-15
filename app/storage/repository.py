"""Idempotent persistence operations for canonical data contracts."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models import MatchRecord, OddsSnapshot
from .models import OddsObservation, ProviderMatchLink, ProviderSyncState, TennisMatch


def save_match(session: Session, record: MatchRecord) -> TennisMatch:
    existing = session.scalar(
        select(TennisMatch).where(
            TennisMatch.provider == record.provider,
            TennisMatch.external_id == record.external_id,
        )
    )
    values = {
        "tour": record.tour.value,
        "status": record.status.value,
        "scheduled_at": record.scheduled_at,
        "tournament_id": record.tournament_id,
        "tournament_name": record.tournament_name,
        "round_name": record.round_name,
        "surface": record.surface.value,
        "player_1_id": record.player_1_id,
        "player_1_name": record.player_1_name,
        "player_2_id": record.player_2_id,
        "player_2_name": record.player_2_name,
        "winner_id": record.winner_id,
        "scores": [score.model_dump(mode="json") for score in record.scores],
        "best_of": record.best_of,
        "is_doubles": record.is_doubles,
        "raw": record.raw,
    }
    if existing is None:
        existing = TennisMatch(provider=record.provider, external_id=record.external_id, **values)
        session.add(existing)
    else:
        for key, value in values.items():
            setattr(existing, key, value)
    session.flush()
    return existing


def save_odds_snapshot(session: Session, record: OddsSnapshot) -> tuple[OddsObservation, bool]:
    existing = session.scalar(
        select(OddsObservation).where(
            OddsObservation.provider == record.provider,
            OddsObservation.external_match_id == record.external_match_id,
            OddsObservation.bookmaker == record.bookmaker,
            OddsObservation.market == record.market,
            OddsObservation.selection == record.selection,
            OddsObservation.captured_at == record.captured_at,
        )
    )
    if existing is not None:
        return existing, False
    observation = OddsObservation(
        provider=record.provider,
        external_match_id=record.external_match_id,
        bookmaker=record.bookmaker,
        market=record.market,
        selection=record.selection,
        odds_decimal=record.odds_decimal,
        captured_at=record.captured_at,
        is_live=record.is_live,
        currency=record.currency,
    )
    session.add(observation)
    session.flush()
    return observation, True


def update_sync_state(
    session: Session,
    *,
    provider: str,
    resource: str,
    scope: str = "all",
    cursor: str | None = None,
    last_success_at=None,
    last_error: str | None = None,
) -> ProviderSyncState:
    state = session.scalar(
        select(ProviderSyncState).where(
            ProviderSyncState.provider == provider,
            ProviderSyncState.resource == resource,
            ProviderSyncState.scope == scope,
        )
    )
    if state is None:
        state = ProviderSyncState(provider=provider, resource=resource, scope=scope)
        session.add(state)
    state.cursor = cursor
    state.last_success_at = last_success_at
    state.last_error = last_error[:500] if last_error else None
    session.flush()
    return state



def save_provider_match_link(
    session: Session,
    *,
    source_provider: str,
    source_external_id: str,
    canonical_provider: str,
    canonical_external_id: str,
    match_method: str,
    confidence: float = 1.0,
) -> ProviderMatchLink:
    """Create or refresh a deterministic cross-provider identity link."""
    link = session.scalar(
        select(ProviderMatchLink).where(
            ProviderMatchLink.source_provider == source_provider,
            ProviderMatchLink.source_external_id == source_external_id,
        )
    )
    if link is None:
        link = ProviderMatchLink(
            source_provider=source_provider,
            source_external_id=source_external_id,
            canonical_provider=canonical_provider,
            canonical_external_id=canonical_external_id,
            match_method=match_method,
            confidence=confidence,
        )
        session.add(link)
    else:
        link.canonical_provider = canonical_provider
        link.canonical_external_id = canonical_external_id
        link.match_method = match_method
        link.confidence = confidence
    session.flush()
    return link
