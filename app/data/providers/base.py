"""Provider interface shared by historical, live and odds sources."""

from typing import Any, Protocol

from app.data.models import MatchRecord, OddsSnapshot, Tour


class DataProvider(Protocol):
    name: str

    async def get_matches(
        self,
        *,
        tour: Tour,
        start_date: str,
        end_date: str,
        status: str | None = None,
    ) -> list[MatchRecord]: ...

    async def get_live_matches(self, *, tour: Tour | None = None) -> list[MatchRecord]: ...

    async def get_odds(self, *, external_match_id: str, live_only: bool = False) -> list[OddsSnapshot]: ...


class ProviderNotImplementedError(NotImplementedError):
    """Raised when a provider does not offer a requested data capability."""


JsonObject = dict[str, Any]
