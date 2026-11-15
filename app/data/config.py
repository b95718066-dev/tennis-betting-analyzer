"""Environment-based provider configuration. Secrets are never logged."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSettings:
    historical_provider: str = "none"
    live_provider: str = "none"
    odds_provider: str = "none"
    api_base_url: str | None = None
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    odds_api_key: str | None = None
    odds_api_base_url: str | None = None
    odds_api_regions: str = "eu"
    odds_api_markets: str = "h2h,spreads,totals"
    timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        timeout = float(os.getenv("TENNIS_DATA_TIMEOUT_SECONDS", "15"))
        if timeout <= 0:
            raise ValueError("TENNIS_DATA_TIMEOUT_SECONDS must be positive.")
        return cls(
            historical_provider=os.getenv("TENNIS_HISTORICAL_PROVIDER", "none").lower(),
            live_provider=os.getenv("TENNIS_LIVE_PROVIDER", "none").lower(),
            odds_provider=os.getenv("TENNIS_ODDS_PROVIDER", "none").lower(),
            api_base_url=os.getenv("TENNIS_DATA_API_BASE_URL") or None,
            api_key=os.getenv("TENNIS_DATA_API_KEY") or None,
            api_key_header=os.getenv("TENNIS_DATA_API_KEY_HEADER", "Authorization"),
            api_key_prefix=os.getenv("TENNIS_DATA_API_KEY_PREFIX", "Bearer "),
            odds_api_key=os.getenv("THE_ODDS_API_KEY") or None,
            odds_api_base_url=os.getenv("THE_ODDS_API_BASE_URL") or None,
            odds_api_regions=os.getenv("THE_ODDS_API_REGIONS", "eu"),
            odds_api_markets=os.getenv("THE_ODDS_API_MARKETS", "h2h,spreads,totals"),
            timeout_seconds=timeout,
        )


def build_live_provider(settings: ProviderSettings | None = None):
    """Construct the configured live provider; no network request is made here."""
    from app.data.providers.cito import CitoTennisProvider

    settings = settings or ProviderSettings.from_env()
    if settings.live_provider == "none":
        return None
    if settings.live_provider == "cito":
        return CitoTennisProvider(
            api_key=settings.api_key,
            base_url=settings.api_base_url,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"Unsupported live tennis provider: {settings.live_provider}")



def build_odds_provider(settings: ProviderSettings | None = None):
    """Construct the configured bookmaker-odds provider."""
    from app.data.providers.the_odds_api import TheOddsApiProvider

    settings = settings or ProviderSettings.from_env()
    if settings.odds_provider == "none":
        return None
    if settings.odds_provider == "the_odds_api":
        return TheOddsApiProvider(
            api_key=settings.odds_api_key,
            base_url=settings.odds_api_base_url,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"Unsupported odds provider: {settings.odds_provider}")
