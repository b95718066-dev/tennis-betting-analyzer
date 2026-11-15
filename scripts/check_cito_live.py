"""Safe live connectivity check for the configured Cito provider."""

import asyncio
import os

from app.data.providers.cito import CitoTennisProvider


async def main() -> None:
    key = os.getenv("TENNIS_DATA_API_KEY")
    if not key:
        raise SystemExit("TENNIS_DATA_API_KEY is not configured.")
    provider = CitoTennisProvider(api_key=key)
    try:
        rows = await provider.fetch_live_payload()
    except Exception as exc:
        # Do not print request headers, URLs with query secrets, or response bodies.
        raise SystemExit(f"Cito connectivity check failed: {type(exc).__name__}") from None
    print(f"Cito connection successful. Live rows returned: {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
