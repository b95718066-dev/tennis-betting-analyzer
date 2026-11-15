"""Safe live connectivity check for the configured Cito provider."""

import asyncio
import os

from app.data.providers.cito import CitoTennisProvider
from app.data.providers.http import ProviderHttpError


async def main() -> None:
    key = os.getenv("TENNIS_DATA_API_KEY")
    if not key:
        raise SystemExit("TENNIS_DATA_API_KEY is not configured.")
    provider = CitoTennisProvider(api_key=key)
    try:
        rows = await provider.fetch_live_payload()
    except ProviderHttpError as exc:
        # Status code and sanitized message are safe; never print credentials or body.
        code = f"; provider_code={exc.error_code}" if exc.error_code else ""
        hint = (
            "; verify the account email in the Cito dashboard"
            if exc.error_code == "EMAIL_NOT_VERIFIED"
            else ""
        )
        raise SystemExit(
            f"Cito API request failed: HTTP {exc.status_code or 'network-error'}{code}{hint}; {exc}"
        ) from None
    except Exception as exc:
        raise SystemExit(f"Cito connectivity check failed: {type(exc).__name__}") from None
    print(f"Cito connection successful. Live rows returned: {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
