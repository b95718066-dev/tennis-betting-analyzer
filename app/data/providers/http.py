"""Small async HTTP JSON transport used by concrete provider adapters."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx


class ProviderHttpError(RuntimeError):
    """Sanitized provider request error; credentials are never included."""

    def __init__(
        self,
        provider: str,
        status_code: int | None,
        message: str,
        error_code: str | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"{provider}: {message}")


class HttpJsonProvider:
    """Reusable transport; endpoint paths must be relative to the configured base URL."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Provider base_url must be an absolute HTTPS URL.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.name = name
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self._headers = {"Accept": "application/json"}
        if api_key:
            self._headers[api_key_header] = f"{api_key_prefix}{api_key}"
        self._client = client

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        if not path or path.startswith(("http://", "https://", "//")):
            raise ValueError("path must be a non-empty relative API path.")
        if ".." in path.split("/"):
            raise ValueError("path cannot traverse outside the provider base URL.")

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await client.get(
                self.base_url + path.lstrip("/"),
                params=params,
                headers=self._headers,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, (dict, list)):
                raise ProviderHttpError(self.name, response.status_code, "unexpected JSON response shape")
            return payload
        except httpx.HTTPStatusError as exc:
            # Preserve only the provider's machine-readable error code. Never log
            # response bodies, request headers, or credentials.
            error_code = None
            try:
                body = exc.response.json()
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict) and isinstance(error.get("code"), str):
                    error_code = error["code"][:100]
                elif isinstance(body, dict) and isinstance(body.get("code"), str):
                    error_code = body["code"][:100]
            except ValueError:
                pass
            raise ProviderHttpError(
                self.name,
                exc.response.status_code,
                "upstream returned an HTTP error",
                error_code=error_code,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderHttpError(self.name, None, "request timed out") from exc
        except httpx.RequestError as exc:
            raise ProviderHttpError(self.name, None, "request failed") from exc
        except ValueError as exc:
            if isinstance(exc, ProviderHttpError):
                raise
            raise ProviderHttpError(self.name, None, "upstream returned invalid JSON") from exc
        finally:
            if owns_client:
                await client.aclose()
