"""Provider adapters."""

from .base import DataProvider
from .http import HttpJsonProvider

__all__ = ["DataProvider", "HttpJsonProvider"]
