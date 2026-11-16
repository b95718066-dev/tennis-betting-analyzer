"""Leakage-safe expanding-window evaluation for historical Elo.

Date-only archive rows can update ratings only after their calendar day. They
are never scored out-of-sample because their true pre-match time is unknown.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timezone
from math import log
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.elo import DEFAULT_K_FACTOR, DEFAULT_RATING, expected_probability, update_elo
from app.analytics.historical_elo import MIN_SURFACE_MATCHES, SURFACE_WEIGHT, VALID_SURFACES
from app.data.normalization import normalize_surface
from app.storage.models import HistoricalMatchSnapshot


def _parse_match_date(
    value: str | None, provider: str,
) -> tuple[datetime | None, bool]:
    """Parse source timestamps; Valuebetennis documents naive CSV datetimes as UTC."""
    if not value:
        return None, False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            day = date.fromisoformat(value[:10])
        except ValueError:
            return None, False
        return datetime.combine(day, time.max, tzinfo=timezone.utc), True
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if provider == "valuebetennis" and len(value.strip()) > 10:
            return parsed.replace(tzinfo=timezone.utc), False
        return datetime.comֻǱ˞