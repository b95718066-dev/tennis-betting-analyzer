"""CC BY 4.0 Valuebetennis historical results/odds CSV ingestion."""
from __future__ import annotations

import csv
from io import StringIO
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.storage.models import HistoricalMatchSnapshot


REQUIRED_COLUMNS = {
    "match_id", "date", "genre", "surface",
    "joueur1", "joueur1_id", "joueur2", "joueur2_id", "vainqueur_id",
}


def _prefixed_player_id(value: str | None) -> str | None:
    value = (value or "").strip()
    return f"vb_{value}" if value else None


def import_valuebetennis_csv_chunk(
    session: Session, *, csv_text: str, year: int, offset: int, limit: int,
) -> dict[str, Any]:
    """Import a deterministic row slice from a Valuebetennis season CSV."""
    if year < 2021 or year > 2026:
        raise ValueError("year must be between 2021 and 2026.")
    if offset < 0 or limit < 1 or limit > 20000:
        raise ValueError("offset must be non-negative and limit must be 1..20000.")

    reader = csv.DictReader(StringIO(csv_text), delimiter=";")
    columns = set(reader.fieldnames or [])
    if not REQUIRED_COLUMNS.issubset(columns):
        raise ValueError("Valuebetennis CSV schema does not match the documented columns.")

    selected: list[dict[str, Any]] = []
    total_rows = 0
    for row in reader:
        if total_rows >= offset and len(selected) < limit:
            match_id = (row.get("match_id") or "").strip()
            p1 = _prefixed_player_id(row.get("joueur1_id"))
            p2 = _prefixed_player_id(row.get("joueur2_id"))
            winner = _prefixed_player_id(row.get("vainqueur_id"))
            match_date = (row.get("date") or "").strip()
            tour = (row.get("genre") or "").strip().upper()
            if (
                match_id and p1 and p2 and p1 != p2
                and winner in {p1, p2}
                and match_date and tour in {"ATP", "WTA"}
            ):
                raw = dict(row)
                raw.update({
                    "source": "Valuebetennis",
                    "source_license": "CC BY 4.0",
                    "player_1_name": row.get("joueur1"),
                    "player_2_name": row.get("joueur2"),
                    "player_1_source_id": row.get("joueur1_id"),
                    "player_2_source_id": row.get("joueur2_id"),
                    "winner_source_id": row.get("vainqueur_id"),
                    "opening_odds_player_1": row.get("cote1_ouverture"),
                    "opening_odds_player_2": row.get("cote2_ouverture"),
                    "closing_odds_player_1": row.get("cote1_cloture"),
                    "closing_odds_player_2": row.get("cote2_cloture"),
                    "tournament": row.get("tournoi"),
                    "tournament_category": row.get("categorie"),
                    "round": row.get("tour"),
                    "score": row.get("score"),
                    "duration_minutes": row.get("duree_min"),
                })
                selected.append({
                    "provider": "valuebetennis",
                    "external_id": match_id,
                    "match_date": match_date,
                    "tour": tour,
                    "surface": (row.get("surface") or "").strip().lower() or None,
                    "player_1_id": p1,
                    "player_2_id": p2,
                    "winner_id": winner,
                    "stats_available": False,
                    "stats_unavailable_code": "source_has_no_match_level_serve_stats",
                    "raw": raw,
                })
        total_rows += 1

    if selected:
        statement = pg_insert(HistoricalMatchSnapshot).on_conflict_do_nothing(
            index_elements=["provider", "external_id"]
        )
        session.execute(statement, selected)

    return {
        "source": "Valuebetennis",
        "license": "CC BY 4.0",
        "year": year,
        "offset": offset,
        "requested_limit": limit,
        "total_rows_in_file": total_rows,
        "selected_valid_rows": len(selected),
        "has_next": offset + limit < total_rows,
    }
