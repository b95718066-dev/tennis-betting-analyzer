import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.historical_sync import persist_historical_rows
from app.services.pre_match_forecast import generate_pre_match_forecast
from app.storage.database import create_schema


def _row(mid, when, p1, p2, winner, surface="Hard"):
    return {
        "id": mid, "date": when, "surface": surface,
        "player1": {"id": p1}, "player2": {"id": p2},
        "winner_id": winner,
    }


def test_pre_match_forecast_builds_from_history_and_excludes_future_rows():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        _row("1", "2025-01-01T10:00:00+00:00", "A", "B", "A"),
        _row("2", "2025-01-02T10:00:00+00:00", "A", "C", "A"),
        _row("3", "2025-01-03T10:00:00+00:00", "D", "B", "D"),
        _row("4", "2025-01-04T10:00:00+00:00", "A", "B", "B"),
        _row("future", "2025-01-05T15:00:00+00:00", "A", "B", "A"),
        _row("date-only-same-day", "2025-01-05", "A", "B", "A"),
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        result = generate_pre_match_forecast(
            session,
            player_1_id="A",
            player_2_id="B",
            surface="hard",
            before="2025-01-05T12:00:00+00:00",
            player_1_odds=1.8,
            player_2_odds=2.2,
        )
        assert result["forecast_type"] == "pre_match_historical"
        assert result["history_coverage"]["player_1_matches"] == 3
        assert result["history_coverage"]["player_2_matches"] == 3
        assert result["history_coverage"]["head_to_head_matches"] == 2
        assert result["history_coverage"]["stored_rows_used"] == 4
        assert result["as_of"] == "2025-01-05T12:00:00+00:00"
        assert "betting" in result
        assert sum(result["betting"]["market_probability_without_margin"].values()) == pytest.approx(1.0)
        assert result["methodology"]["calibration_status"] == "not_calibrated"
    engine.dispose()


def test_pre_match_forecast_rejects_naive_cutoff_and_one_sided_odds():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="timezone"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="hard", before="2025-01-05T12:00:00",
            )
        with pytest.raises(ValueError, match="Both player odds"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="hard", before="2025-01-05T12:00:00+00:00",
                player_1_odds=1.8,
            )
    engine.dispose()


def test_pre_match_forecast_rejects_invalid_player_pair_and_surface():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match="distinct"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="A",
                surface="hard", before="2025-01-05T12:00:00+00:00",
            )
        with pytest.raises(ValueError, match="surface"):
            generate_pre_match_forecast(
                session, player_1_id="A", player_2_id="B",
                surface="indoor", before="2025-01-05T12:00:00+00:00",
            )
    engine.dispose()


def test_historical_player_search_returns_only_matching_archived_identities():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    rows = [
        {
            "id": "one", "date": "2025-01-01T10:00:00+00:00", "surface": "Hard",
            "tour": "WTA",
            "player1": {"id": "wta_1", "name": "Elina Svitolina"},
            "player2": {"id": "wta_2", "name": "Mirra Andreeva"},
            "winner_id": "wta_1",
        },
        {
            "id": "two", "date": "2025-01-02T10:00:00+00:00", "surface": "Hard",
            "tour": "ATP",
            "player1": {"id": "atp_1", "name": "Alexander Zverev"},
            "player2": {"id": "atp_2", "name": "Jannik Sinner"},
            "winner_id": "atp_2",
        },
    ]
    with Session(engine) as session, session.begin():
        persist_historical_rows(session, rows)
        from app.services.pre_match_forecast import search_historical_players
        result = search_historical_players(session, query="svi", tour="WTA")
        assert result["count"] == 1
        assert result["players"][0]["player_id"] == "wta_1"
        assert result["players"][0]["name"] == "Elina Svitolina"
        assert result["players"][0]["tour"] == "WTA"

        # The tour selector must filter archive rows, not merely echo a label.
        atp_only = search_historical_players(session, query="svi", tour="ATP")
        assert atp_only["count"] == 0
        all_tours = search_historical_players(session, query="svi")
        assert all_tours["count"] == 1
    engine.dispose()


def test_pre_match_forecast_uses_valuebetennis_history_and_searches_flat_csv_names():
    from app.storage.models import HistoricalMatchSnapshot
    from app.services.pre_match_forecast import search_historical_players

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        session.add_all([
            HistoricalMatchSnapshot(
                provider="valuebetennis", external_id="vb-1",
                match_date="2024-01-01 12:00:00", tour="ATP", surface="hard",
                player_1_id="vb_100", player_2_id="vb_200", winner_id="vb_100",
                stats_available=False,
                raw={"player_1_name": "Carlos Alcaraz", "player_2_name": "Jannik Sinner"},
            ),
            HistoricalMatchSnapshot(
                provider="valuebetennis", external_id="vb-2",
                match_date="2024-01-02 12:00:00", tour="ATP", surface="hard",
                player_1_id="vb_100", player_2_id="vb_300", winner_id="vb_100",
                stats_available=False,
                raw={"player_1_name": "Carlos Alcaraz", "player_2_name": "Daniil Medvedev"},
            ),
            HistoricalMatchSnapshot(
                provider="valuebetennis", external_id="vb-future",
                match_date="2025-01-06 12:00:00", tour="ATP", surface="hard",
                player_1_id="vb_100", player_2_id="vb_200", winner_id="vb_200",
                stats_available=False, raw={},
            ),
        ])
        result = generate_pre_match_forecast(
            session, player_1_id="vb_100", player_2_id="vb_200",
            surface="hard", before="2025-01-05T12:00:00+00:00",
        )
        assert result["history_coverage"]["player_1_matches"] == 2
        assert result["history_coverage"]["player_2_matches"] == 1
        assert result["history_coverage"]["stored_rows_used"] == 2
        assert "Valuebetennis" in result["history_coverage"]["source"]

        players = search_historical_players(session, query="alcaraz", tour="ATP")
        assert players["count"] == 1
        assert players["players"][0]["player_id"] == "vb_100"
        assert players["players"][0]["matches_in_archive"] == 2
        assert players["players"][0]["provider"] == "valuebetennis"
    engine.dispose()



def test_pre_match_forecast_uses_valuebetennis_utc_time_within_cutoff_day():
    from app.storage.models import HistoricalMatchSnapshot

    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    with Session(engine) as session, session.begin():
        session.add_all([
            HistoricalMatchSnapshot(
                provider="valuebetennis", external_id="vb-early",
                match_date="2025-01-05 10:00:00", tour="ATP", surface="dur",
                player_1_id="vb_a", player_2_id="vb_b", winner_id="vb_a",
                stats_available=False, raw={},
            ),
            HistoricalMatchSnapshot(
                provider="valuebetennis", external_id="vb-late",
                match_date="2025-01-05 13:00:00", tour="ATP", surface="dur",
                player_1_id="vb_a", player_2_id="vb_c", winner_id="vb_c",
                stats_available=False, raw={},
            ),
        ])
        result = generate_pre_match_forecast(
            session, player_1_id="vb_a", player_2_id="vb_b",
            surface="hard", before="2025-01-05T12:00:00Z",
        )
        assert result["history_coverage"]["player_1_matches"] == 1
        assert result["history_coverage"]["player_2_matches"] == 1
        assert result["history_coverage"]["head_to_head_matches"] == 1
        assert result["history_coverage"]["stored_rows_used"] == 1
        assert result["historical_elo"]["player"]["overall_matches"] == 1
        assert result["historical_elo"]["opponent"]["overall_matches"] == 1
    engine.dispose()
