from app.data.normalization import normalize_cito_live_match
from app.data.models import MatchStatus, Surface, Tour


def sample_live_row():
    return {
        "id": "m_2026_4891306",
        "canonical_id": "m_2026_4891306",
        "tour": "ATP",
        "status": "IN_PROGRESS",
        "scheduled_at": "2026-11-08T18:20:00+00:00",
        "started_at": "2026-11-08T18:20:00+00:00",
        "tournament_id": "atp_2026_0429",
        "tournament_name": "Stockholm",
        "tournament_display_name": "Stockholm",
        "round_name": "Round of 32",
        "surface": "Hard",
        "best_of": 3,
        "current_set": 1,
        "game_score": "15-0",
        "server": 1,
        "is_simulated": False,
        "last_updated_at": "2026-11-08T18:29:54.118590+00:00",
        "player1": {
            "id": "atp_200335",
            "canonical_id": "atp_200335",
            "name": "Felipe Meligeni",
            "current_game_score": "15",
            "is_serving": True,
        },
        "player2": {
            "id": "atp_111797",
            "canonical_id": "atp_111797",
            "name": "Nicolas Jarry",
            "current_game_score": "0",
            "is_serving": False,
        },
        "sets": [
            {
                "set_number": 1,
                "player1_games": 0,
                "player2_games": 0,
                "tiebreak_points": None,
            }
        ],
        "winner": None,
    }


def test_normalize_cito_live_row_to_canonical_match_and_live_state():
    normalized = normalize_cito_live_match(sample_live_row())

    assert normalized is not None
    assert normalized["provider"] == "cito"
    assert normalized["external_id"] == "m_2026_4891306"
    assert normalized["tour"] == Tour.ATP.value
    assert normalized["status"] == MatchStatus.LIVE.value
    assert normalized["surface"] == Surface.HARD.value
    assert normalized["player_1_name"] == "Felipe Meligeni"
    assert normalized["player_2_name"] == "Nicolas Jarry"
    assert normalized["tournament_name"] == "Stockholm"
    assert normalized["scores"][0]["player_1_games"] == 0
    assert normalized["live"]["current_set"] == 1
    assert normalized["live"]["game_score"] == "15-0"
    assert normalized["live"]["server"] == 1
    assert normalized["live"]["player_1_is_serving"] is True
    assert normalized["live"]["is_simulated"] is False


def test_normalize_cito_wta_and_tiebreak_points():
    row = sample_live_row()
    row["tour"] = "WTA"
    row["surface"] = "Clay"
    row["sets"][0]["player1_games"] = 6
    row["sets"][0]["player2_games"] = 6
    row["sets"][0]["tiebreak_points"] = {"player1": 4, "player2": 3}

    normalized = normalize_cito_live_match(row)

    assert normalized is not None
    assert normalized["tour"] == Tour.WTA.value
    assert normalized["surface"] == Surface.CLAY.value
    assert normalized["scores"][0]["player_1_tiebreak"] == 4
    assert normalized["scores"][0]["player_2_tiebreak"] == 3


def test_normalizer_does_not_invent_missing_player_names_or_ids():
    row = sample_live_row()
    row["player2"].pop("name")

    assert normalize_cito_live_match(row) is None
