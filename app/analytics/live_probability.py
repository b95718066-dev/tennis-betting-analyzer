"""Point-state Markov pricing for in-play tennis match winner markets.

The model assumes constant point-win probabilities on each player's serve,
standard advantage games and a 7-point tiebreak (win by two). It is a
transparent baseline, not a calibrated production model.
"""
from __future__ import annotations

from functools import lru_cache
from math import isfinite
from typing import Any

from app.analytics.probability import expected_value, remove_margin, value_bet


def _validate_probability(value: float, label: str) -> None:
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{label} must be finite and strictly between 0 and 1.")


def _point_result(p1: int, p2: int, winner: int) -> tuple[int, int] | None:
    """Return updated point score, or None when the game is won."""
    if p1 >= 3 and p2 >= 3:
        if p1 == 3 and p2 == 3:
            return (4, 3) if winner == 1 else (3, 4)
        if p1 == 4 and p2 == 3:
            return None if winner == 1 else (3, 3)
        if p1 == 3 and p2 == 4:
            return (3, 3) if winner == 1 else None
        raise ValueError("Invalid advantage-game point state.")
    if winner == 1:
        return None if p1 == 3 else (p1 + 1, p2)
    return None if p2 == 3 else (p1, p2 + 1)


def _validate_point_state(p1: int, p2: int) -> None:
    if p1 not in range(5) or p2 not in range(5):
        raise ValueError("Point scores must be 0, 1, 2, 3 (40), or 4 (advantage).")
    if (p1 == 4 and p2 != 3) or (p2 == 4 and p1 != 3):
        raise ValueError("Advantage can only occur at 4-3 or 3-4.")


def _game_win_probability(
    p1: int, p2: int, server: int, p1_serve: float, p2_serve: float
) -> float:
    _validate_point_state(p1, p2)

    @lru_cache(maxsize=None)
    def solve(a: int, b: int) -> float:
        p1_point = p1_serve if server == 1 else 1.0 - p2_serve
        next_p1 = _point_result(a, b, 1)
        next_p2 = _point_result(a, b, 2)
        win_if_p1 = 1.0 if next_p1 is None else solve(*next_p1)
        win_if_p2 = 0.0 if next_p2 is None else solve(*next_p2)
        return p1_point * win_if_p1 + (1.0 - p1_point) * win_if_p2

    return solve(p1, p2)


def calculate_live_probabilities(
    *,
    sets_player_1: int,
    sets_player_2: int,
    games_player_1: int,
    games_player_2: int,
    points_player_1: int,
    points_player_2: int,
    server: int,
    best_of: int,
    player_1_serve_point_win: float,
    player_2_serve_point_win: float,
    tiebreak_points_player_1: int = 0,
    tiebreak_points_player_2: int = 0,
    tiebreak_first_server: int | None = None,
    player_1_odds: float | None = None,
    player_2_odds: float | None = None,
    min_ev: float = 0.03,
) -> dict[str, Any]:
    """Price current game, current set and match winner from a live score."""
    for label, value in (
        ("Player 1 serve point-win probability", player_1_serve_point_win),
        ("Player 2 serve point-win probability", player_2_serve_point_win),
    ):
        _validate_probability(value, label)
    _validate_point_state(points_player_1, points_player_2)
    if server not in (1, 2):
        raise ValueError("server must be 1 or 2.")
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5.")
    if min(sets_player_1, sets_player_2, games_player_1, games_player_2) < 0:
        raise ValueError("Set and game scores cannot be negative.")
    if sets_player_1 >= (best_of + 1) // 2 or sets_player_2 >= (best_of + 1) // 2:
        raise ValueError("The match is already complete.")
    if games_player_1 > 7 or games_player_2 > 7:
        raise ValueError("Game scores cannot exceed 7.")
    if games_player_1 == games_player_2 == 7:
        raise ValueError("A set cannot have a 7-7 score.")
    if min(tiebreak_points_player_1, tiebreak_points_player_2) < 0:
        raise ValueError("Tiebreak points cannot be negative.")
    in_tiebreak = games_player_1 == games_player_2 == 6
    if not in_tiebreak and (tiebreak_points_player_1 or tiebreak_points_player_2 or tiebreak_first_server is not None):
        raise ValueError("Tiebreak fields are only valid when the set score is 6-6.")
    if tiebreak_first_server not in (None, 1, 2):
        raise ValueError("tiebreak_first_server must be 1 or 2.")
    tb_first = (tiebreak_first_server or server) if in_tiebreak else 0

    needed = (best_of + 1) // 2
    if in_tiebreak:
        current_game_p1 = _calculate_tiebreak_probability(
            tiebreak_points_player_1, tiebreak_points_player_2, tb_first,
            player_1_serve_point_win, player_2_serve_point_win,
        )
    else:
        current_game_p1 = _game_win_probability(
            points_player_1, points_player_2, server,
            player_1_serve_point_win, player_2_serve_point_win,
        )

    @lru_cache(maxsize=None)
    def match_dp(
        s1: int, s2: int, g1: int, g2: int, p1: int, p2: int,
        current_server: int, tb_first_server: int, tb1: int, tb2: int,
    ) -> float:
        if s1 >= needed:
            return 1.0
        if s2 >= needed:
            return 0.0

        # A 6-6 set enters a standard 7-point tiebreak. The next scheduled
        # server is the first server in the tiebreak.
        if g1 == 6 and g2 == 6 and tb_first_server == 0:
            return match_dp(s1, s2, g1, g2, 0, 0, current_server,
                            current_server, 0, 0)

        if tb_first_server:
            played = tb1 + tb2
            if played == 0:
                tb_server = tb_first_server
            else:
                pair_index = (played - 1) // 2
                tb_server = (3 - tb_first_server) if pair_index % 2 == 0 else tb_first_server
            p1_point = player_1_serve_point_win if tb_server == 1 else 1.0 - player_2_serve_point_win
            next_tb1, next_tb2 = tb1 + 1, tb2
            p1_state = (next_tb1, next_tb2)
            p2_state = (tb1, tb2 + 1)
            p1_done = next_tb1 >= 7 and next_tb1 - next_tb2 >= 2
            p2_done = next_tb2 >= 7 and next_tb2 - next_tb1 >= 2

            def continue_tb(a: int, b: int, p1_won: bool) -> float:
                if p1_won or (a >= 7 and a - b >= 2):
                    return match_dp(s1 + 1, s2, 0, 0, 0, 0,
                                    3 - tb_first_server, 0, 0, 0)
                if b >= 7 and b - a >= 2:
                    return match_dp(s1, s2 + 1, 0, 0, 0, 0,
                                    3 - tb_first_server, 0, 0, 0)
                return match_dp(s1, s2, g1, g2, 0, 0,
                                current_server, tb_first_server, a, b)

            win_if_p1 = continue_tb(*p1_state, p1_done)
            win_if_p2 = continue_tb(*p2_state, p2_done)
            return p1_point * win_if_p1 + (1.0 - p1_point) * win_if_p2

        p1_point = player_1_serve_point_win if current_server == 1 else 1.0 - player_2_serve_point_win
        after_p1 = _point_result(p1, p2, 1)
        after_p2 = _point_result(p1, p2, 2)

        def continue_game(next_points: tuple[int, int] | None, winner: int) -> float:
            if next_points is None:
                ng1, ng2 = (g1 + 1, g2) if winner == 1 else (g1, g2 + 1)
                if (ng1 >= 6 and ng1 - ng2 >= 2) or ng1 == 7:
                    return match_dp(s1 + 1, s2, 0, 0, 0, 0,
                                    3 - current_server, 0, 0, 0)
                if (ng2 >= 6 and ng2 - ng1 >= 2) or ng2 == 7:
                    return match_dp(s1, s2 + 1, 0, 0, 0, 0,
                                    3 - current_server, 0, 0, 0)
                return match_dp(s1, s2, ng1, ng2, 0, 0,
                                3 - current_server, 0, 0, 0)
            return match_dp(s1, s2, g1, g2, next_points[0], next_points[1],
                            current_server, 0, 0, 0)

        win_if_p1 = continue_game(after_p1, 1)
        win_if_p2 = continue_game(after_p2, 2)
        return p1_point * win_if_p1 + (1.0 - p1_point) * win_if_p2

    # First compute current-set probability by pricing a best-of-one match
    # from the exact same score. Then price the full match with current sets.
    set_needed = 1
    # A local solver shares the same state transitions through a target override.
    # Reuse match_dp by translating the current sets to zero and best-of-one
    # semantics via a dedicated current-set wrapper below.
    # For set probability, calculate chance of winning the current set using
    # a second DP with the same point-state mechanics.
    current_set_p1 = _calculate_set_probability(
        games_player_1, games_player_2, points_player_1, points_player_2,
        server, player_1_serve_point_win, player_2_serve_point_win,
        tb_first if in_tiebreak else 0,
        tiebreak_points_player_1, tiebreak_points_player_2,
    )
    match_p1 = match_dp(
        sets_player_1, sets_player_2, games_player_1, games_player_2,
        points_player_1, points_player_2, server, tb_first,
        tiebreak_points_player_1, tiebreak_points_player_2,
    )

    result: dict[str, Any] = {
        "player_1_match_probability": round(match_p1, 6),
        "player_2_match_probability": round(1.0 - match_p1, 6),
        "player_1_current_game_probability": round(current_game_p1, 6),
        "player_2_current_game_probability": round(1.0 - current_game_p1, 6),
        "player_1_current_set_probability": round(current_set_p1, 6),
        "player_2_current_set_probability": round(1.0 - current_set_p1, 6),
        "model": {
            "method": "point_state_markov",
            "assumption": "constant serve point-win rates; advantage games; 7-point tiebreak, win by two",
            "calibrated": False,
        },
    }
    if (player_1_odds is None) != (player_2_odds is None):
        raise ValueError("Both player odds are required to calculate market value.")
    if player_1_odds is not None and player_2_odds is not None:
        fair_market = remove_margin({"player_1": player_1_odds, "player_2": player_2_odds})
        ev1 = expected_value(match_p1, player_1_odds)
        ev2 = expected_value(1.0 - match_p1, player_2_odds)
        result["market"] = {
            "fair_probability": fair_market,
            "overround": round((1.0 / player_1_odds + 1.0 / player_2_odds) - 1.0, 6),
            "player_1_ev": round(ev1, 6),
            "player_2_ev": round(ev2, 6),
            "player_1_value_bet": value_bet(match_p1, player_1_odds, min_ev),
            "player_2_value_bet": value_bet(1.0 - match_p1, player_2_odds, min_ev),
            "min_ev": min_ev,
        }
    return result


def _calculate_set_probability(
    g1: int, g2: int, p1: int, p2: int, server: int,
    p1_serve: float, p2_serve: float,
    initial_tb_first: int = 0, initial_tb1: int = 0, initial_tb2: int = 0,
) -> float:
    @lru_cache(maxsize=None)
    def solve(a: int, b: int, x: int, y: int, srv: int,
              tb_first: int, tb_a: int, tb_b: int) -> float:
        if a >= 6 and a - b >= 2 or a == 7:
            return 1.0
        if b >= 6 and b - a >= 2 or b == 7:
            return 0.0
        if a == 6 and b == 6 and tb_first == 0:
            return solve(a, b, 0, 0, srv, srv, 0, 0)
        if tb_first:
            n = tb_a + tb_b
            tb_srv = tb_first if n == 0 or ((n - 1) // 2) % 2 == 1 else 3 - tb_first
            p = p1_serve if tb_srv == 1 else 1.0 - p2_serve
            aa, bb = tb_a + 1, tb_b
            cc, dd = tb_a, tb_b + 1
            if aa >= 7 and aa - bb >= 2:
                wa = 1.0
            else:
                wa = solve(a, b, 0, 0, srv, tb_first, aa, bb)
            if dd >= 7 and dd - cc >= 2:
                wb = 0.0
            else:
                wb = solve(a, b, 0, 0, srv, tb_first, cc, dd)
            return p * wa + (1 - p) * wb
        p = p1_serve if srv == 1 else 1.0 - p2_serve
        nx = _point_result(x, y, 1)
        ny = _point_result(x, y, 2)
        if nx is None:
            na, nb = a + 1, b
            wa = 1.0 if (na >= 6 and na - nb >= 2) or na == 7 else solve(na, nb, 0, 0, 3 - srv, 0, 0, 0)
        else:
            wa = solve(a, b, nx[0], nx[1], srv, 0, 0, 0)
        if ny is None:
            na, nb = a, b + 1
            wb = 0.0 if (nb >= 6 and nb - na >= 2) or nb == 7 else solve(na, nb, 0, 0, 3 - srv, 0, 0, 0)
        else:
            wb = solve(a, b, ny[0], ny[1], srv, 0, 0, 0)
        return p * wa + (1 - p) * wb
    return solve(g1, g2, p1, p2, server, initial_tb_first, initial_tb1, initial_tb2)


def _calculate_tiebreak_probability(
    tb1: int, tb2: int, first_server: int, p1_serve: float, p2_serve: float,
) -> float:
    if first_server not in (1, 2):
        raise ValueError("A tiebreak requires its first server.")
    if min(tb1, tb2) < 0:
        raise ValueError("Tiebreak points cannot be negative.")

    @lru_cache(maxsize=None)
    def solve(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        played = a + b
        if played == 0:
            srv = first_server
        else:
            pair_index = (played - 1) // 2
            srv = (3 - first_server) if pair_index % 2 == 0 else first_server
        p = p1_serve if srv == 1 else 1.0 - p2_serve
        return p * solve(a + 1, b) + (1.0 - p) * solve(a, b + 1)
    return solve(tb1, tb2)
