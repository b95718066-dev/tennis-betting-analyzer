
from typing import Any


def analyze_h2h(
    matches: list[dict[str, Any]],
    player: str,
    opponent: str,
    surface: str | None = None,
) -> dict[str, Any]:
    """Analyze head-to-head history between two players."""

    selected = [
        match for match in matches
        if {
            match.get("player"),
            match.get("opponent"),
        } == {player, opponent}
    ]

    if surface:
        selected = [
            match for match in selected
            if match.get("surface", "").lower() == surface.lower()
        ]

    wins = 0
    losses = 0

    for match in selected:
        winner = match.get("winner")

        if winner == player:
            wins += 1
        elif winner == opponent:
            losses += 1
        else:
            raise ValueError("Invalid winner in match data.")

    total = wins + losses

    return {
        "player": player,
        "opponent": opponent,
        "matches": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "surface": surface,
    }
