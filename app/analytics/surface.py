
from typing import Any


SURFACES = {"hard", "clay", "grass", "carpet"}


def analyze_surface(
    matches: list[dict[str, Any]],
    surface: str,
) -> dict[str, Any]:
    """Analyze player performance on a specific surface."""

    surface = surface.lower()

    if surface not in SURFACES:
        raise ValueError(f"Unsupported surface: {surface}")

    selected = [
        match for match in matches
        if match.get("surface", "").lower() == surface
    ]

    if not selected:
        return {
            "surface": surface,
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
        }

    results = [
        match.get("result", "").upper()
        for match in selected
    ]

    if any(result not in {"W", "L"} for result in results):
        raise ValueError("Results must be W or L.")

    wins = results.count("W")
    losses = results.count("L")

    return {
        "surface": surface,
        "matches": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(results), 4),
    }
