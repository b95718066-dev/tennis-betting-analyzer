
from typing import Any


def calculate_form(
    matches: list[dict[str, Any]],
    last_n: int = 10,
    surface: str | None = None,
) -> dict[str, Any]:
    """Calculate recent tennis performance."""

    if last_n < 1:
        raise ValueError("last_n must be positive.")

    selected = matches[-last_n:]

    if surface is not None:
        selected = [
            match for match in selected
            if match.get("surface", "").lower() == surface.lower()
        ]

    if not selected:
        return {
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "form_score": 0.0,
            "streak": 0,
        }

    results = []

    for match in selected:
        result = match.get("result", "").upper()

        if result not in ("W", "L"):
            raise ValueError("Result must be W or L.")

        results.append(result)

    wins = results.count("W")
    losses = results.count("L")

    weighted_score = 0.0
    total_weight = 0.0

    for index, result in enumerate(results):
        weight = index + 1
        weighted_score += weight * (1.0 if result == "W" else 0.0)
        total_weight += weight

    streak = 0
    latest_result = results[-1]

    for result in reversed(results):
        if result != latest_result:
            break
        streak += 1

    return {
        "matches": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(results),
        "form_score": weighted_score / total_weight,
        "streak": streak if latest_result == "W" else -streak,
    }
