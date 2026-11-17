"""Run the existing live Markov model from a JSON file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Support direct execution from the repository root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.live_probability import calculate_live_probabilities


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiments/model_02_markov/run.py input.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = calculate_live_probabilities(**payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
