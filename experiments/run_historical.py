"""Run read-only historical experiments against the configured Neon/Postgres DB."""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from experiments.historical_backtest import run_all_historical_experiments


def main() -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is required. Add your isolated Neon Preview connection "
            "string to .env; do not use the Production connection for this experiment."
        )
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]

    engine = create_engine(url, pool_pre_ping=True)
    try:
        with Session(engine) as session:
            result = run_all_historical_experiments(session)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
