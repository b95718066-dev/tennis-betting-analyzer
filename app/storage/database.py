"""Database engine and session helpers."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def database_url_from_env() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./tennis.db").strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgres://")
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if not url:
        raise ValueError("DATABASE_URL cannot be empty.")
    return url


def build_engine(url: str | None = None) -> Engine:
    database_url = url or database_url_from_env()
    options = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **options)


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        with session.begin():
            yield session
    finally:
        session.close()
