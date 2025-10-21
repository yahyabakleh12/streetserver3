# db.py

import os

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
# ─── Database connection ───
# `DATABASE_URL` can be supplied by the environment. If missing we fall back to
# a local PostgreSQL instance suitable for development.
DEFAULT_DATABASE_URL = "postgresql+psycopg2://street:street@localhost:5432/streetserver"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

# Create engine with sensible defaults.  For SQLite we must also disable the
# thread check so the background worker can access the database.
engine_kwargs = dict(
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)


def _initialize_timescale_hypertables() -> None:
    """Ensure TimescaleDB extensions and hypertables exist when using Postgres."""

    if engine.url.get_backend_name() != "postgresql":
        return

    camera_reports_table_sql = """
    CREATE TABLE IF NOT EXISTS camera_reports (
      id BIGSERIAL PRIMARY KEY,
      ts timestamptz NOT NULL,
      camera_id text NOT NULL,
      event_type text NOT NULL,
      occupancy int,
      bbox jsonb,
      image_key text,
      payload jsonb NOT NULL
    );
    """

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        connection.execute(text(camera_reports_table_sql))
        connection.execute(
            text("SELECT create_hypertable('camera_reports','ts', if_not_exists=>true);")
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS camera_reports_camera_id_ts_idx
                ON camera_reports (camera_id, ts DESC);
                """
            )
        )


_initialize_timescale_hypertables()

Base = declarative_base()

# We set expire_on_commit=False so that after commit our objects do not expire
SessionLocal = sessionmaker(
    bind            = engine,
    autoflush       = False,
    autocommit      = False,
    expire_on_commit=False
)
