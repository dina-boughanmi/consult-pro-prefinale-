import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DATABASE_URL = os.environ.get("CONSULTPRO_DATABASE_URL", "").strip()
ENVIRONMENT = os.environ.get("CONSULTPRO_ENVIRONMENT", "development").lower()
DB_SSLMODE = os.environ.get("CONSULTPRO_DB_SSLMODE")

if ENVIRONMENT == "production" and DB_SSLMODE not in {
    None,
    "require",
    "verify-ca",
    "verify-full",
}:
    raise RuntimeError("SSL PostgreSQL doit rester activé en production.")


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    if not DATABASE_URL:
        raise RuntimeError("CONSULTPRO_DATABASE_URL doit être configurée.")
    sslmode = DB_SSLMODE or ("require" if ENVIRONMENT == "production" else "prefer")
    connection = psycopg.connect(
        DATABASE_URL,
        sslmode=sslmode,
        connect_timeout=5,
        application_name="consultpro-api",
        row_factory=dict_row,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx ON users(lower(email))",
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id)",
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL,
            duration_s DOUBLE PRECISION,
            score INTEGER,
            result_json JSONB,
            error TEXT,
            audio_path TEXT,
            progress TEXT,
            worker_heartbeat_at TIMESTAMPTZ
        )
        """,
        "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS audio_path TEXT",
        "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS progress TEXT",
        "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS worker_heartbeat_at TIMESTAMPTZ",
        """
        CREATE INDEX IF NOT EXISTS analyses_user_created_idx
        ON analyses(user_id, created_at DESC)
        """,
    )
    with connect() as connection:
        for statement in statements:
            connection.execute(statement)


def create_analysis(
    analysis_id: str,
    user_id: str,
    created_at: str,
    audio_path: str,
) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO analyses (id, user_id, created_at, status, audio_path) "
            "VALUES (%s, %s, %s::timestamptz, 'pending', %s)",
            (analysis_id, user_id, created_at, audio_path),
        )


def update_analysis_status(
    analysis_id: str,
    status: str,
    error: str | None = None,
    progress: str | None = None,
) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE analyses SET status = %s, error = %s, progress = %s, "
            "worker_heartbeat_at = CASE WHEN %s = 'processing' THEN now() ELSE NULL END, "
            "audio_path = CASE WHEN %s = 'error' THEN NULL ELSE audio_path END "
            "WHERE id = %s",
            (status, error, progress, status, status, analysis_id),
        )


def claim_next_analysis() -> dict | None:
    with connect() as connection:
        return connection.execute(
            """
            WITH next_job AS (
                SELECT id
                FROM analyses
                WHERE audio_path IS NOT NULL
                  AND (
                    status = 'pending'
                    OR (
                        status = 'processing'
                        AND worker_heartbeat_at < now() - interval '5 minutes'
                    )
                  )
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE analyses AS analysis
            SET status = 'processing',
                progress = 'En file de traitement...',
                error = NULL,
                worker_heartbeat_at = now()
            FROM next_job
            WHERE analysis.id = next_job.id
            RETURNING analysis.id, analysis.user_id, analysis.audio_path
            """
        ).fetchone()


def update_analysis_progress(analysis_id: str, progress: str) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE analyses SET progress = %s, worker_heartbeat_at = now() WHERE id = %s",
            (progress, analysis_id),
        )


def heartbeat_analysis(analysis_id: str) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE analyses SET worker_heartbeat_at = now() "
            "WHERE id = %s AND status = 'processing'",
            (analysis_id,),
        )


def save_analysis_result(
    analysis_id: str,
    result: dict,
    duration_s: float | None,
    score: int | None,
) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE analyses SET status = 'done', error = NULL, result_json = %s, "
            "duration_s = %s, score = %s, progress = NULL, audio_path = NULL, "
            "worker_heartbeat_at = NULL WHERE id = %s",
            (Jsonb(result), duration_s, score, analysis_id),
        )


def _isoformat(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


def get_analysis_for_user(analysis_id: str, user_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id, user_id, created_at, status, duration_s, score, result_json, error, progress "
            "FROM analyses WHERE id = %s AND user_id = %s",
            (analysis_id, user_id),
        ).fetchone()
    if row is None:
        return None
    analysis = dict(row)
    analysis["created_at"] = _isoformat(analysis["created_at"])
    result = analysis.pop("result_json")
    analysis["result"] = json.loads(result) if isinstance(result, str) else result
    return analysis


def list_analyses_for_user(user_id: str) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, created_at, status, duration_s, score, error "
            "FROM analyses WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
            (user_id,),
        ).fetchall()
    return [
        {**row, "created_at": _isoformat(row["created_at"])}
        for row in rows
    ]
