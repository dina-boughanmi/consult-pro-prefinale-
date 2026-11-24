import json
import sqlite3
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = ROOT_DIR / "backend" / "data" / "consultpro.sqlite3"
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.services.database import DATABASE_URL, DB_SSLMODE, ENVIRONMENT, initialize_database


def main() -> None:
    if not DATABASE_URL:
        raise SystemExit("CONSULTPRO_DATABASE_URL doit pointer vers PostgreSQL.")
    if not DEFAULT_SQLITE_PATH.is_file():
        raise SystemExit(f"Base SQLite introuvable : {DEFAULT_SQLITE_PATH}")

    initialize_database()
    source = sqlite3.connect(DEFAULT_SQLITE_PATH)
    source.row_factory = sqlite3.Row
    sslmode = DB_SSLMODE or ("require" if ENVIRONMENT == "production" else "prefer")
    try:
        with psycopg.connect(
            DATABASE_URL,
            sslmode=sslmode,
            connect_timeout=5,
            application_name="consultpro-data-migration",
            row_factory=dict_row,
        ) as target:
            with target.cursor() as cursor:
                users = source.execute("SELECT * FROM users").fetchall()
                cursor.executemany(
                    """
                    INSERT INTO users (id, name, email, password_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s::timestamptz)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    [
                        (row["id"], row["name"], row["email"], row["password_hash"], row["created_at"])
                        for row in users
                    ],
                )

                sessions = source.execute("SELECT * FROM sessions").fetchall()
                cursor.executemany(
                    """
                    INSERT INTO sessions (token_hash, user_id, expires_at)
                    VALUES (%s, %s, %s::timestamptz)
                    ON CONFLICT (token_hash) DO NOTHING
                    """,
                    [
                        (row["token_hash"], row["user_id"], row["expires_at"])
                        for row in sessions
                    ],
                )

                analyses = source.execute("SELECT * FROM analyses").fetchall()
                cursor.executemany(
                    """
                    INSERT INTO analyses
                        (id, user_id, created_at, status, duration_s, score, result_json, error)
                    VALUES (%s, %s, %s::timestamptz, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    [
                        (
                            row["id"],
                            row["user_id"],
                            row["created_at"],
                            row["status"],
                            row["duration_s"],
                            row["score"],
                            Jsonb(json.loads(row["result_json"]))
                            if row["result_json"] is not None
                            else None,
                            row["error"],
                        )
                        for row in analyses
                    ],
                )
            target.commit()
        print(
            "Migration terminée : "
            f"{len(users)} comptes, {len(sessions)} sessions, {len(analyses)} analyses."
        )
    finally:
        source.close()


if __name__ == "__main__":
    main()
