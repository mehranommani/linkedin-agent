"""
SQLite Connection Manager
=========================
Singleton connection with WAL mode, schema initialization, and query helpers.
Replaces DuckDB — SQLite avoids the mmap-flush zombie-process issue.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

_DB_PATH = os.environ.get(
    "DATABASE_PATH",
    str(Path(__file__).parent.parent / "data" / "linkedin_agent.db"),
)
_SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"

import threading
_local = threading.local()
_schema_initialized = False


def get_connection() -> sqlite3.Connection:
    """Get or create a per-thread SQLite connection (thread-safe for FastAPI threadpool)."""
    global _schema_initialized
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=True)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")  # 10s wait on lock
        conn.isolation_level = None  # autocommit
        _local.conn = conn
        if not _schema_initialized:
            _init_schema(conn)
            _schema_initialized = True
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Run schema.sql to ensure all tables exist."""
    schema_sql = _SCHEMA_PATH.read_text()
    for statement in schema_sql.split(";"):
        stmt = "\n".join(
            line for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ).strip()
        if stmt:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e).lower():
                    raise


def execute(sql: str, params: list | None = None) -> sqlite3.Cursor:
    """Execute a SQL statement and return the cursor."""
    conn = get_connection()
    # Serialize any list/dict params to JSON strings
    if params:
        params = [json.dumps(p) if isinstance(p, (list, dict)) else p for p in params]
        return conn.execute(sql, params)
    return conn.execute(sql)


def fetch_one(sql: str, params: list | None = None) -> tuple | None:
    """Execute and fetch a single row."""
    return execute(sql, params).fetchone()


def fetch_all(sql: str, params: list | None = None) -> list[tuple]:
    """Execute and fetch all rows."""
    return execute(sql, params).fetchall()


def fetch_df(sql: str, params: list | None = None):
    """Execute and return as a pandas DataFrame."""
    import pandas as pd
    rows = fetch_all(sql, params)
    if not rows:
        return pd.DataFrame()
    cursor = execute(sql, params)
    cols = [d[0] for d in cursor.description] if cursor.description else []
    return pd.DataFrame(rows, columns=cols)


def close() -> None:
    """Close the current thread's database connection."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


def get_db_path() -> str:
    """Return the current database file path."""
    return _DB_PATH
