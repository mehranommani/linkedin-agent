"""
DuckDB → SQLite Data Migration
================================
Run once after switching from DuckDB to SQLite.
Reads data from the old .duckdb file and imports it into the new .db file.

Usage:
    python scripts/migrate_duckdb_to_sqlite.py

The old DuckDB file is left in place (renamed to .duckdb.migrated).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
OLD_DB = DATA_DIR / "linkedin_agent.duckdb"
NEW_DB = DATA_DIR / "linkedin_agent.db"


def _serialize(value):
    """Ensure lists/dicts are stored as JSON strings."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def migrate():
    if not OLD_DB.exists():
        print(f"Old DuckDB file not found: {OLD_DB}")
        print("Nothing to migrate.")
        return

    try:
        import duckdb
    except ImportError:
        print("duckdb package not installed — cannot read old database.")
        sys.exit(1)

    import sqlite3

    print(f"Reading from: {OLD_DB}")
    print(f"Writing to:   {NEW_DB}")

    duck = duckdb.connect(str(OLD_DB), read_only=True)

    # Bootstrap the SQLite schema via backend
    os.environ["DATABASE_PATH"] = str(NEW_DB)
    from backend.database import get_connection, close
    sqlite = get_connection()

    tables = [
        "config", "sources", "posts", "evaluations", "pipeline_runs",
        "source_cache", "trending_topics", "human_feedback", "detailed_evaluations",
    ]

    total = 0
    for table in tables:
        try:
            rows = duck.execute(f"SELECT * FROM {table}").fetchall()
            cols = [d[0] for d in duck.description]
        except Exception as e:
            print(f"  {table}: skipped ({e})")
            continue

        if not rows:
            print(f"  {table}: empty")
            continue

        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        insert_sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"

        inserted = 0
        for row in rows:
            serialized = tuple(_serialize(v) for v in row)
            try:
                sqlite.execute(insert_sql, serialized)
                inserted += 1
            except Exception as e:
                print(f"    Row error in {table}: {e}")

        print(f"  {table}: {inserted}/{len(rows)} rows migrated")
        total += inserted

    duck.close()
    close()

    # Rename old file so it won't conflict
    migrated_path = OLD_DB.with_suffix(".duckdb.migrated")
    OLD_DB.rename(migrated_path)
    print(f"\nTotal rows migrated: {total}")
    print(f"Old DuckDB renamed to: {migrated_path.name}")
    print("Migration complete. Start the backend normally.")


if __name__ == "__main__":
    migrate()
