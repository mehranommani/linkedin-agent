"""
Migration Runner
================
Runs schema.sql and v2 migrations in order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import get_connection, get_db_path


def _run_sql_file(conn, filepath: Path) -> None:
    """Execute a SQL file, splitting on semicolons and skipping comments."""
    if not filepath.exists():
        return
    raw = filepath.read_text()
    for chunk in raw.split(";"):
        # Strip comment-only lines, keep actual SQL
        sql_lines = [l for l in chunk.split("\n") if l.strip() and not l.strip().startswith("--")]
        stmt = "\n".join(sql_lines).strip()
        if not stmt:
            continue
        try:
            conn.execute(stmt)
        except Exception as e:
            err = str(e).lower()
            if "already exists" in err or "duplicate" in err:
                continue
            print(f"  Warning: {e}")


def _run_alter_columns(conn) -> None:
    """Run ALTER TABLE statements for v2 (idempotent)."""
    alters = [
        ("posts", "human_rating_avg", "REAL DEFAULT 0.0"),
        ("posts", "human_feedback_count", "INTEGER DEFAULT 0"),
        ("posts", "detailed_eval_id", "TEXT"),
        ("sources", "last_error", "TEXT"),
        ("sources", "last_error_at", "DATETIME"),
        ("sources", "consecutive_failures", "INTEGER DEFAULT 0"),
    ]
    for table, col, col_type in alters:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except Exception as e:
            if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"  Warning: ALTER {table}.{col}: {e}")


def run_migrations():
    """Apply schema and verify tables."""
    conn = get_connection()  # This runs schema.sql automatically

    # Run v2 schema migration
    db_dir = Path(__file__).parent
    _run_sql_file(conn, db_dir / "schema_v2.sql")
    _run_alter_columns(conn)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()

    table_names = [t[0] for t in tables]
    expected = [
        "posts", "evaluations", "pipeline_runs", "sources", "source_cache",
        "config", "trending_topics", "human_feedback", "detailed_evaluations",
    ]

    print(f"Database: {get_db_path()}")
    print(f"Tables found: {len(table_names)}")

    for t in expected:
        status = "OK" if t in table_names else "MISSING"
        print(f"  {t}: {status}")

    missing = [t for t in expected if t not in table_names]
    if missing:
        print(f"\nERROR: Missing tables: {missing}")
        sys.exit(1)

    print("\nAll migrations applied successfully.")


if __name__ == "__main__":
    run_migrations()
