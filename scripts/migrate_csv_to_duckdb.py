"""
CSV → DuckDB Migration
======================
One-time migration of linkedin_content_bank.csv into DuckDB.
Preserves all existing data with proper types and computed fields.

Usage: python scripts/migrate_csv_to_duckdb.py
"""

import sys
import os
import hashlib
import re
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from backend.database import get_connection, get_db_path

CSV_PATH = Path(__file__).parent.parent / "linkedin_content_bank.csv"


def generate_id(url: str) -> str:
    """Deterministic ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


def compute_fingerprint(text: str) -> str:
    """Content fingerprint for dedup detection."""
    # Normalize: lowercase, strip whitespace, remove hashtags
    normalized = re.sub(r"#\w+", "", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from post text."""
    return re.findall(r"#\w+", text)


def compute_final_score(relevance: float, quality: float, trending: float) -> float:
    """Weighted score: relevance*0.3 + quality*0.4 + trending*0.1 (faithfulness defaults 0)."""
    return round(relevance * 0.3 + quality * 0.4 + trending * 0.1, 2)


def migrate():
    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Found {len(df)} rows")

    conn = get_connection()

    # Check if posts table already has data
    existing = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    if existing > 0:
        print(f"  Posts table already has {existing} rows. Skipping migration.")
        print("  To re-migrate, run: DELETE FROM posts; then re-run this script.")
        return

    migrated = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            url = str(row.get("original_link", ""))
            if not url or url == "nan":
                skipped += 1
                continue

            post_text = str(row.get("linkedin_post", ""))
            if not post_text or post_text == "nan":
                skipped += 1
                continue

            title = str(row.get("title", "Untitled"))
            source = str(row.get("source", "unknown"))
            image_url = str(row.get("image_url", "")) if pd.notna(row.get("image_url")) else None
            date_str = str(row.get("date", ""))

            relevance = float(row.get("relevance_score", 0.0)) if pd.notna(row.get("relevance_score")) else 0.0
            quality = float(row.get("quality_score", 0.0)) if pd.notna(row.get("quality_score")) else 0.0
            trending = float(row.get("trending_boost", 0.0)) if pd.notna(row.get("trending_boost")) else 0.0
            is_used = bool(row.get("used", False)) if pd.notna(row.get("used")) else False

            post_id = generate_id(url)
            char_count = len(post_text)
            hashtags = extract_hashtags(post_text)
            fingerprint = compute_fingerprint(post_text)
            final_score = compute_final_score(relevance, quality, trending)

            # Parse date
            created_at = None
            if date_str and date_str != "nan":
                try:
                    created_at = pd.to_datetime(date_str).isoformat()
                except Exception:
                    created_at = None

            conn.execute(
                """
                INSERT INTO posts (
                    id, created_at, title, original_url, source, image_url,
                    post_text, char_count, hashtags, relevance_score, quality_score,
                    trending_boost, final_score, is_used, content_fingerprint
                ) VALUES (?, ?::TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (original_url) DO NOTHING
                """,
                [
                    post_id, created_at, title, url, source, image_url,
                    post_text, char_count, hashtags, relevance, quality,
                    trending, final_score, is_used, fingerprint,
                ],
            )
            migrated += 1

        except Exception as e:
            print(f"  Error migrating row: {e}")
            skipped += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    print(f"  Total posts in DB: {count}")

    # Show source distribution
    sources = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM posts GROUP BY source ORDER BY cnt DESC"
    ).fetchall()
    print(f"\n  Source distribution:")
    for src, cnt in sources:
        print(f"    {src}: {cnt}")

    print(f"\n  Database: {get_db_path()}")


if __name__ == "__main__":
    migrate()
