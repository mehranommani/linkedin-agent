"""
Named Query Functions
=====================
Centralized SQL queries for the database layer.
Used by MCP database tools and API endpoints.
"""

from __future__ import annotations

from backend.database import execute, fetch_one, fetch_all, fetch_df


# --- Posts ---

def get_posts(
    page: int = 1,
    per_page: int = 20,
    source: str | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    min_score: float | None = None,
    is_used: bool | None = None,
) -> tuple[list[dict], int]:
    """Paginated post listing with filters."""
    conditions = []
    params = []

    if source:
        conditions.append("source = ?")
        params.append(source)
    if search:
        conditions.append("(title LIKE ? OR post_text LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])
    if min_score is not None:
        conditions.append("final_score >= ?")
        params.append(min_score)
    if is_used is not None:
        conditions.append("is_used = ?")
        params.append(is_used)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    allowed_sorts = {"created_at", "final_score", "relevance_score", "quality_score", "char_count"}
    sort_col = sort_by if sort_by in allowed_sorts else "created_at"

    # Count total
    total = fetch_one(f"SELECT COUNT(*) FROM posts {where}", params)[0]

    # Fetch page
    offset = (page - 1) * per_page
    rows = fetch_all(
        f"""
        SELECT id, created_at, title, original_url, source, image_url,
               post_text, char_count, hashtags, relevance_score, quality_score,
               faithfulness_score, trending_boost, final_score, is_used,
               generation_attempts, pipeline_run_id
        FROM posts {where}
        ORDER BY {sort_col} DESC
        LIMIT ? OFFSET ?
        """,
        params + [per_page, offset],
    )

    columns = [
        "id", "created_at", "title", "original_url", "source", "image_url",
        "post_text", "char_count", "hashtags", "relevance_score", "quality_score",
        "faithfulness_score", "trending_boost", "final_score", "is_used",
        "generation_attempts", "pipeline_run_id",
    ]
    posts = [dict(zip(columns, row)) for row in rows]

    return posts, total


def get_post_by_id(post_id: str) -> dict | None:
    """Get a single post by ID."""
    row = fetch_one("SELECT * FROM posts WHERE id = ?", [post_id])
    if not row:
        return None
    columns = [desc[0] for desc in execute("SELECT * FROM posts LIMIT 0").description]
    return dict(zip(columns, row))


def url_exists(url: str) -> bool:
    """Check if a URL already exists in posts."""
    result = fetch_one("SELECT 1 FROM posts WHERE original_url = ?", [url])
    return result is not None


def get_post_stats() -> dict:
    """Summary statistics for the dashboard."""
    row = fetch_one("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE is_used) as used,
            ROUND(AVG(relevance_score), 2) as avg_relevance,
            ROUND(AVG(quality_score), 2) as avg_quality,
            ROUND(AVG(final_score), 2) as avg_score,
            ROUND(AVG(char_count), 0) as avg_length
        FROM posts
    """)
    return {
        "total_posts": row[0],
        "used_posts": row[1],
        "avg_relevance": row[2],
        "avg_quality": row[3],
        "avg_score": row[4],
        "avg_length": int(row[5]) if row[5] else 0,
    }


def get_source_distribution() -> list[dict]:
    """Post count per source."""
    rows = fetch_all("""
        SELECT source, COUNT(*) as count,
               ROUND(AVG(quality_score), 2) as avg_quality,
               ROUND(AVG(final_score), 2) as avg_score
        FROM posts GROUP BY source ORDER BY count DESC
    """)
    return [{"source": r[0], "count": r[1], "avg_quality": r[2], "avg_score": r[3]} for r in rows]


# --- Sources ---

def get_active_sources() -> list[dict]:
    """Get all active sources with their configs."""
    rows = fetch_all("""
        SELECT id, source_type, name, url, category, params,
               is_active, priority, total_fetched, total_accepted,
               avg_quality, last_fetched_at
        FROM sources WHERE is_active = 1
        ORDER BY priority ASC, name ASC
    """)
    columns = [
        "id", "source_type", "name", "url", "category", "params",
        "is_active", "priority", "total_fetched", "total_accepted",
        "avg_quality", "last_fetched_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_all_sources() -> list[dict]:
    """Get all sources regardless of active status."""
    rows = fetch_all("""
        SELECT id, source_type, name, url, category, params,
               is_active, priority, total_fetched, total_accepted,
               avg_quality, last_fetched_at
        FROM sources ORDER BY source_type, priority ASC, name ASC
    """)
    columns = [
        "id", "source_type", "name", "url", "category", "params",
        "is_active", "priority", "total_fetched", "total_accepted",
        "avg_quality", "last_fetched_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


# --- Config ---

def get_config(key: str) -> any:
    """Get a config value by key."""
    row = fetch_one("SELECT value FROM config WHERE key = ?", [key])
    return row[0] if row else None


def set_config(key: str, value, description: str | None = None) -> bool:
    """Set a config value."""
    import json
    execute(
        """
        INSERT INTO config (key, value, description, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT (key) DO UPDATE SET
            value = excluded.value,
            description = COALESCE(excluded.description, config.description),
            updated_at = datetime('now')
        """,
        [key, json.dumps(value), description],
    )
    return True


# --- Pipeline Runs ---

def get_recent_issues(last_n_runs: int = 5) -> dict:
    """Aggregate common issues from recent pipeline runs."""
    rows = fetch_all(
        """
        SELECT common_issues FROM pipeline_runs
        WHERE common_issues IS NOT NULL
        ORDER BY started_at DESC LIMIT ?
        """,
        [last_n_runs],
    )

    all_issues = {}
    for (issues,) in rows:
        if issues:
            for issue in issues:
                all_issues[issue] = all_issues.get(issue, 0) + 1

    sorted_issues = sorted(all_issues.items(), key=lambda x: x[1], reverse=True)
    return {
        "common_issues": [i[0] for i in sorted_issues[:10]],
        "recommendations": [f"Address: {i[0]} (seen {i[1]}x)" for i in sorted_issues[:5]],
    }
