"""
MCP Database Tools
==================
Tools for DuckDB access: queries, inserts, config, dedup checks.
"""
from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from typing import Any

from backend.mcp.server import mcp_server
from backend.database import execute, fetch_one, fetch_all
from backend.config import ConfigManager


@mcp_server.tool()
def query_posts(
    page: int = 1,
    per_page: int = 20,
    source: str | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    min_score: float | None = None,
    is_used: bool | None = None,
) -> dict:
    """Query posts with pagination and filters."""
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

    total = fetch_one(f"SELECT COUNT(*) FROM posts {where}", params)[0]
    offset = (page - 1) * per_page

    rows = fetch_all(
        f"""
        SELECT id, created_at, title, original_url, source, image_url,
               post_text, char_count, hashtags, relevance_score, quality_score,
               faithfulness_score, trending_boost, final_score, is_used,
               generation_attempts, pipeline_run_id,
               human_rating_avg, human_feedback_count, detailed_eval_id
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
        "human_rating_avg", "human_feedback_count", "detailed_eval_id",
    ]

    import json as _json

    def _jlist(v):
        if v is None or isinstance(v, list):
            return v or []
        try:
            r = _json.loads(v)
            return r if isinstance(r, list) else []
        except Exception:
            return []

    posts = []
    for row in rows:
        post = dict(zip(columns, row))
        if post["created_at"]:
            post["created_at"] = str(post["created_at"])
        post["hashtags"] = _jlist(post.get("hashtags"))
        posts.append(post)

    return {"posts": posts, "total": total, "page": page, "per_page": per_page}


@mcp_server.tool()
def insert_post(
    title: str,
    original_url: str,
    source: str,
    post_text: str,
    relevance_score: float,
    quality_score: float,
    faithfulness_score: float = 0.0,
    trending_boost: float = 0.0,
    source_summary: str | None = None,
    image_url: str | None = None,
    extracted_content: str | None = None,
    hashtags: list[str] | None = None,
    generation_attempts: int = 1,
    pipeline_run_id: str | None = None,
) -> dict:
    """Insert a new post into the database."""
    post_id = hashlib.md5(original_url.encode()).hexdigest()[:16]
    char_count = len(post_text)
    final_score = round(
        relevance_score * 0.3 + quality_score * 0.4 +
        faithfulness_score * 0.2 + trending_boost * 0.1, 2
    )

    # Content fingerprint for dedup
    normalized = re.sub(r"#\w+", "", post_text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    fingerprint = hashlib.sha256(normalized.encode()).hexdigest()[:32]

    if hashtags is None:
        hashtags = re.findall(r"#\w+", post_text)

    execute(
        """
        INSERT INTO posts (
            id, title, original_url, source, source_summary, image_url,
            extracted_content, post_text, char_count, hashtags,
            relevance_score, quality_score, faithfulness_score, trending_boost,
            final_score, content_fingerprint, generation_attempts, pipeline_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (original_url) DO NOTHING
        """,
        [
            post_id, title, original_url, source, source_summary, image_url,
            extracted_content, post_text, char_count, hashtags,
            relevance_score, quality_score, faithfulness_score, trending_boost,
            final_score, fingerprint, generation_attempts, pipeline_run_id,
        ],
    )
    return {"id": post_id, "success": True}


@mcp_server.tool()
def get_config(key: str) -> dict:
    """Get a configuration value by key."""
    value = ConfigManager.get(key)
    return {"key": key, "value": value}


@mcp_server.tool()
def set_config(key: str, value: Any, description: str | None = None) -> dict:
    """Set a configuration value."""
    ConfigManager.set(key, value, description)
    return {"success": True}


@mcp_server.tool()
def get_active_sources() -> dict:
    """Get all active content sources with their configs."""
    rows = fetch_all("""
        SELECT id, source_type, name, url, category, params,
               priority, total_fetched, total_accepted, avg_quality
        FROM sources WHERE is_active = 1
        ORDER BY priority ASC, name ASC
    """)
    columns = [
        "id", "source_type", "name", "url", "category", "params",
        "priority", "total_fetched", "total_accepted", "avg_quality",
    ]
    sources = []
    for row in rows:
        src = dict(zip(columns, row))
        if isinstance(src["params"], str):
            try:
                src["params"] = json.loads(src["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        sources.append(src)
    return {"sources": sources}


@mcp_server.tool()
def check_url_exists(url: str) -> dict:
    """Check if a URL already exists in the posts table."""
    result = fetch_one("SELECT 1 FROM posts WHERE original_url = ?", [url])
    return {"exists": result is not None}


@mcp_server.tool()
def get_source_stats() -> dict:
    """Get statistics per source type."""
    rows = fetch_all("""
        SELECT source, COUNT(*) as total,
               COUNT(*) FILTER (WHERE quality_score >= 7.0) as accepted,
               ROUND(AVG(quality_score), 2) as avg_quality,
               ROUND(AVG(final_score), 2) as avg_score
        FROM posts GROUP BY source ORDER BY total DESC
    """)
    stats = []
    for row in rows:
        stats.append({
            "source": row[0], "total": row[1], "accepted": row[2],
            "avg_quality": row[3], "avg_score": row[4],
        })
    return {"stats": stats}


@mcp_server.tool()
def log_pipeline_run(
    run_id: str,
    status: str,
    items_fetched: int = 0,
    items_filtered: int = 0,
    posts_accepted: int = 0,
    posts_rejected: int = 0,
    avg_quality: float | None = None,
    avg_relevance: float | None = None,
    pass_rate: float | None = None,
    common_issues: list[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Log or update a pipeline run record."""
    existing = fetch_one("SELECT 1 FROM pipeline_runs WHERE id = ?", [run_id])

    if existing:
        execute(
            """
            UPDATE pipeline_runs SET
                status = ?, completed_at = current_timestamp,
                items_fetched = ?, items_filtered = ?,
                posts_accepted = ?, posts_rejected = ?,
                avg_quality = ?, avg_relevance = ?,
                pass_rate = ?, common_issues = ?
            WHERE id = ?
            """,
            [
                status, items_fetched, items_filtered,
                posts_accepted, posts_rejected,
                avg_quality, avg_relevance, pass_rate,
                common_issues, run_id,
            ],
        )
    else:
        execute(
            """
            INSERT INTO pipeline_runs (
                id, status, config, items_fetched, items_filtered,
                posts_accepted, posts_rejected, avg_quality, avg_relevance,
                pass_rate, common_issues
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, status, json.dumps(config) if config else None,
                items_fetched, items_filtered,
                posts_accepted, posts_rejected,
                avg_quality, avg_relevance, pass_rate, common_issues,
            ],
        )
    return {"success": True}


@mcp_server.tool()
def get_recent_issues(last_n_runs: int = 5) -> dict:
    """Get common issues from recent pipeline runs for self-improvement."""
    rows = fetch_all(
        """
        SELECT common_issues FROM pipeline_runs
        WHERE common_issues IS NOT NULL AND common_issues != '[]' AND common_issues != ''
        ORDER BY started_at DESC LIMIT ?
        """,
        [last_n_runs],
    )

    import json as _json
    all_issues: dict[str, int] = {}
    for (issues,) in rows:
        if issues:
            try:
                issue_list = _json.loads(issues) if isinstance(issues, str) else issues
            except Exception:
                continue
            for issue in (issue_list or []):
                all_issues[issue] = all_issues.get(issue, 0) + 1

    sorted_issues = sorted(all_issues.items(), key=lambda x: x[1], reverse=True)
    return {
        "common_issues": [i[0] for i in sorted_issues[:10]],
        "recommendations": [f"Address: {i[0]} (seen {i[1]}x)" for i in sorted_issues[:5]],
    }
