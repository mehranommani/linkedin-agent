"""
Posts API
=========
CRUD + paginated listing + stats for LinkedIn posts.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException

from backend.database import execute, fetch_one
from backend.mcp.tools.database import query_posts, get_source_stats

router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("")
def list_posts(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    source: str | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    min_score: float | None = None,
    is_used: bool | None = None,
):
    """Paginated post listing with filters."""
    result = query_posts(
        page=page, per_page=per_page, source=source,
        search=search, sort_by=sort_by, min_score=min_score, is_used=is_used,
    )
    total = result["total"]
    pages = (total + per_page - 1) // per_page
    return {**result, "pages": pages}


@router.get("/stats/summary")
def post_stats():
    """Dashboard summary statistics."""
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
    source_stats = get_source_stats()
    return {
        "total_posts": row[0],
        "used_posts": row[1],
        "avg_relevance": row[2],
        "avg_quality": row[3],
        "avg_score": row[4],
        "avg_length": int(row[5]) if row[5] else 0,
        "source_distribution": source_stats.get("stats", []),
    }


@router.get("/{post_id}")
def get_post(post_id: str):
    """Get a single post by ID."""
    row = fetch_one("SELECT * FROM posts WHERE id = ?", [post_id])
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    columns = [desc[0] for desc in execute("SELECT * FROM posts LIMIT 0").description]
    post = dict(zip(columns, row))
    if post.get("created_at"):
        post["created_at"] = str(post["created_at"])
    return post


@router.patch("/{post_id}/used")
def toggle_used(post_id: str):
    """Toggle the used status of a post."""
    row = fetch_one("SELECT is_used FROM posts WHERE id = ?", [post_id])
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    new_val = not row[0]
    execute("UPDATE posts SET is_used = ? WHERE id = ?", [new_val, post_id])
    return {"id": post_id, "is_used": new_val}


@router.delete("/{post_id}")
def delete_post(post_id: str):
    """Delete a post."""
    row = fetch_one("SELECT 1 FROM posts WHERE id = ?", [post_id])
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    execute("DELETE FROM evaluations WHERE post_id = ?", [post_id])
    execute("DELETE FROM posts WHERE id = ?", [post_id])
    return {"deleted": True}
