"""
Feedback API
============
CRUD for human feedback on generated posts.
Supports rating (1-5 stars), text feedback, good example marking.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from backend.database import execute, fetch_one, fetch_all
from backend.models import HumanFeedbackCreate, HumanFeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/{post_id}")
async def submit_feedback(post_id: str, feedback: HumanFeedbackCreate):
    """Submit human feedback for a post and store in Hindsight for autonomous learning."""
    # Verify post exists and get post details
    post = fetch_one(
        "SELECT id, post_text, title, source FROM posts WHERE id = ?",
        [post_id]
    )
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post_text, title, source = post[1], post[2], post[3]

    # Store feedback in DuckDB (for stats and history)
    feedback_id = uuid.uuid4().hex[:12]
    execute(
        """INSERT INTO human_feedback
        (id, post_id, rating, text_feedback, is_good_example)
        VALUES (?, ?, ?, ?, ?)""",
        [feedback_id, post_id, feedback.rating, feedback.text_feedback, feedback.is_good_example],
    )

    # Update post's aggregated feedback stats
    stats = fetch_one(
        """SELECT AVG(rating), COUNT(*) FROM human_feedback WHERE post_id = ?""",
        [post_id],
    )
    if stats:
        execute(
            """UPDATE posts SET human_rating_avg = ?, human_feedback_count = ? WHERE id = ?""",
            [stats[0] or 0, stats[1] or 0, post_id],
        )

    # Store in Hindsight for autonomous learning (non-blocking)
    from backend.memory.hindsight_client import store_feedback as store_in_hindsight

    try:
        await store_in_hindsight(
            post_id=post_id,
            post_text=post_text,
            post_title=title,
            rating=feedback.rating,
            feedback_text=feedback.text_feedback,
            is_good_example=feedback.is_good_example,
            source=source,
        )
    except Exception as e:
        # Log error but don't fail the feedback submission
        import logging
        logging.getLogger(__name__).warning(
            "Failed to store feedback in Hindsight (non-critical): %s", e
        )

    return {"id": feedback_id, "post_id": post_id, "created": True}


@router.get("/{post_id}")
def get_post_feedback(post_id: str):
    """Get all feedback for a specific post."""
    rows = fetch_all(
        """SELECT id, post_id, created_at, rating, text_feedback,
                  is_good_example, feedback_weight, feedback_source
           FROM human_feedback WHERE post_id = ?
           ORDER BY created_at DESC""",
        [post_id],
    )
    columns = [
        "id", "post_id", "created_at", "rating", "text_feedback",
        "is_good_example", "feedback_weight", "feedback_source",
    ]
    feedbacks = []
    for row in rows:
        fb = dict(zip(columns, row))
        if fb["created_at"]:
            fb["created_at"] = str(fb["created_at"])
        feedbacks.append(fb)

    return {"feedbacks": feedbacks, "count": len(feedbacks)}


@router.get("/summary/stats")
def feedback_summary():
    """Aggregate feedback stats: avg rating by source, most-rated posts."""
    # Overall stats
    overall = fetch_one(
        "SELECT AVG(rating), COUNT(*), COUNT(DISTINCT post_id) FROM human_feedback"
    )

    # Avg rating by source
    by_source = fetch_all("""
        SELECT p.source, AVG(hf.rating) as avg_rating, COUNT(*) as count
        FROM human_feedback hf
        JOIN posts p ON hf.post_id = p.id
        GROUP BY p.source
        ORDER BY avg_rating DESC
    """)

    # Top rated posts
    top_posts = fetch_all("""
        SELECT p.id, p.title, p.human_rating_avg, p.human_feedback_count
        FROM posts p
        WHERE p.human_feedback_count > 0
        ORDER BY p.human_rating_avg DESC
        LIMIT 10
    """)

    return {
        "overall": {
            "avg_rating": overall[0] if overall else 0,
            "total_feedbacks": overall[1] if overall else 0,
            "unique_posts_rated": overall[2] if overall else 0,
        },
        "by_source": [
            {"source": r[0], "avg_rating": r[1], "count": r[2]}
            for r in by_source
        ],
        "top_rated_posts": [
            {"id": r[0], "title": r[1], "avg_rating": r[2], "feedback_count": r[3]}
            for r in top_posts
        ],
    }


@router.get("/good-examples/list")
def get_good_examples(limit: int = 20):
    """Get posts marked as good examples (for few-shot prompting)."""
    rows = fetch_all("""
        SELECT DISTINCT p.id, p.title, p.post_text, p.source,
               p.quality_score, p.human_rating_avg
        FROM posts p
        JOIN human_feedback hf ON hf.post_id = p.id
        WHERE hf.is_good_example = TRUE
        ORDER BY p.human_rating_avg DESC
        LIMIT ?
    """, [limit])

    return {
        "examples": [
            {
                "id": r[0], "title": r[1], "post_text": r[2],
                "source": r[3], "quality_score": r[4], "human_rating_avg": r[5],
            }
            for r in rows
        ],
        "count": len(rows),
    }
