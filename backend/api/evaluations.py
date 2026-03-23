"""
Evaluations API
===============
Evaluation reports and history.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Query

from backend.database import fetch_all, fetch_one


def _parse_json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
from backend.mcp.tools.database import get_recent_issues

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/reports")
def list_reports(limit: int = Query(20, ge=1, le=100)):
    """List recent pipeline run evaluation reports."""
    rows = fetch_all("""
        SELECT id, started_at, completed_at, status,
               items_fetched, items_filtered, posts_accepted, posts_rejected,
               avg_quality, avg_relevance, pass_rate, common_issues
        FROM pipeline_runs
        ORDER BY started_at DESC LIMIT ?
    """, [limit])

    columns = [
        "run_id", "started_at", "completed_at", "status",
        "items_fetched", "items_filtered", "posts_accepted", "posts_rejected",
        "avg_quality", "avg_relevance", "pass_rate", "common_issues",
    ]
    reports = []
    for row in rows:
        report = dict(zip(columns, row))
        for key in ("started_at", "completed_at"):
            if report.get(key):
                report[key] = str(report[key])
        report["common_issues"] = _parse_json_list(report.get("common_issues"))
        reports.append(report)

    return {"reports": reports}


@router.get("/issues")
def get_common_issues(last_n_runs: int = 5):
    """Get common issues from recent pipeline runs."""
    return get_recent_issues(last_n_runs)


@router.get("/post/{post_id}")
def get_post_evaluations(post_id: str):
    """Get evaluations for a specific post (v2 detailed + v1 legacy)."""
    from backend.evaluation.evaluator import get_evaluation_for_post

    # Try v2 detailed evaluation first
    detailed = get_evaluation_for_post(post_id)
    if detailed:
        return {"found": True, "evaluation": detailed, "post_id": post_id}

    # Fallback to v1 evaluations
    rows = fetch_all("""
        SELECT id, evaluated_at, relevance_score, quality_score,
               faithfulness_score, confidence_score, passed,
               issues, recommendations, judge_model, evaluation_type
        FROM evaluations
        WHERE post_id = ?
        ORDER BY evaluated_at DESC
    """, [post_id])

    columns = [
        "id", "evaluated_at", "relevance_score", "quality_score",
        "faithfulness_score", "confidence_score", "passed",
        "issues", "recommendations", "judge_model", "evaluation_type",
    ]
    evals = []
    for row in rows:
        ev = dict(zip(columns, row))
        if ev.get("evaluated_at"):
            ev["evaluated_at"] = str(ev["evaluated_at"])
        ev["issues"] = _parse_json_list(ev.get("issues"))
        ev["recommendations"] = _parse_json_list(ev.get("recommendations"))
        evals.append(ev)

    if evals:
        return {"found": True, "evaluation": evals[0], "post_id": post_id}
    return {"found": False, "post_id": post_id}
