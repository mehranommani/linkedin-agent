"""
Agent API
=========
Start/stop/status for pipeline runs.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

import json

from backend.database import fetch_all, fetch_one


def _parse_json_list(value) -> list:
    """Deserialize a JSON string to a list, returning [] on failure."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
from backend.pipeline.graph import run_pipeline
from backend.models import PipelineRunConfig
from backend.api.websocket import broadcast

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# Track running pipeline tasks
_running_tasks: dict[str, asyncio.Task] = {}
_latest_state: dict[str, dict] = {}


@router.post("/run")
async def start_pipeline(config: PipelineRunConfig, background_tasks: BackgroundTasks):
    """Start a pipeline run."""
    # Check if already running
    for run_id, task in list(_running_tasks.items()):
        if not task.done():
            raise HTTPException(status_code=409, detail=f"Pipeline already running: {run_id}")

    async def _run():
        try:
            logger.info("Starting pipeline run with config: %s", config.dict())
            state = await run_pipeline(
                max_posts=config.max_posts,
                max_age_days=config.max_age_days,
                limit_per_source=config.limit_per_source,
                max_retries=config.max_retries,
                sources=config.sources,
                broadcast_fn=broadcast,
            )
            run_id = state.get("run_id", "unknown")
            _latest_state[run_id] = dict(state)
            logger.info("Pipeline completed: run_id=%s, accepted=%d, error=%s",
                        run_id, len(state.get("accepted_posts", [])), state.get("error"))
            return state
        except Exception as e:
            logger.error("Pipeline task crashed: %s\n%s", e, traceback.format_exc())
            raise

    task = asyncio.create_task(_run())
    # We'll get the run_id from the task result, but we need a placeholder
    placeholder_id = f"pending_{id(task)}"
    _running_tasks[placeholder_id] = task

    return {"status": "started", "message": "Pipeline started in background"}


@router.get("/status/{run_id}")
def get_pipeline_status(run_id: str):
    """Get the status of a pipeline run."""
    row = fetch_one("""
        SELECT id, started_at, completed_at, status, items_fetched,
               items_filtered, posts_accepted, posts_rejected,
               avg_quality, pass_rate, common_issues
        FROM pipeline_runs WHERE id = ?
    """, [run_id])

    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": row[0],
        "started_at": str(row[1]) if row[1] else None,
        "completed_at": str(row[2]) if row[2] else None,
        "status": row[3],
        "items_fetched": row[4],
        "items_filtered": row[5],
        "posts_accepted": row[6],
        "posts_rejected": row[7],
        "avg_quality": row[8],
        "pass_rate": row[9],
        "common_issues": _parse_json_list(row[10]),
    }


@router.post("/cancel/{run_id}")
def cancel_pipeline(run_id: str):
    """Cancel a running pipeline."""
    for key, task in _running_tasks.items():
        if not task.done():
            task.cancel()
            return {"cancelled": True, "run_id": run_id}
    raise HTTPException(status_code=404, detail="No running pipeline found")


@router.get("/runs")
def list_runs(limit: int = 20):
    """List recent pipeline runs."""
    rows = fetch_all("""
        SELECT id, started_at, completed_at, status, items_fetched,
               items_filtered, posts_accepted, posts_rejected,
               avg_quality, pass_rate
        FROM pipeline_runs
        ORDER BY started_at DESC LIMIT ?
    """, [limit])

    columns = [
        "run_id", "started_at", "completed_at", "status", "items_fetched",
        "items_filtered", "posts_accepted", "posts_rejected",
        "avg_quality", "pass_rate",
    ]
    return {"runs": [dict(zip(columns, row)) for row in rows]}


@router.post("/cleanup-stale")
def cleanup_stale_runs(older_than_minutes: int = 30):
    """Mark stale 'running' runs as 'failed'. Default threshold: 30 minutes."""
    from backend.database import execute
    count = fetch_one(
        "SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running' AND completed_at IS NULL"
        " AND started_at < datetime('now', ? || ' minutes')",
        [f"-{older_than_minutes}"],
    )[0]
    execute(
        "UPDATE pipeline_runs SET status = 'failed', completed_at = datetime('now')"
        " WHERE status = 'running' AND completed_at IS NULL"
        " AND started_at < datetime('now', ? || ' minutes')",
        [f"-{older_than_minutes}"],
    )
    return {"cleaned": count, "message": f"Marked {count} stale runs as failed"}
