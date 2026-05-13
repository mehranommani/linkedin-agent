"""
Sources API
===========
CRUD for content sources.
"""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, HTTPException

from backend.database import execute, fetch_one, fetch_all
from backend.models import SourceCreate

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("")
def list_sources():
    """List all sources."""
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
    sources = []
    for row in rows:
        src = dict(zip(columns, row))
        if isinstance(src["params"], str):
            try:
                src["params"] = json.loads(src["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        if src["last_fetched_at"]:
            src["last_fetched_at"] = str(src["last_fetched_at"])
        sources.append(src)
    return {"sources": sources, "total": len(sources)}


@router.post("")
def create_source(source: SourceCreate):
    """Create a new source."""
    source_id = hashlib.md5(
        f"{source.source_type}-{source.name}".encode()
    ).hexdigest()[:12]

    execute(
        """
        INSERT INTO sources (id, source_type, name, url, category, params, is_active, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            source_id, source.source_type, source.name, source.url,
            source.category, json.dumps(source.params),
            source.is_active, source.priority,
        ],
    )
    return {"id": source_id, "created": True}


# Static routes MUST come before parametrized /{source_id} routes
@router.get("/health")
def source_health():
    """Get health status of all sources including error tracking."""
    rows = fetch_all("""
        SELECT id, source_type, name, url, is_active,
               last_error, last_error_at, consecutive_failures,
               total_fetched, total_accepted, avg_quality, last_fetched_at
        FROM sources
        ORDER BY consecutive_failures DESC, source_type, name
    """)
    columns = [
        "id", "source_type", "name", "url", "is_active",
        "last_error", "last_error_at", "consecutive_failures",
        "total_fetched", "total_accepted", "avg_quality", "last_fetched_at",
    ]
    sources = []
    healthy = 0
    unhealthy = 0
    disabled = 0
    for row in rows:
        src = dict(zip(columns, row))
        failures = src.get("consecutive_failures") or 0
        active = src.get("is_active", True)

        if not active:
            src["health"] = "disabled"
            disabled += 1
        elif failures >= 3:
            src["health"] = "critical"
            unhealthy += 1
        elif failures >= 1:
            src["health"] = "degraded"
            unhealthy += 1
        else:
            src["health"] = "healthy"
            healthy += 1

        for ts_field in ("last_error_at", "last_fetched_at"):
            if src[ts_field]:
                src[ts_field] = str(src[ts_field])
        sources.append(src)

    return {
        "sources": sources,
        "summary": {
            "total": len(sources),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "disabled": disabled,
        },
    }


@router.patch("/{source_id}")
def update_source(source_id: str, updates: dict):
    """Update a source's fields."""
    row = fetch_one("SELECT 1 FROM sources WHERE id = ?", [source_id])
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    allowed = {"name", "url", "category", "params", "is_active", "priority"}
    for key, value in updates.items():
        if key not in allowed:
            continue
        if key == "params":
            execute("UPDATE sources SET params = ? WHERE id = ?",
                    [json.dumps(value), source_id])
        else:
            execute(f"UPDATE sources SET {key} = ? WHERE id = ?", [value, source_id])

    return {"updated": True}


@router.delete("/{source_id}")
def delete_source(source_id: str):
    """Delete a source."""
    row = fetch_one("SELECT 1 FROM sources WHERE id = ?", [source_id])
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    execute("DELETE FROM sources WHERE id = ?", [source_id])
    return {"deleted": True}
