"""
Config API
==========
Dynamic config CRUD.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.config import ConfigManager
from backend.database import fetch_all

router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
def list_config():
    """List all config entries."""
    rows = fetch_all("SELECT key, value, description, updated_at FROM config ORDER BY key")
    configs = []
    for row in rows:
        configs.append({
            "key": row[0],
            "value": row[1],
            "description": row[2],
            "updated_at": str(row[3]) if row[3] else None,
        })
    return {"configs": configs}


@router.get("/{key}")
def get_config(key: str):
    """Get a config value."""
    value = ConfigManager.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Config key not found: {key}")
    return {"key": key, "value": value}


@router.put("/{key}")
def set_config(key: str, body: dict):
    """Set a config value."""
    value = body.get("value")
    description = body.get("description")
    if value is None:
        raise HTTPException(status_code=400, detail="'value' field required")
    ConfigManager.set(key, value, description)
    return {"key": key, "updated": True}
