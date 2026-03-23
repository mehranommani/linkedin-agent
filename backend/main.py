"""
LinkedIn AI Content Agent - Backend
===================================
FastAPI application with MCP server and LangGraph pipeline.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import asyncio

from backend.database import get_connection, close as close_db, execute, fetch_one
from backend.mcp.server import register_all_tools
from backend.api.router import api_router
from backend.api.websocket import router as ws_router

# Configure logging to stderr so we can see pipeline errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)
# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)


async def _stale_run_cleanup_loop():
    """Every 5 minutes mark pipeline runs stuck in 'running' for >30 min as 'failed'."""
    while True:
        await asyncio.sleep(300)
        try:
            count = fetch_one(
                "SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running'"
                " AND completed_at IS NULL AND started_at < datetime('now', '-30 minutes')"
            )[0]
            if count:
                execute(
                    "UPDATE pipeline_runs SET status = 'failed', completed_at = datetime('now')"
                    " WHERE status = 'running' AND completed_at IS NULL"
                    " AND started_at < datetime('now', '-30 minutes')"
                )
                logging.getLogger(__name__).warning("Marked %d stale pipeline run(s) as failed", count)
        except Exception as e:
            logging.getLogger(__name__).error("Stale-run cleanup failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: initialize DB and register MCP tools
    get_connection()
    register_all_tools()

    # Startup: background task to auto-fail stuck pipeline runs
    cleanup_task = asyncio.create_task(_stale_run_cleanup_loop())

    # Startup: initialize Hindsight memory bank (non-blocking)
    try:
        from backend.memory.hindsight_client import ensure_bank
        await ensure_bank()
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Hindsight memory bank init skipped (server may not be running): %s", e
        )

    yield

    # Shutdown: cancel background cleanup task
    cleanup_task.cancel()

    # Shutdown: close Hindsight client
    try:
        from backend.memory.hindsight_client import close_client as close_hindsight
        await close_hindsight()
    except Exception:
        pass

    # Shutdown: close HTTP client pool and DB
    from backend.mcp.tools.llm import close_client
    await close_client()
    close_db()


app = FastAPI(
    title="LinkedIn AI Content Agent",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(api_router)
# WebSocket routes mounted at root (not under /api) so Vite can proxy /ws -> ws://localhost:9000
app.include_router(ws_router)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    from backend.database import fetch_one
    count = fetch_one("SELECT COUNT(*) FROM posts")[0]
    return {"status": "healthy", "posts_count": count}


# MCP tools diagnostic endpoint
@app.get("/api/mcp/tools")
def list_mcp_tools():
    """List all registered MCP tools."""
    from backend.mcp.server import mcp_server
    tools = mcp_server._tool_manager._tools
    return {
        "tools": [
            {"name": name, "description": tool.description[:100] if tool.description else ""}
            for name, tool in sorted(tools.items())
        ],
        "count": len(tools),
    }
