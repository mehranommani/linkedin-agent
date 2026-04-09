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
logging.getLogger("aiohttp").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


async def _stale_run_cleanup_loop():
    """Every 10 minutes mark pipeline runs stuck in 'running' for >3 hours as 'failed'.
    3h covers heavy runs: 15 posts × 3 retries × ~6 min/post (generate + 4-parallel eval)."""
    while True:
        await asyncio.sleep(600)
        try:
            count = fetch_one(
                "SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running'"
                " AND completed_at IS NULL AND started_at < datetime('now', '-180 minutes')"
            )[0]
            if count:
                execute(
                    "UPDATE pipeline_runs SET status = 'failed', completed_at = datetime('now')"
                    " WHERE status = 'running' AND completed_at IS NULL"
                    " AND started_at < datetime('now', '-180 minutes')"
                )
                logging.getLogger(__name__).warning("Marked %d stale pipeline run(s) as failed", count)
        except Exception as e:
            logging.getLogger(__name__).error("Stale-run cleanup failed: %s", e)


def _configure_dspy() -> None:
    """Configure DSPy LM using the first available cloud key (Cerebras → Groq → Ollama)."""
    try:
        import dspy
        from backend.mcp.tools.llm import _load_env_vars

        env = _load_env_vars()
        log = logging.getLogger(__name__)

        if key := env.get("CEREBRAS_API_KEY"):
            dspy.configure(lm=dspy.LM(
                "openai/qwen-3-235b-a22b-instruct-2507",
                api_key=key,
                api_base="https://api.cerebras.ai/v1",
                max_tokens=2000,
            ))
            log.info("DSPy LM configured: Cerebras (qwen-3-235b)")
            return

        if key := (env.get("GROQ_API_KEY") or env.get("GROQ_API_KEY_1")):
            dspy.configure(lm=dspy.LM(
                "openai/llama-3.3-70b-versatile",
                api_key=key,
                api_base="https://api.groq.com/openai/v1",
                max_tokens=2000,
            ))
            log.info("DSPy LM configured: Groq (llama-3.3-70b)")
            return

        # Fallback: local Ollama
        dspy.configure(lm=dspy.LM(
            "ollama_chat/qwen2.5:14b",
            api_base="http://localhost:11434",
            max_tokens=2000,
        ))
        log.info("DSPy LM configured: Ollama (qwen2.5:14b)")
    except Exception as exc:
        logging.getLogger(__name__).warning("DSPy configuration failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: initialize DB and register MCP tools
    get_connection()
    register_all_tools()

    # Startup: configure DSPy LM for post generation
    _configure_dspy()

    # Startup: background task to auto-fail stuck pipeline runs
    cleanup_task = asyncio.create_task(_stale_run_cleanup_loop())

    # Startup: initialize Hindsight memory bank in background (non-blocking)
    async def _init_hindsight():
        try:
            from backend.memory.hindsight_client import ensure_bank
            await ensure_bank()
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Hindsight memory bank init skipped: %s", e
            )
    hindsight_task = asyncio.create_task(_init_hindsight())

    yield

    # Shutdown: cancel background tasks
    cleanup_task.cancel()
    hindsight_task.cancel()

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
