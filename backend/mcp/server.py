"""
MCP Composite Server
====================
Single MCP server exposing all tools across 4 categories:
  - database.*  — DuckDB access
  - llm.*       — LLM operations
  - sources.*   — Content fetching
  - evaluator.* — Quality evaluation

Uses the FastMCP high-level API from the mcp SDK.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# Create the composite server
mcp_server = FastMCP(
    "LinkedIn Agent",
    instructions="MCP server for the LinkedIn AI content agent platform",
)


def register_all_tools():
    """Import and register all tool modules.

    Each module uses the @mcp_server.tool() decorator at import time,
    so we just need to trigger the imports.
    """
    from backend.mcp.tools import database as _db  # noqa: F401
    from backend.mcp.tools import llm as _llm  # noqa: F401
    from backend.mcp.tools import sources as _sources  # noqa: F401
    from backend.mcp.tools import evaluator as _eval  # noqa: F401
    from backend.mcp.resources import prompts as _prompts  # noqa: F401


def get_server() -> FastMCP:
    """Get the MCP server with all tools registered."""
    register_all_tools()
    return mcp_server
