"""
MCP Source Tools
================
Tools for fetching content from all sources.
The pipeline discovers available sources dynamically via list_available.
"""
from __future__ import annotations

import json
import httpx
from typing import Any

from backend.mcp.server import mcp_server
from backend.database import fetch_all
from backend.sources.github import GitHubSource
from backend.sources.hackernews import HackerNewsSource
from backend.sources.reddit import RedditSource
from backend.sources.rss import RSSSource
from backend.sources.arxiv_source import ArxivSource
from backend.sources.devto import DevToSource
from backend.sources.producthunt import ProductHuntSource
from backend.sources.papers import PapersWithCodeSource

# Registry of source implementations
_SOURCE_REGISTRY: dict[str, Any] = {
    "github": GitHubSource(),
    "hackernews": HackerNewsSource(),
    "reddit": RedditSource(),
    "rss": RSSSource(),
    "arxiv": ArxivSource(),
    "devto": DevToSource(),
    "producthunt": ProductHuntSource(),
    "papers": PapersWithCodeSource(),
}


@mcp_server.tool()
def list_available() -> dict:
    """List all available content sources and their status."""
    rows = fetch_all("""
        SELECT id, source_type, name, is_active, priority, avg_quality
        FROM sources ORDER BY priority ASC, name ASC
    """)
    sources = []
    for row in rows:
        sources.append({
            "id": row[0],
            "source_type": row[1],
            "name": row[2],
            "is_active": row[3],
            "priority": row[4],
            "avg_quality": row[5],
            "has_implementation": row[1] in _SOURCE_REGISTRY,
        })
    return {"sources": sources}


@mcp_server.tool()
async def fetch_source(
    source_type: str,
    params: dict | None = None,
    limit: int = 15,
) -> dict:
    """Fetch content items from a specific source type.

    Args:
        source_type: One of github, hackernews, reddit, rss, arxiv, devto, producthunt, papers
        params: Source-specific parameters (e.g. subreddit, feed_url, languages)
        limit: Maximum items to return
    """
    if source_type not in _SOURCE_REGISTRY:
        return {"items": [], "count": 0, "error": f"Unknown source type: {source_type}"}

    source = _SOURCE_REGISTRY[source_type]
    try:
        items = await source.fetch(params or {}, limit=limit)
        return {
            "items": [item.model_dump() for item in items],
            "count": len(items),
        }
    except Exception as e:
        return {"items": [], "count": 0, "error": str(e)}


@mcp_server.tool()
async def extract_article(url: str) -> dict:
    """Extract article content from a URL using html2text."""
    try:
        import html2text

        async with httpx.AsyncClient(
            timeout=15,
            headers={"User-Agent": "LinkedInAgent/2.0"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.body_width = 0
        content = converter.handle(resp.text)

        # Trim to reasonable length
        content = content[:5000]

        return {"content": content, "char_count": len(content), "success": True}
    except Exception as e:
        return {"content": "", "char_count": 0, "success": False, "error": str(e)}
