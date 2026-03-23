"""
Dev.to Source
=============
Fetches recent articles from the Dev.to public API, filtered by tag.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "LinkedInAgent/2.0",
    "Accept": "application/json",
}
_API_BASE = "https://dev.to/api/articles"


class DevToSource(BaseSource):
    """Recent articles from Dev.to filtered by tags."""

    source_type: str = "devto"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch articles from Dev.to.

        ``params`` keys
        ---------------
        tags : list[str] – e.g. ["ai", "machinelearning"]
        """
        tags: list[str] = params.get("tags", ["ai", "machinelearning"])

        all_articles: list[dict] = []

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                for tag in tags:
                    articles = await self._fetch_tag(client, tag)
                    all_articles.extend(articles)
        except Exception as exc:
            logger.error("DevToSource.fetch failed: %s", exc)
            return []

        # deduplicate by article id
        seen_ids: set[int] = set()
        unique: list[dict] = []
        for art in all_articles:
            aid = art.get("id", 0)
            if aid not in seen_ids:
                seen_ids.add(aid)
                unique.append(art)

        # sort by positive reactions descending
        unique.sort(
            key=lambda a: a.get("positive_reactions_count", 0), reverse=True
        )

        items: list[ContentItem] = []
        for art in unique[:limit]:
            title = art.get("title", "")
            url = art.get("url", "")
            description = art.get("description", "") or ""
            cover = art.get("cover_image", "") or art.get("social_image", "") or ""

            reactions = art.get("positive_reactions_count", 0)
            comments = art.get("comments_count", 0)

            ts = self._parse_iso(art.get("published_at", ""))

            items.append(
                ContentItem(
                    title=title,
                    url=url,
                    source="devto",
                    summary=description[:500],
                    image_url=cover,
                    metrics={
                        "reactions": reactions,
                        "comments": comments,
                    },
                    timestamp=ts,
                )
            )

        return items

    # ------------------------------------------------------------------

    @staticmethod
    async def _fetch_tag(client: httpx.AsyncClient, tag: str) -> list[dict]:
        """Fetch articles for a single tag."""
        try:
            resp = await client.get(
                _API_BASE,
                params={"tag": tag, "per_page": 25, "top": 7},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("DevTo: failed to fetch tag %r: %s", tag, exc)
            return []

    @staticmethod
    def _parse_iso(s: str) -> float:
        """Parse an ISO-8601 datetime string to a Unix timestamp."""
        if not s:
            return time.time()
        try:
            # Dev.to returns e.g. "2025-05-18T14:30:00Z"
            s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.timestamp()
        except (ValueError, TypeError):
            return time.time()
