"""
Hacker News Source
==================
Fetches top stories from the Hacker News Firebase API and filters
by minimum score.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource

logger = logging.getLogger(__name__)

_BASE = "https://hacker-news.firebaseio.com/v0"
_HEADERS = {"User-Agent": "LinkedInAgent/2.0"}


class HackerNewsSource(BaseSource):
    """Top stories from Hacker News filtered by score."""

    source_type: str = "hackernews"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch top HN stories.

        ``params`` keys
        ---------------
        min_score : int – minimum points to include (default 50)
        limit     : int – override for *limit*
        """
        min_score: int = params.get("min_score", 50)
        effective_limit: int = params.get("limit", limit)

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                # 1. get top story IDs
                resp = await client.get(f"{_BASE}/topstories.json")
                resp.raise_for_status()
                story_ids: list[int] = resp.json()

                # Fetch more than we need to account for filtering
                candidate_ids = story_ids[: effective_limit * 3]

                # 2. fetch each story concurrently (bounded)
                stories = await self._fetch_stories(client, candidate_ids)

            # 3. filter & sort
            filtered = [
                s for s in stories
                if s is not None and s.get("score", 0) >= min_score
            ]
            filtered.sort(key=lambda s: s.get("score", 0), reverse=True)

            items: list[ContentItem] = []
            for story in filtered[: effective_limit]:
                url = story.get("url") or f"https://news.ycombinator.com/item?id={story['id']}"
                items.append(
                    ContentItem(
                        title=story.get("title", ""),
                        url=url,
                        source="hackernews",
                        summary=story.get("title", ""),
                        metrics={
                            "score": story.get("score", 0),
                            "comments": story.get("descendants", 0),
                        },
                        timestamp=float(story.get("time", time.time())),
                    )
                )
            return items

        except Exception as exc:
            logger.error("HackerNewsSource.fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------

    @staticmethod
    async def _fetch_stories(
        client: httpx.AsyncClient, ids: list[int]
    ) -> list[dict | None]:
        """Fetch story details concurrently with a semaphore."""
        sem = asyncio.Semaphore(20)

        async def _get(story_id: int) -> dict | None:
            async with sem:
                try:
                    r = await client.get(f"{_BASE}/item/{story_id}.json")
                    r.raise_for_status()
                    return r.json()
                except httpx.HTTPError:
                    return None

        return await asyncio.gather(*[_get(sid) for sid in ids])
