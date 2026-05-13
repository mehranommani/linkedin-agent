"""
Papers Source
=============
Fetches trending AI papers from the HuggingFace Daily Papers API.
Papers with Code API was deprecated and now redirects to HuggingFace.

API docs: https://huggingface.co/api/daily_papers
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
_HF_DAILY_PAPERS = "https://huggingface.co/api/daily_papers"


class PapersWithCodeSource(BaseSource):
    """Trending AI papers from HuggingFace Daily Papers."""

    source_type: str = "papers"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch today's trending papers from HuggingFace.

        ``params`` keys
        ---------------
        limit : int – number of papers to return
        """
        effective_limit: int = params.get("limit", limit)

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                resp = await client.get(_HF_DAILY_PAPERS)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("PapersWithCodeSource.fetch failed: %s", exc)
            return []

        items: list[ContentItem] = []

        for entry in data[:effective_limit * 2]:
            try:
                paper = entry.get("paper", {})

                title = paper.get("title", "").strip()
                if not title:
                    continue

                paper_id = paper.get("id", "")
                if not paper_id:
                    continue

                paper_url = f"https://huggingface.co/papers/{paper_id}"

                abstract = paper.get("summary", "") or paper.get("ai_summary", "")
                abstract = abstract.strip()
                if not abstract:
                    abstract = title

                pub_str = paper.get("publishedAt", "")
                ts = self._parse_date(pub_str)

                authors = paper.get("authors", []) or []
                author_names = [
                    a.get("name", "") for a in authors[:5]
                    if isinstance(a, dict) and a.get("name")
                ]

                upvotes = paper.get("upvotes", 0)
                github_repo = paper.get("githubRepo", "")

                items.append(
                    ContentItem(
                        title=title,
                        url=paper_url,
                        source="papers",
                        summary=abstract[:600],
                        metrics={
                            "authors": author_names,
                            "upvotes": upvotes,
                            "github_repo": github_repo or "",
                        },
                        timestamp=ts,
                    )
                )
            except Exception as exc:
                logger.debug("Skipping paper: %s", exc)
                continue

        return items[:effective_limit]

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(s: str) -> float:
        """Parse an ISO-8601 date string from the HuggingFace API."""
        if not s:
            return time.time()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        return time.time()
