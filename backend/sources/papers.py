"""
Papers with Code Source
=======================
Fetches trending papers from the Papers with Code public API.
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
_API_BASE = "https://paperswithcode.com/api/v1/papers/"


class PapersWithCodeSource(BaseSource):
    """Trending papers from Papers with Code."""

    source_type: str = "papers"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch papers from Papers with Code.

        ``params`` keys
        ---------------
        limit : int – number of papers to return
        """
        effective_limit: int = params.get("limit", limit)

        query_params = {
            "ordering": "-proceeding",
            "items_per_page": effective_limit * 2,  # over-fetch to filter blanks
        }

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                resp = await client.get(_API_BASE, params=query_params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("PapersWithCodeSource.fetch failed: %s", exc)
            return []

        results = data.get("results", [])
        items: list[ContentItem] = []

        for paper in results:
            try:
                title = paper.get("title", "").strip()
                if not title:
                    continue

                paper_id = paper.get("id", "")
                url_slug = paper.get("url_slug", "") or paper_id
                paper_url = paper.get("paper_url", "")
                if not paper_url:
                    paper_url = f"https://paperswithcode.com/paper/{url_slug}" if url_slug else ""
                if not paper_url:
                    continue

                abstract = paper.get("abstract", "").strip()
                if not abstract:
                    abstract = title

                # published date
                pub_date = paper.get("published", "")
                ts = self._parse_date(pub_date)

                # authors
                authors = paper.get("authors", []) or []
                author_names = []
                for a in authors[:5]:
                    if isinstance(a, str):
                        author_names.append(a)
                    elif isinstance(a, dict):
                        author_names.append(a.get("name", ""))

                # conference / proceeding
                proceeding = paper.get("proceeding", "") or ""

                items.append(
                    ContentItem(
                        title=title,
                        url=paper_url,
                        source="papers",
                        summary=abstract[:600],
                        metrics={
                            "authors": author_names,
                            "proceeding": proceeding,
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
        """Parse a date string from the PwC API into a Unix timestamp."""
        if not s:
            return time.time()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        return time.time()
