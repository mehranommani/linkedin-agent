"""
ArXiv Source
============
Fetches recent papers from the ArXiv API filtered by category
and maximum age.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "LinkedInAgent/2.0"}
_ARXIV_API = "https://export.arxiv.org/api/query"

# ArXiv Atom namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivSource(BaseSource):
    """Recent papers from ArXiv, filtered by category and age."""

    source_type: str = "arxiv"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch papers from ArXiv.

        ``params`` keys
        ---------------
        categories   : list[str] – ArXiv categories, e.g. ["cs.AI", "cs.LG"]
        max_age_days : int       – only include papers newer than this (default 7)
        """
        categories: list[str] = params.get("categories", ["cs.AI", "cs.LG"])
        max_age_days: int = params.get("max_age_days", 7)

        # build the query string  e.g. "cat:cs.AI OR cat:cs.LG"
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        query_params = {
            "search_query": cat_query,
            "start": 0,
            "max_results": limit * 3,  # over-fetch to allow age filtering
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                resp = await client.get(_ARXIV_API, params=query_params)
                resp.raise_for_status()
                xml_text = resp.text
        except Exception as exc:
            logger.error("ArxivSource.fetch failed: %s", exc)
            return []

        return self._parse_feed(xml_text, max_age_days, limit)

    # ------------------------------------------------------------------

    def _parse_feed(
        self, xml_text: str, max_age_days: int, limit: int
    ) -> list[ContentItem]:
        """Parse the Atom XML returned by the ArXiv API."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("ArxivSource: XML parse error: %s", exc)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        items: list[ContentItem] = []

        for entry in root.findall("atom:entry", _NS):
            try:
                title_el = entry.find("atom:title", _NS)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

                # paper link (prefer the abstract page)
                link = ""
                for link_el in entry.findall("atom:link", _NS):
                    if link_el.get("type") == "text/html":
                        link = link_el.get("href", "")
                        break
                if not link:
                    id_el = entry.find("atom:id", _NS)
                    link = (id_el.text or "").strip() if id_el is not None else ""

                # abstract / summary
                summary_el = entry.find("atom:summary", _NS)
                abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

                # published date
                published_el = entry.find("atom:published", _NS)
                pub_str = (published_el.text or "").strip() if published_el is not None else ""
                pub_dt = self._parse_datetime(pub_str)

                if pub_dt and pub_dt < cutoff:
                    continue

                ts = pub_dt.timestamp() if pub_dt else time.time()

                # categories
                cats = [
                    cat_el.get("term", "")
                    for cat_el in entry.findall("atom:category", _NS)
                ]

                # authors
                authors = []
                for author_el in entry.findall("atom:author", _NS):
                    name_el = author_el.find("atom:name", _NS)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                items.append(
                    ContentItem(
                        title=title,
                        url=link,
                        source="arxiv",
                        summary=abstract[:600],
                        metrics={
                            "categories": cats,
                            "authors": authors[:5],
                        },
                        timestamp=ts,
                    )
                )
            except Exception as exc:
                logger.debug("Skipping arxiv entry: %s", exc)
                continue

        return items[:limit]

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_datetime(s: str) -> datetime | None:
        """Parse an ISO-8601 timestamp from ArXiv."""
        if not s:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None
