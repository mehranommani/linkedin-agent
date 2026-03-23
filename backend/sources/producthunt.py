"""
Product Hunt Source
===================
Scrapes Product Hunt's AI / artificial-intelligence topic page
for recently-launched products.
"""
from __future__ import annotations

import logging
import re
import time

import httpx
from bs4 import BeautifulSoup

from backend.models import ContentItem
from backend.sources.base import BaseSource

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "LinkedInAgent/2.0",
    "Accept": "text/html,application/xhtml+xml",
}
_TOPIC_URL = "https://www.producthunt.com/topics/artificial-intelligence"


class ProductHuntSource(BaseSource):
    """AI products from Product Hunt (scraped)."""

    source_type: str = "producthunt"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch AI products from Product Hunt.

        ``params`` keys
        ---------------
        limit : int – override for *limit*
        """
        effective_limit: int = params.get("limit", limit)

        try:
            async with httpx.AsyncClient(
                timeout=15, headers=_HEADERS, follow_redirects=True
            ) as client:
                resp = await client.get(_TOPIC_URL)
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            logger.error("ProductHuntSource.fetch failed: %s", exc)
            return []

        return self._parse_html(html, effective_limit)

    # ------------------------------------------------------------------

    def _parse_html(self, html: str, limit: int) -> list[ContentItem]:
        """Parse the Product Hunt topic page."""
        soup = BeautifulSoup(html, "html.parser")
        items: list[ContentItem] = []

        # Product Hunt renders product cards in various ways.
        # We try multiple selectors to be resilient to layout changes.
        product_cards = (
            soup.select('[data-test="post-item"]')
            or soup.select("div.styles_item__")
            or soup.select("li[class*='item']")
            or soup.select("a[href*='/posts/']")
        )

        for card in product_cards[:limit * 2]:
            try:
                item = self._parse_card(card)
                if item is not None:
                    items.append(item)
            except Exception as exc:
                logger.debug("Skipping PH card: %s", exc)
                continue

        # deduplicate by URL
        seen: set[str] = set()
        unique: list[ContentItem] = []
        for item in items:
            if item.url and item.url not in seen:
                seen.add(item.url)
                unique.append(item)

        return unique[:limit]

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_card(card: BeautifulSoup) -> ContentItem | None:
        """Extract a single product from a card element."""
        # title
        title_el = (
            card.select_one("h3")
            or card.select_one("h2")
            or card.select_one("a[data-test='post-name']")
            or card.select_one("strong")
        )
        if title_el is None:
            return None
        title = title_el.get_text(strip=True)
        if not title:
            return None

        # link
        link_el = card if card.name == "a" else card.select_one("a[href]")
        href = link_el.get("href", "") if link_el else ""
        if href and not href.startswith("http"):
            href = f"https://www.producthunt.com{href}"
        if not href:
            return None

        # tagline / description
        tagline_el = card.select_one("p") or card.select_one("[class*='tagline']")
        tagline = tagline_el.get_text(strip=True) if tagline_el else ""

        # vote count
        votes = 0
        vote_el = (
            card.select_one("[data-test='vote-button']")
            or card.select_one("button[class*='vote']")
            or card.select_one("[class*='upvote']")
        )
        if vote_el:
            m = re.search(r"(\d+)", vote_el.get_text())
            if m:
                votes = int(m.group(1))

        # image
        img_el = card.select_one("img")
        image_url = img_el.get("src", "") if img_el else ""

        return ContentItem(
            title=title,
            url=href,
            source="producthunt",
            summary=tagline[:500],
            image_url=image_url,
            metrics={"votes": votes},
            timestamp=time.time(),
        )
