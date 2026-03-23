"""
RSS Source
==========
Fetches and parses any RSS / Atom feed via ``feedparser``.
Handles redirects, tracks errors, auto-disables broken feeds.
"""
from __future__ import annotations

import logging
import time
from calendar import timegm
from email.utils import parsedate

import feedparser
import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource
from backend.database import execute

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "LinkedInAgent/2.0"}


class RSSSource(BaseSource):
    """Parse an RSS or Atom feed into ContentItems."""

    source_type: str = "rss"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch items from an RSS/Atom feed.

        ``params`` keys
        ---------------
        feed_url : str – the URL of the feed
        """
        feed_url: str = params.get("feed_url", "")
        source_id: str = params.get("_source_id", "")
        if not feed_url:
            logger.warning("RSSSource: no feed_url provided")
            return []

        try:
            async with httpx.AsyncClient(
                timeout=15,
                headers=_HEADERS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(feed_url)
                resp.raise_for_status()
                raw = resp.text

                # If the final URL differs (redirect), update stored URL
                final_url = str(resp.url)
                if final_url != feed_url and source_id:
                    logger.info("RSS redirect: %s -> %s", feed_url, final_url)
                    try:
                        import json
                        execute(
                            "UPDATE sources SET url = ?, params = ? WHERE id = ?",
                            [final_url, json.dumps({"feed_url": final_url}), source_id],
                        )
                    except Exception as e:
                        logger.debug("Could not update redirect URL: %s", e)

        except httpx.HTTPStatusError as exc:
            error_msg = f"HTTP {exc.response.status_code}"
            self._record_error(source_id, error_msg)
            logger.error("RSSSource.fetch HTTP error %s: %s", feed_url, error_msg)
            return []
        except httpx.ConnectError as exc:
            self._record_error(source_id, f"Connect error: {exc}")
            logger.error("RSSSource.fetch connect error %s: %s", feed_url, exc)
            return []
        except Exception as exc:
            self._record_error(source_id, str(exc))
            logger.error("RSSSource.fetch failed to download %s: %s", feed_url, exc)
            return []

        try:
            feed = feedparser.parse(raw)
        except Exception as exc:
            self._record_error(source_id, f"Parse error: {exc}")
            logger.error("RSSSource.fetch failed to parse %s: %s", feed_url, exc)
            return []

        # Success — reset error tracking
        if source_id:
            self._clear_errors(source_id)

        items: list[ContentItem] = []
        for entry in feed.entries[:limit]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            # strip HTML tags from summary (simple approach)
            summary = _strip_html(summary)[:500]

            timestamp = self._parse_entry_date(entry)

            items.append(
                ContentItem(
                    title=title,
                    url=link,
                    source="rss",
                    summary=summary,
                    timestamp=timestamp,
                )
            )

        return items

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _record_error(source_id: str, error_msg: str) -> None:
        """Record a fetch error and increment consecutive failure count."""
        if not source_id:
            return
        try:
            execute(
                """UPDATE sources SET
                    last_error = ?,
                    last_error_at = now(),
                    consecutive_failures = COALESCE(consecutive_failures, 0) + 1
                WHERE id = ?""",
                [error_msg[:500], source_id],
            )
            # Auto-disable after 3 consecutive failures
            execute(
                """UPDATE sources SET is_active = FALSE
                WHERE id = ? AND consecutive_failures >= 3""",
                [source_id],
            )
        except Exception as e:
            logger.debug("Could not record RSS error: %s", e)

    @staticmethod
    def _clear_errors(source_id: str) -> None:
        """Reset error tracking on successful fetch."""
        try:
            execute(
                """UPDATE sources SET
                    last_error = NULL,
                    last_error_at = NULL,
                    consecutive_failures = 0
                WHERE id = ?""",
                [source_id],
            )
        except Exception as e:
            logger.debug("Could not clear RSS errors: %s", e)

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entry_date(entry: dict) -> float:
        """Try to extract a Unix timestamp from a feed entry."""
        # feedparser normalises into ``published_parsed`` or ``updated_parsed``
        for key in ("published_parsed", "updated_parsed"):
            struct = entry.get(key)
            if struct is not None:
                try:
                    return float(timegm(struct))
                except Exception:
                    pass

        # fallback: try RFC-2822 strings
        for key in ("published", "updated"):
            raw = entry.get(key, "")
            if raw:
                parsed = parsedate(raw)
                if parsed is not None:
                    try:
                        return float(timegm(parsed))
                    except Exception:
                        pass

        return time.time()


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text* (best-effort, no external dep)."""
    import re

    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()
