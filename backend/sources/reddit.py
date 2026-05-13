"""
Reddit Source
=============
Fetches hot posts from a subreddit via the public Reddit JSON API.
"""
from __future__ import annotations

import logging
import time

import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "LinkedInAgent/2.0",
    "Accept": "application/json",
}


class RedditSource(BaseSource):
    """Hot posts from a subreddit, filtered by score."""

    source_type: str = "reddit"

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Fetch hot posts from Reddit.

        ``params`` keys
        ---------------
        subreddit : str – subreddit name (default "LocalLLaMA")
        min_score : int – minimum upvotes to include (default 100)
        """
        subreddit: str = params.get("subreddit", "LocalLLaMA")
        min_score: int = params.get("min_score", 100)

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=20"

        try:
            async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("RedditSource.fetch failed: %s", exc)
            return []

        items: list[ContentItem] = []
        children = data.get("data", {}).get("children", [])

        for child in children:
            post = child.get("data", {})

            # skip stickied / pinned
            if post.get("stickied", False):
                continue

            score = post.get("score", 0)
            if score < min_score:
                continue

            title = post.get("title", "")
            permalink = post.get("permalink", "")
            post_url = f"https://www.reddit.com{permalink}" if permalink else ""
            selftext = post.get("selftext", "") or ""
            summary = selftext[:500] if selftext else title

            # optional thumbnail / image
            image_url = ""
            preview = post.get("preview", {})
            if preview:
                images = preview.get("images", [])
                if images:
                    image_url = images[0].get("source", {}).get("url", "")
                    # Reddit HTML-encodes the URL
                    image_url = image_url.replace("&amp;", "&")

            items.append(
                ContentItem(
                    title=title,
                    url=post_url,
                    source="reddit",
                    summary=summary,
                    image_url=image_url,
                    metrics={
                        "score": score,
                        "comments": post.get("num_comments", 0),
                        "subreddit": subreddit,
                    },
                    timestamp=float(post.get("created_utc", time.time())),
                )
            )

        # sort by score descending
        items.sort(key=lambda i: i.metrics.get("score", 0), reverse=True)
        return items[:limit]
