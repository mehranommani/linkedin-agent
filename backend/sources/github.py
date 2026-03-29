"""
GitHub Source
=============
Fetches trending and notable AI/ML repositories from the GitHub Search API.
No HTML scraping — API-only for reliability.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from backend.models import ContentItem
from backend.sources.base import BaseSource
from backend.config import ConfigManager

logger = logging.getLogger(__name__)

_API_HEADERS = {
    "User-Agent": "LinkedInAgent/2.0",
    "Accept": "application/vnd.github+json",
}


class GitHubSource(BaseSource):
    """Fetches repos via the GitHub Search API (no scraping)."""

    source_type: str = "github"

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    async def fetch(self, params: dict, limit: int = 15) -> list[ContentItem]:
        """Return up to *limit* ContentItems from GitHub.

        ``params`` keys
        ---------------
        languages : list[str]   – e.g. ["python", "jupyter-notebook"]
        since     : str         – "daily", "weekly", or "monthly"
        queries   : list[str]   – free-text queries for the Search API
        github_token : str      – optional PAT for higher rate limits
        min_stars_new : int     – min stars for newly-created repos (default 500)
        min_stars_established : int – min stars for established repos (default 5000)
        """
        items: list[ContentItem] = []
        languages: list[str] = params.get("languages", ["python"])
        since: str = params.get("since", "weekly")
        queries: list[str] = params.get("queries", [])
        # Viral new repos: 500+ stars in a week is genuinely trending
        min_stars_new: int = params.get("min_stars_new", params.get("min_stars", 500))
        # Established repos: 5k+ stars with recent activity
        min_stars_established: int = params.get("min_stars_established", 5000)

        # Optional GitHub token for higher rate limits
        headers = dict(_API_HEADERS)
        token = params.get("github_token") or ConfigManager.get("github_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(timeout=20, headers=headers) as client:
                # --- viral new repos: recently created, gaining traction fast ---
                for lang in languages:
                    trending = await self._search_trending(client, lang, since, min_stars_new)
                    items.extend(trending)

                # --- established repos with recent activity (releases, pushes) ---
                established = await self._search_established(client, languages, min_stars_established, since)
                items.extend(established)

                # --- explicit queries (with star floor) ---
                for query in queries:
                    searched = await self._search_repos(client, query, min_stars_new)
                    items.extend(searched)
        except Exception as exc:
            logger.error("GitHubSource.fetch failed: %s", exc)
            return []

        # deduplicate by URL, keep first occurrence
        seen: set[str] = set()
        unique: list[ContentItem] = []
        for item in items:
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)

        # Filter out well-known repos from skip list
        skip = set(r.lower() for r in ConfigManager.skip_repos())
        filtered = [
            i for i in unique
            if i.title.split("/")[-1].lower() not in skip
        ]

        return filtered[:limit]

    # ------------------------------------------------------------------
    # trending via search API
    # ------------------------------------------------------------------

    async def _search_trending(
        self, client: httpx.AsyncClient, language: str, since: str, min_stars: int
    ) -> list[ContentItem]:
        """Find trending repos via GitHub Search API (replaces HTML scraper)."""
        days = {"daily": 1, "weekly": 7, "monthly": 30}.get(since, 7)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        api_url = "https://api.github.com/search/repositories"
        api_params = {
            "q": f"language:{language} stars:>{min_stars} created:>{cutoff}",
            "sort": "stars",
            "order": "desc",
            "per_page": 20,
        }
        return await self._fetch_search(client, api_url, api_params)

    # ------------------------------------------------------------------
    # explicit query search
    # ------------------------------------------------------------------

    async def _search_established(
        self, client: httpx.AsyncClient, languages: list[str], min_stars: int, since: str
    ) -> list[ContentItem]:
        """Find established high-star AI/ML repos with recent activity (pushes/releases).

        GitHub Search API does not support OR for topic filters, so we run two
        targeted queries with different topic anchors and merge results.
        """
        days = {"daily": 3, "weekly": 14, "monthly": 30}.get(since, 14)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        lang_filter = " ".join(f"language:{lang}" for lang in languages[:2])
        api_url = "https://api.github.com/search/repositories"

        # Two complementary topic groups — broad enough to cover LLMs, CV, RL, data science
        topic_groups = ["machine-learning", "llm"]
        items: list[ContentItem] = []
        seen: set[str] = set()

        for topic in topic_groups:
            result = await self._fetch_search(
                client, api_url,
                {
                    "q": f"topic:{topic} {lang_filter} stars:>{min_stars} pushed:>{cutoff}",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 6,
                },
                fetch_readme=False,
            )
            for item in result:
                if item.url not in seen:
                    seen.add(item.url)
                    items.append(item)

        return items

    async def _search_repos(
        self, client: httpx.AsyncClient, query: str, min_stars: int = 500
    ) -> list[ContentItem]:
        """Use the GitHub Search API to find recently-created repos matching a query."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        api_url = "https://api.github.com/search/repositories"
        api_params = {
            "q": f"{query} stars:>{min_stars} created:>{cutoff}",
            "sort": "stars",
            "order": "desc",
            "per_page": 20,
        }
        return await self._fetch_search(client, api_url, api_params)

    # ------------------------------------------------------------------
    # shared search helper
    # ------------------------------------------------------------------

    async def _fetch_search(
        self, client: httpx.AsyncClient, api_url: str, api_params: dict,
        fetch_readme: bool = True,
    ) -> list[ContentItem]:
        """Execute a search query and return ContentItems."""
        try:
            resp = await client.get(api_url, params=api_params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("GitHub search failed for %r: %s", api_params.get("q"), exc)
            return []

        items: list[ContentItem] = []
        for repo in data.get("items", []):
            # README fetching is skipped for established repos to avoid rate limits
            readme_content = ""
            if fetch_readme:
                readme_content = await self._fetch_readme(client, repo.get("full_name", ""))

            description = repo.get("description") or ""
            summary = description
            if readme_content:
                # Use first 500 chars of README for richer context
                summary = f"{description}\n\n{readme_content[:500]}" if description else readme_content[:500]

            topics = repo.get("topics", [])

            items.append(
                ContentItem(
                    title=repo.get("full_name", ""),
                    url=repo.get("html_url", ""),
                    source="github",
                    summary=summary,
                    metrics={
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "language": repo.get("language") or "",
                        "topics": topics,
                        "open_issues": repo.get("open_issues_count", 0),
                    },
                    timestamp=time.time(),
                    extracted_content=readme_content,
                )
            )
        return items

    # ------------------------------------------------------------------
    # README extraction
    # ------------------------------------------------------------------

    async def _fetch_readme(self, client: httpx.AsyncClient, full_name: str) -> str:
        """Fetch the README content of a repo (best-effort)."""
        if not full_name:
            return ""
        url = f"https://api.github.com/repos/{full_name}/readme"
        try:
            resp = await client.get(url, headers={"Accept": "application/vnd.github.raw+json"})
            if resp.status_code == 200:
                return resp.text[:2000]
        except Exception:
            pass
        return ""
