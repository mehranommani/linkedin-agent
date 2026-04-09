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
from backend.mcp.tools.llm import _load_env_vars as _load_env_file

logger = logging.getLogger(__name__)

_API_HEADERS = {
    "User-Agent": "LinkedInAgent/2.0",
    "Accept": "application/vnd.github+json",
}


def _get_github_token(params: dict) -> str:
    """Return a GitHub token, checking params → DB config → .env in order."""
    if t := params.get("github_token"):
        return t
    if t := ConfigManager.get("github_token"):
        return t
    env = _load_env_file()
    return env.get("GITHUB_TOKEN", "")


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
        # Viral new repos: 1k+ stars in a week = genuinely trending (high signal)
        min_stars_new: int = params.get("min_stars_new", params.get("min_stars", 1000))
        # Established repos: 5k+ stars with recent activity
        min_stars_established: int = params.get("min_stars_established", 5000)
        # Topics for established repo search (configurable per-source)
        established_topics: list[str] | None = params.get("established_topics")

        # Optional GitHub token for higher rate limits (params → DB config → .env)
        headers = dict(_API_HEADERS)
        token = _get_github_token(params)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(timeout=20, headers=headers) as client:
                # --- viral new repos: recently created, gaining traction fast ---
                for lang in languages:
                    trending = await self._search_trending(client, lang, since, min_stars_new)
                    items.extend(trending)

                # --- established repos with recent activity (releases, pushes) ---
                established = await self._search_established(
                    client, languages, min_stars_established, since, established_topics
                )
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
        self, client: httpx.AsyncClient, languages: list[str], min_stars: int, since: str,
        established_topics: list[str] | None = None,
    ) -> list[ContentItem]:
        """Find established high-star AI/ML repos with recent activity (pushes/releases).

        GitHub Search API does not support OR for topic filters, so we run one
        targeted query per topic anchor and merge results.
        """
        days = {"daily": 3, "weekly": 14, "monthly": 30}.get(since, 14)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        # Use only the primary language — GitHub Search AND-joins multiple language: filters,
        # which returns nothing for repos that aren't simultaneously written in all of them.
        primary_lang = languages[0] if languages else "python"
        lang_filter = f"language:{primary_lang}"
        api_url = "https://api.github.com/search/repositories"

        topic_groups = established_topics or ["machine-learning", "llm"]
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
                fetch_readme=True,
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

            releases_summary = await self._fetch_releases(client, repo.get("full_name", ""))

            description = repo.get("description") or ""
            summary = description
            if readme_content:
                # Use first 2000 chars of README for richer context
                summary = f"{description}\n\n{readme_content[:2000]}" if description else readme_content[:2000]
            if releases_summary:
                summary = f"{summary}\n\nRecent releases: {releases_summary}"

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
                return resp.text[:10000]
        except Exception:
            pass
        return ""

    async def _fetch_releases(self, client: httpx.AsyncClient, full_name: str) -> str:
        """Fetch the 3 most recent releases for a repo (best-effort)."""
        if not full_name:
            return ""
        url = f"https://api.github.com/repos/{full_name}/releases"
        try:
            resp = await client.get(url, params={"per_page": 3})
            if resp.status_code == 200:
                releases = resp.json()
                parts = []
                for r in releases[:3]:
                    name = r.get("name") or r.get("tag_name", "")
                    body = (r.get("body") or "")[:500].strip()
                    if name:
                        parts.append(f"{name}: {body}" if body else name)
                return " | ".join(parts)
        except Exception:
            pass
        return ""
