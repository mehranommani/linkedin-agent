"""
MCP LLM Tools
=============
Tools for LLM operations: generate posts, batch relevance judging,
structured output generation. Backend-agnostic (Ollama/vLLM).
"""
from __future__ import annotations

import json
import logging
import httpx
from typing import Any

from backend.mcp.server import mcp_server
from backend.config import ConfigManager
from backend.models import LinkedInPost

logger = logging.getLogger(__name__)

# Module-level persistent HTTP client (connection pooling)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create a persistent httpx client with connection pooling."""
    global _client
    if _client is None or _client.is_closed:
        read_timeout = ConfigManager.llm().get("timeout", 180)
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,    # fail fast if Ollama is unreachable
                read=read_timeout,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=60,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the persistent client. Called during app shutdown."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def _get_ollama_url() -> str:
    return ConfigManager.llm().get("ollama_url", "http://localhost:11434")


def _get_model() -> str:
    return ConfigManager.llm().get("model", "qwen2.5:7b")


def _get_judge_model() -> str:
    """Model used for relevance judging — can be a larger model than the generation model."""
    return ConfigManager.llm().get("judge_model") or _get_model()


async def _ollama_request(url: str, payload: dict, retries: int = 2) -> dict:
    """Make a request to Ollama with retry logic and connection pooling."""
    for attempt in range(retries + 1):
        try:
            client = _get_client()
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < retries:
                import asyncio
                wait = 3 * (attempt + 1)
                logger.warning(f"Ollama connection error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


async def _ollama_generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    format_schema: dict | None = None,
    seed: int | None = None,
    model: str | None = None,
) -> str:
    """Call Ollama API for text generation."""
    url = f"{_get_ollama_url()}/api/generate"
    payload: dict[str, Any] = {
        "model": model or _get_model(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if format_schema:
        payload["format"] = format_schema
    if seed is not None:
        payload["options"]["seed"] = seed

    data = await _ollama_request(url, payload)
    return data["response"]


async def _ollama_chat(
    messages: list[dict],
    temperature: float = 0.7,
    format_schema: dict | None = None,
    seed: int | None = None,
) -> str:
    """Call Ollama chat API."""
    url = f"{_get_ollama_url()}/api/chat"
    payload: dict[str, Any] = {
        "model": _get_model(),
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_schema:
        payload["format"] = format_schema
    if seed is not None:
        payload["options"]["seed"] = seed

    data = await _ollama_request(url, payload)
    return data["message"]["content"]


@mcp_server.tool()
async def generate_post(
    title: str,
    summary: str,
    source: str,
    source_url: str = "",
    content: str | None = None,
    trending_topics: list[str] | None = None,
    avoid_phrases: list[str] | None = None,
    previous_issues: list[str] | None = None,
    lessons_learned: str | None = None,
    feedback_patterns: str | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
) -> dict:
    """Generate a LinkedIn post using structured LLM output with v2 prompts.

    Uses LinkedIn-optimized prompts with hook patterns, SEO keywords,
    recruiter-targeted terms, emoji guidance, and dynamic config.
    """
    from backend.mcp.resources.prompts import post_generation_prompt

    # Build context additions
    trending_ctx = ""
    if trending_topics:
        trending_ctx = f"\nTrending topics to consider weaving in: {', '.join(trending_topics)}"

    avoid_ctx = ""
    if avoid_phrases:
        avoid_ctx = f"\nAVOID these phrases: {', '.join(avoid_phrases[:10])}"

    issues_ctx = ""
    if previous_issues:
        issues_ctx = f"\nPrevious attempt had these issues — FIX THEM: {'; '.join(previous_issues[:5])}"

    lessons_ctx = ""
    if lessons_learned:
        lessons_ctx = f"\nLessons from past runs: {lessons_learned}"

    feedback_ctx = ""
    if feedback_patterns:
        feedback_ctx = f"\nHuman feedback patterns: {feedback_patterns}"

    content_ctx = ""
    if content:
        content_ctx = f"\n\nExtracted article content:\n{content[:3000]}"

    # Use the v2 system prompt from resources (dynamic, config-driven)
    system_prompt = post_generation_prompt()

    url_line = f"\nSource URL (MUST appear verbatim in the url field): {source_url}" if source_url else ""

    user_prompt = f"""Write a LinkedIn post about this content:

Title: {title}
Source: {source}{url_line}
Summary: {summary}{content_ctx}{trending_ctx}{avoid_ctx}{issues_ctx}{lessons_ctx}{feedback_ctx}

Respond in JSON format matching the output format specified above."""

    # Extended schema with v2 fields
    schema = {
        "type": "object",
        "properties": {
            "hook": {"type": "string"},
            "body": {"type": "string"},
            "takeaway": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "url": {"type": "string"},
            "hook_pattern_used": {"type": "string"},
            "seo_keywords_used": {"type": "array", "items": {"type": "string"}},
            "emoji_placement": {"type": "string"},
        },
        "required": ["hook", "body", "takeaway", "hashtags", "url"],
    }

    raw = await _ollama_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        format_schema=schema,
        seed=seed,
    )

    try:
        data = json.loads(raw)
        post = LinkedInPost(**data)
        return {
            "post_text": post.full_text,
            "hook": post.hook,
            "body": post.body,
            "takeaway": post.takeaway,
            "hashtags": post.hashtags,
            "url": post.url,
            "char_count": len(post.full_text),
            "hook_pattern_used": post.hook_pattern_used,
            "seo_keywords_used": post.seo_keywords_used,
            "emoji_placement": post.emoji_placement,
        }
    except (json.JSONDecodeError, Exception) as e:
        return {
            "post_text": raw,
            "error": f"Structured output parse failed: {e}",
            "char_count": len(raw),
        }


@mcp_server.tool()
async def batch_judge_relevance(
    items: list[dict],
) -> dict:
    """Judge relevance of multiple content items in a single LLM call.

    Each item should have 'title' and 'summary' fields.
    Returns relevance scores and boolean judgments.
    """
    items_text = "\n".join(
        f"{i+1}. [{item.get('title', '')}] {item.get('summary', '')[:150]}"
        for i, item in enumerate(items)
    )

    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "is_relevant": {"type": "boolean"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "is_relevant", "score"],
                },
            },
        },
        "required": ["results"],
    }

    prompt = f"""You are an AI content relevance judge for a LinkedIn page focused on AI, ML, Data Science, GenAI, and Agentic AI.

Rate each item's relevance on a 0-10 scale:
- 9-10: Core AI/ML topic, major breakthrough or tool
- 7-8: Relevant AI-adjacent content useful for practitioners
- 5-6: Somewhat related to AI/tech
- 0-4: Not relevant

Items to judge:
{items_text}

Return a JSON object with a "results" array. Each result has: index (1-based), is_relevant (true if score >= 7), score (0-10), reason (brief)."""

    raw = await _ollama_generate(
        prompt=prompt,
        temperature=0.3,
        format_schema=schema,
        model=_get_judge_model(),
    )

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        # Fallback: mark all as relevant
        return {
            "results": [
                {"index": i + 1, "is_relevant": True, "score": 7.0, "reason": "parse_fallback"}
                for i in range(len(items))
            ]
        }


@mcp_server.tool()
async def structured_generate(
    prompt: str,
    schema: dict,
    system: str = "",
    temperature: float = 0.7,
) -> dict:
    """General-purpose structured generation with a JSON schema."""
    raw = await _ollama_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ] if system else [
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        format_schema=schema,
    )

    try:
        return {"result": json.loads(raw), "success": True}
    except json.JSONDecodeError as e:
        return {"result": raw, "success": False, "error": str(e)}
