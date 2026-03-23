"""
Hindsight Memory Client
=======================
Wrapper for Hindsight agent memory system (vectorize.io/hindsight).

Uses the official hindsight-client SDK:
- Hindsight(base_url=...) for initialization
- retain(): store information with metadata and tags
- recall(): multi-strategy retrieval (semantic + BM25 + graph + temporal)
- reflect(): LLM-powered reasoning over stored memories

All config driven by ConfigManager — no hardcoded values.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx
from hindsight_client import Hindsight

from backend.config import ConfigManager

logger = logging.getLogger(__name__)

# Singleton client
_client: Hindsight | None = None


def _get_config() -> dict:
    """Get Hindsight config from ConfigManager."""
    return ConfigManager.hindsight()


def _get_base_url() -> str:
    """Resolve Hindsight URL from env var or config."""
    return os.environ.get("HINDSIGHT_URL", _get_config()["url"])


def _get_bank_id() -> str:
    """Get the memory bank ID from config."""
    return _get_config().get("bank_id", "linkedin_writing_style")


def get_client() -> Hindsight:
    """Get or create the singleton Hindsight client."""
    global _client
    if _client is None:
        url = _get_base_url()
        _client = Hindsight(base_url=url)
        logger.info("Initialized Hindsight client at %s", url)
    return _client


async def close_client() -> None:
    """Close the Hindsight client (call on app shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            _client.close()
        _client = None
        logger.info("Hindsight client closed")


async def ensure_bank() -> None:
    """Create the memory bank if it doesn't exist yet."""
    client = get_client()
    try:
        await client.acreate_bank(
            bank_id=_get_bank_id(),
            name="LinkedIn Writing Style",
            mission="Learn user writing style preferences from feedback on LinkedIn posts.",
            enable_observations=True,
            observations_mission=(
                "Synthesize patterns about user writing style preferences: "
                "length, tone, structure, emoji usage, hashtag density, "
                "topic-specific patterns, and what to avoid."
            ),
            reflect_mission=(
                "You are a writing style advisor. Based on stored user feedback, "
                "provide specific, actionable guidance on how to write LinkedIn posts "
                "that match the user's preferences. Focus on style patterns, not exact content."
            ),
        )
        logger.info("Created Hindsight memory bank: %s", _get_bank_id())
    except Exception as e:
        # Bank may already exist — that's fine
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            logger.debug("Memory bank %s already exists", _get_bank_id())
        else:
            logger.warning("Could not create memory bank: %s", e)


async def store_feedback(
    post_id: str,
    post_text: str,
    post_title: str,
    rating: int,
    feedback_text: str | None,
    is_good_example: bool,
    source: str,
) -> bool:
    """
    Store human feedback in Hindsight for autonomous learning.

    Uses official SDK: client.aretain(bank_id, content, metadata, tags)
    """
    client = get_client()

    # Build content string (context param is str, not dict)
    parts = [
        f"Post Title: {post_title}",
        f"Post Preview: {post_text[:300]}",
        f"User Rating: {rating}/5 stars",
    ]
    if feedback_text:
        parts.append(f"User Feedback: {feedback_text}")
    if is_good_example:
        parts.append("User Marked As: Good Example")

    content = "\n".join(parts)

    # Metadata must be dict[str, str] per SDK
    metadata = {
        "post_id": post_id,
        "rating": str(rating),
        "source": source,
        "is_good_example": str(is_good_example).lower(),
    }

    # Tags for filtering
    tags = [f"rating:{rating}", f"source:{source}", "feedback"]
    if is_good_example:
        tags.append("good_example")
    if rating >= 4:
        tags.append("positive")
    elif rating <= 2:
        tags.append("negative")

    try:
        result = await client.aretain(
            bank_id=_get_bank_id(),
            content=content,
            metadata=metadata,
            tags=tags,
            timestamp=datetime.now(),
        )
        logger.info(
            "Stored feedback in Hindsight: post=%s rating=%d success=%s",
            post_id[:8], rating, result.success,
        )
        return result.success
    except Exception as e:
        logger.error("Failed to store feedback in Hindsight: %s", e)
        return False


async def get_style_guidance(
    topic: str,
    source: str,
    content_summary: str | None = None,
) -> str:
    """
    Get autonomous style guidance from learned preferences.

    Uses official SDK: client.areflect(bank_id, query) -> ReflectResponse.text
    """
    client = get_client()

    query_parts = [
        f"I'm writing a LinkedIn post about: {topic}",
        f"Content source: {source}",
    ]
    if content_summary:
        query_parts.append(f"Summary: {content_summary[:200]}")

    query_parts.append(
        "Based on all stored user feedback and learned preferences, "
        "provide concise style guidance for this specific post."
    )
    query = "\n".join(query_parts)

    try:
        hs_config = _get_config()
        result = await client.areflect(
            bank_id=_get_bank_id(),
            query=query,
            budget=hs_config.get("reflect_budget", "mid"),
        )
        guidance = result.text
        if guidance:
            logger.info(
                "Retrieved style guidance from Hindsight (%d chars) for topic='%s'",
                len(guidance), topic[:40],
            )
        return guidance or ""
    except httpx.ConnectError:
        logger.warning("Hindsight server not reachable — generating without style guidance")
        return ""
    except Exception as e:
        logger.error("Failed to get style guidance from Hindsight: %s", e)
        return ""


async def recall_feedback(query: str, max_tokens: int | None = None) -> list[dict]:
    """
    Recall relevant memories using multi-strategy retrieval.

    Uses official SDK: client.arecall(bank_id, query) -> RecallResponse.results
    """
    client = get_client()
    try:
        hs_config = _get_config()
        tokens = max_tokens or hs_config.get("recall_max_tokens", 4096)
        result = await client.arecall(
            bank_id=_get_bank_id(),
            query=query,
            max_tokens=tokens,
        )
        return [
            {
                "id": r.id,
                "text": r.text,
                "type": r.type,
                "metadata": r.metadata,
                "tags": r.tags,
            }
            for r in result.results
        ]
    except Exception as e:
        logger.error("Failed to recall from Hindsight: %s", e)
        return []


async def check_health() -> dict:
    """Check if Hindsight server is reachable."""
    url = _get_base_url()
    try:
        async with httpx.AsyncClient() as http:
            response = await http.get(f"{url}/health", timeout=5.0)
            return {"status": "healthy", "code": response.status_code}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
