"""
MCP LLM Tools
=============
Tools for LLM operations: generate posts, batch relevance judging,
structured output generation. Backend-agnostic (Ollama/vLLM).
"""
from __future__ import annotations

import json
import logging
import os
import httpx
import instructor
from openai import AsyncOpenAI
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


def _llm_cfg() -> dict:
    return ConfigManager.llm()


def _get_backend() -> str:
    return _llm_cfg().get("backend", "ollama")


def _get_model() -> str:
    return _llm_cfg().get("model", "qwen2.5:7b")


def _get_judge_model() -> str:
    return _llm_cfg().get("judge_model") or _get_model()


def _get_ollama_url() -> str:
    return _llm_cfg().get("ollama_url", "http://localhost:11434")


def _load_groq_keys_from_env() -> list[str]:
    """Load GROQ_API_KEY_1, _2, _3 ... from .env file and environment."""
    from pathlib import Path

    env_vars: dict[str, str] = {}
    env_file = Path(__file__).parents[3] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()

    keys: list[str] = []
    i = 1
    while True:
        key = env_vars.get(f"GROQ_API_KEY_{i}") or os.environ.get(f"GROQ_API_KEY_{i}", "")
        if not key:
            break
        keys.append(key)
        i += 1

    # Also accept a single GROQ_API_KEY
    single = env_vars.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    if single and single not in keys:
        keys.insert(0, single)

    return keys


def _get_providers() -> list[dict]:
    """Build provider list from .env keys + config.

    Each Groq key becomes a separate provider entry.
    Local Ollama is always the final fallback.
    """
    cfg = _llm_cfg()
    groq_model = cfg.get("model", "llama-3.1-8b-instant")
    groq_judge = cfg.get("judge_model", "llama-3.3-70b-versatile")

    groq_keys = _load_groq_keys_from_env()
    providers: list[dict] = [
        {
            "name": f"groq-{i+1}",
            "api_url": "https://api.groq.com/openai/v1",
            "api_key": key,
            "model": groq_model,
            "judge_model": groq_judge,
        }
        for i, key in enumerate(groq_keys)
    ]

    # Append local Ollama as final fallback
    providers.append({
        "name": "ollama",
        "api_url": None,
        "api_key": "",
        "model": "qwen2.5:7b",
        "judge_model": "qwen2.5:7b",
    })

    return providers


async def _cloud_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    json_mode: bool = False,
    provider: dict | None = None,
) -> str:
    """Call an OpenAI-compatible cloud API for one provider."""
    cfg = provider or _llm_cfg()
    base_url = cfg.get("api_url", "https://api.groq.com/openai/v1")
    api_key = cfg.get("api_key", "")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    client = _get_client()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _chat_with_fallback(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    use_judge: bool = False,
    format_schema: dict | None = None,
    seed: int | None = None,
) -> str:
    """Try each provider in order, falling back on rate limit (429) or error."""
    if _get_backend() == "ollama":
        # Local-only mode — skip cloud entirely
        url = f"{_get_ollama_url()}/api/chat"
        model = _get_judge_model() if use_judge else _get_model()
        payload: dict[str, Any] = {
            "model": model, "messages": messages,
            "stream": False, "options": {"temperature": temperature},
        }
        if format_schema:
            payload["format"] = format_schema
        if seed is not None:
            payload["options"]["seed"] = seed
        data = await _ollama_request(url, payload)
        return data["message"]["content"]

    providers = _get_providers()
    last_error: Exception | None = None

    for p in providers:
        if p.get("api_url") is None:
            # Ollama local fallback
            try:
                url = f"{_get_ollama_url()}/api/chat"
                model = p["judge_model"] if use_judge else p["model"]
                payload = {
                    "model": model, "messages": messages,
                    "stream": False, "options": {"temperature": temperature},
                }
                if format_schema:
                    payload["format"] = format_schema
                if seed is not None:
                    payload["options"]["seed"] = seed
                data = await _ollama_request(url, payload)
                logger.info("Fell back to local Ollama (%s)", model)
                return data["message"]["content"]
            except Exception as e:
                last_error = e
                continue

        model = p["judge_model"] if use_judge else p["model"]
        try:
            result = await _cloud_chat(messages, model, temperature, json_mode, provider=p)
            if p != _get_providers()[0]:
                logger.info("Using fallback provider '%s' (model: %s)", p.get("name"), model)
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit hit on provider '%s', trying next...", p.get("name"))
                last_error = e
                continue
            raise
        except Exception as e:
            logger.warning("Provider '%s' failed: %s, trying next...", p.get("name"), e)
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


async def _generate_post_with_instructor(
    messages: list[dict],
    temperature: float,
    seed: int | None = None,
) -> LinkedInPost:
    """Generate a structured LinkedInPost using instructor with provider fallback.

    instructor enforces the Pydantic schema and auto-retries on validation failure,
    replacing the manual json.loads + bare-except pattern.
    """
    if _get_backend() == "ollama":
        # Local-only mode
        ollama_url = _get_ollama_url()
        raw_client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
        client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        kwargs: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            kwargs["seed"] = seed
        return await client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            response_model=LinkedInPost,
            max_retries=2,
            **kwargs,
        )

    providers = _get_providers()
    last_error: Exception | None = None

    for p in providers:
        try:
            if p.get("api_url") is None:
                # Ollama fallback
                ollama_url = _get_ollama_url()
                raw_client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
                client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
                model = p["model"]
                kwargs = {"temperature": temperature}
                if seed is not None:
                    kwargs["seed"] = seed
            else:
                raw_client = AsyncOpenAI(base_url=p["api_url"], api_key=p["api_key"])
                client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
                model = p["model"]
                kwargs = {"temperature": temperature}

            post = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=LinkedInPost,
                max_retries=2,
                **kwargs,
            )
            if p.get("name", "").startswith("groq-") and p != providers[0]:
                logger.info("Using fallback provider '%s'", p.get("name"))
            return post

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                logger.warning("Rate limit on provider '%s', trying next...", p.get("name"))
            else:
                logger.warning("Provider '%s' failed: %s, trying next...", p.get("name"), e)
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


async def _ollama_request(url: str, payload: dict, retries: int = 2) -> dict:
    """Make a request to Ollama with retry logic."""
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


async def _ollama_chat(
    messages: list[dict],
    temperature: float = 0.7,
    format_schema: dict | None = None,
    seed: int | None = None,
) -> str:
    """Generation calls — tries providers in order, falls back on rate limit."""
    return await _chat_with_fallback(
        messages, temperature,
        json_mode=format_schema is not None,
        use_judge=False,
        format_schema=format_schema,
        seed=seed,
    )


async def _judge_chat(
    messages: list[dict],
    temperature: float = 0.2,
    format_schema: dict | None = None,
) -> str:
    """Evaluation/judging calls — uses judge model, tries providers in order."""
    return await _chat_with_fallback(
        messages, temperature,
        json_mode=format_schema is not None,
        use_judge=True,
        format_schema=format_schema,
    )




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
    avoid_hook_patterns: list[str] | None = None,
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
        content_ctx = f"\n\nExtracted article content:\n{content[:8000]}"

    # Use the v2 system prompt from resources (dynamic, config-driven)
    system_prompt = post_generation_prompt(avoid_patterns=avoid_hook_patterns)

    url_line = f"\nSource URL (MUST appear verbatim in the url field): {source_url}" if source_url else ""

    user_prompt = f"""Write a LinkedIn post about this content:

Title: {title}
Source: {source}{url_line}
Summary: {summary}{content_ctx}{trending_ctx}{avoid_ctx}{issues_ctx}{lessons_ctx}{feedback_ctx}

Respond in JSON format matching the output format specified above."""

    # Use instructor for validated structured output — auto-retries on schema mismatch
    try:
        post = await _generate_post_with_instructor(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            seed=seed,
        )
        return {
            "post_text": post.full_text,
            "angle": post.angle,
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
    except Exception as e:
        logger.error("Post generation failed: %s", e)
        return {
            "post_text": "",
            "error": f"Generation failed: {e}",
            "char_count": 0,
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

    messages = [
        {"role": "system", "content": "You are an AI content relevance judge. Respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    raw = await _judge_chat(messages=messages, temperature=0.3, format_schema=schema)

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
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
