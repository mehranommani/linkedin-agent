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
from backend.models import LinkedInPost, ResearchBrief

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
                connect=10.0,
                read=read_timeout,
                write=30.0,
                pool=60.0,  # wait up to 60s for a free connection from pool
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
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


def _get_ollama_url() -> str:
    return _llm_cfg().get("ollama_url", "http://localhost:11434")


# ── Provider registry ────────────────────────────────────────────────────────
# Each entry maps an env-key prefix to a provider config.
# Keys are loaded as PREFIX_1, PREFIX_2, PREFIX_3 from .env automatically.
# Priority order: first provider with a live key wins; Ollama is always last.
_PROVIDER_REGISTRY = [
    {
        "prefix": "CEREBRAS_API_KEY",
        "name": "cerebras",
        "api_url": "https://api.cerebras.ai/v1",
        "model": "qwen-3-235b-a22b-instruct-2507",  # Qwen3-235B on Cerebras — fast, free
    },
    {
        "prefix": "OPENROUTER_API_KEY",
        "name": "openrouter",
        "api_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",  # free tier on OpenRouter (stable)
        "extra_headers": {"HTTP-Referer": "https://github.com/linkedin-agent", "X-Title": "LinkedIn Agent"},
    },
    {
        "prefix": "SAMBANOVA_API_KEY",
        "name": "sambanova",
        "api_url": "https://api.sambanova.ai/v1",
        "model": "Meta-Llama-3.3-70B-Instruct",
    },
    {
        "prefix": "TOGETHER_API_KEY",
        "name": "together",
        "api_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    {
        "prefix": "MISTRAL_API_KEY",
        "name": "mistral",
        "api_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
    },
    {
        "prefix": "FIREWORKS_API_KEY",
        "name": "fireworks",
        "api_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    {
        "prefix": "GROQ_API_KEY",
        "name": "groq",
        "api_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
]

_OLLAMA_PREFERRED = ["qwen3.5:9b", "qwen2.5:14b", "qwen2.5:7b"]


def _load_env_vars() -> dict[str, str]:
    """Read .env file + os.environ into a single dict (env file takes priority)."""
    from pathlib import Path
    env_vars: dict[str, str] = dict(os.environ)
    env_file = Path(__file__).parents[3] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()
    return env_vars


def _load_keys_for_prefix(env_vars: dict[str, str], prefix: str) -> list[str]:
    """Return all values for PREFIX, PREFIX_1, PREFIX_2, … skipping blanks."""
    keys: list[str] = []
    # single bare key first
    if v := env_vars.get(prefix, ""):
        keys.append(v)
    i = 1
    while True:
        v = env_vars.get(f"{prefix}_{i}", "")
        if not v:
            break
        if v not in keys:
            keys.append(v)
        i += 1
    return keys


def _get_ollama_model() -> str:
    """Return the best Ollama model currently available locally."""
    import httpx as _httpx
    try:
        resp = _httpx.get(
            f"{_llm_cfg().get('ollama_url', 'http://localhost:11434')}/api/tags",
            timeout=3,
        )
        names = [m["name"] for m in resp.json().get("models", [])]
        for preferred in _OLLAMA_PREFERRED:
            base = preferred.split(":")[0]
            match = next((n for n in names if n.startswith(base)), None)
            if match:
                return match
    except Exception:
        pass
    return _OLLAMA_PREFERRED[-1]  # safe fallback


def _get_generation_providers() -> list[dict]:
    """Build ordered provider list for generation tasks.

    Reads all keys from .env in priority order:
      Cerebras → OpenRouter → SambaNova → Groq → Ollama (local fallback)

    Any provider with no key in .env is silently skipped.
    Ollama is always appended as the final fallback.
    """
    env_vars = _load_env_vars()
    providers: list[dict] = []

    for entry in _PROVIDER_REGISTRY:
        keys = _load_keys_for_prefix(env_vars, entry["prefix"])
        for i, key in enumerate(keys):
            providers.append({
                "name": f"{entry['name']}-{i+1}",
                "api_url": entry["api_url"],
                "api_key": key,
                "model": entry["model"],
            })

    # Ollama local fallback — always last
    providers.append({
        "name": "ollama",
        "api_url": None,
        "api_key": "",
        "model": _get_ollama_model(),
    })

    if len(providers) == 1:
        logger.warning("No cloud API keys found in .env — all generation will use local Ollama")

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

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra := cfg.get("extra_headers"):
        headers.update(extra)

    client = _get_client()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _ollama_direct(
    messages: list[dict],
    temperature: float,
    format_schema: dict | None = None,
    seed: int | None = None,
) -> str:
    """Call local Ollama directly — used for judge/evaluation tasks (zero cloud cost)."""
    url = f"{_get_ollama_url()}/api/chat"
    payload: dict[str, Any] = {
        "model": _get_ollama_model(),
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


async def _chat_with_fallback(
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    format_schema: dict | None = None,
    seed: int | None = None,
) -> str:
    """Route all tasks through cloud provider rotation → Ollama fallback.

    Both generation and evaluation tasks use the same cloud-first routing.
    Running 4 parallel eval calls on local Ollama causes timeouts;
    cloud providers handle concurrent requests without queuing.
    Ollama is reserved as the final fallback when all cloud keys are exhausted.
    """
    if _get_backend() == "ollama":
        return await _ollama_direct(messages, temperature, format_schema, seed)

    # Try cloud providers in priority order, fall back to Ollama
    providers = _get_generation_providers()
    last_error: Exception | None = None

    for p in providers:
        if p.get("api_url") is None:
            # Ollama fallback
            try:
                result = await _ollama_direct(messages, temperature, format_schema, seed)
                logger.info("Generation fell back to local Ollama (%s)", _get_ollama_model())
                return result
            except Exception as e:
                last_error = e
                continue

        try:
            result = await _cloud_chat(messages, p["model"], temperature, json_mode, provider=p)
            logger.info("Using provider '%s'", p["name"])
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 402, 404, 422, 429):
                logger.warning("Provider '%s' unavailable (HTTP %d), trying next...", p["name"], e.response.status_code)
                last_error = e
                continue
            raise
        except Exception as e:
            logger.warning("Provider '%s' failed: %s", p["name"], e)
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


async def _generate_post_with_instructor(
    messages: list[dict],
    temperature: float,
    seed: int | None = None,
) -> LinkedInPost:
    """Generate a structured LinkedInPost — delegates to shared _generate_with_model."""
    return await _generate_with_model(
        messages=messages,
        response_model=LinkedInPost,
        temperature=temperature,
        seed=seed,
    )


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
        format_schema=format_schema,
        seed=seed,
    )


async def _judge_chat(
    messages: list[dict],
    temperature: float = 0.2,
    format_schema: dict | None = None,
) -> str:
    """Evaluation/judging calls — cloud providers then Ollama fallback."""
    return await _chat_with_fallback(
        messages, temperature,
        json_mode=format_schema is not None,
        format_schema=format_schema,
    )


@mcp_server.tool()
async def research_content(
    title: str,
    content: str,
    source: str = "github",
    source_url: str = "",
    previous_issues: list[str] | None = None,
    temperature: float = 0.3,
) -> dict:
    """Research Agent: extract a structured brief from source content before writing.

    Uses typed, extractive questions per content type so the LLM cannot generalize
    or invent. Returns a ResearchBrief with angle, evidence, contrast, hook recommendation.

    Temperature defaults to 0.3 (research is factual extraction, low creativity needed).
    On retry, caller bumps temperature to encourage different extraction strategy.
    """
    from backend.mcp.resources.prompts import research_prompt, _classify_content_type

    content_type = _classify_content_type(
        source=source, title=title, url=source_url, content=content
    )

    system_prompt, user_prompt = research_prompt(
        content_type=content_type,
        title=title,
        content=content,
        source_url=source_url,
        previous_issues=previous_issues,
    )

    try:
        brief = await _generate_with_model(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=ResearchBrief,
            temperature=temperature,
        )
        return {
            "success": True,
            "content_type": brief.content_type,
            "angle": brief.angle,
            "evidence": brief.evidence,
            "contrast": brief.contrast,
            "limitations": brief.limitations,
            "hook_recommendation": brief.hook_recommendation,
            "hook_draft": brief.hook_draft,
            "confidence": brief.confidence,
            "data_gaps": brief.data_gaps,
        }
    except Exception as e:
        logger.error("Research content failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "content_type": "github_tool",
            "angle": "",
            "evidence": [],
            "contrast": "[not stated]",
            "limitations": ["[not mentioned]"],
            "hook_recommendation": "bold-statement",
            "hook_draft": "",
            "confidence": 0.0,
            "data_gaps": [],
        }


async def _generate_with_model(
    messages: list[dict],
    response_model: type,
    temperature: float,
    seed: int | None = None,
):
    """Structured generation via instructor — cloud key rotation → Ollama fallback.

    Used for research_content and generate_post (both are generation tasks).
    Judge/evaluation tasks never call this function.
    """
    providers = _get_generation_providers()
    last_error: Exception | None = None

    for p in providers:
        try:
            if p.get("api_url") is None:
                # Ollama fallback
                ollama_url = _get_ollama_url()
                raw_client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")
                client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
                model = _get_ollama_model()
                kwargs: dict[str, Any] = {"temperature": temperature}
                if seed is not None:
                    kwargs["seed"] = seed
            else:
                # Cloud provider (OpenAI-compatible)
                raw_client = AsyncOpenAI(
                    base_url=p["api_url"],
                    api_key=p["api_key"],
                    default_headers=p.get("extra_headers") or {},
                )
                client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
                model = p["model"]
                kwargs = {"temperature": temperature}

            result = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                max_retries=2,
                **kwargs,
            )
            logger.info("_generate_with_model used provider '%s'", p["name"])
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "422" in err_str or "402" in err_str or "404" in err_str or "400" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "credit" in err_str:
                logger.warning("Provider '%s' unavailable (rate/credit limit), trying next...", p.get("name"))
            else:
                logger.warning("Provider '%s' failed: %s", p.get("name"), e)
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


@mcp_server.tool()
async def generate_post(
    title: str,
    summary: str,
    source: str,
    source_url: str = "",
    content: str | None = None,
    research_brief: dict | None = None,
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

    # When a research brief is available, the writer's job is expression — not analysis.
    # The brief provides the angle, evidence, and hook draft; the writer builds on it.
    if research_brief and research_brief.get("confidence", 0.0) >= 0.3 and research_brief.get("angle"):
        evidence_str = "\n".join(f"  - {e}" for e in research_brief.get("evidence", []))
        limitations_str = "\n".join(f"  - {l}" for l in research_brief.get("limitations", []))
        data_gaps_str = "\n".join(f"  - {g}" for g in research_brief.get("data_gaps", []))
        brief_ctx = f"""
RESEARCH BRIEF (pre-analyzed — use as your factual foundation, do NOT re-analyze):
  Content type: {research_brief.get("content_type", "")}
  Angle: {research_brief.get("angle", "")}
  Evidence (verbatim from source):
{evidence_str}
  Before/after contrast: {research_brief.get("contrast", "[not stated]")}
  Recommended hook type: {research_brief.get("hook_recommendation", "")}
  Suggested opening: {research_brief.get("hook_draft", "")}
  Known limitations:
{limitations_str}
  Missing data (acknowledge gaps honestly):
{data_gaps_str}

Your job: EXPRESS this angle powerfully. Do NOT change the angle. Do NOT invent evidence.
Build the post structure around what the research brief provides.
If evidence is sparse (marked [no benchmark data]), focus on the contrast and angle instead."""
        user_prompt = f"""Write a LinkedIn post about this content.{brief_ctx}

Title: {title}
Source: {source}{url_line}
Full source content (for additional detail only):
{(content or summary)[:6000]}{trending_ctx}{avoid_ctx}{issues_ctx}{lessons_ctx}{feedback_ctx}

Respond in JSON format matching the output format specified above."""
    else:
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
    # Relevance filtering is lightweight — use cloud providers for speed
    raw = await _ollama_chat(messages=messages, temperature=0.3, format_schema=schema)

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
