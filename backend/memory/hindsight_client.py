"""
Hindsight Memory Client
=======================
Full implementation of Hindsight agent memory using all four memory layers:

  World Facts    — objective info from external sources (user feedback ratings)
  Experience Facts — the agent's own actions ("I generated X using story-hook, scored 8.5")
  Observations   — auto-synthesized patterns by Hindsight (e.g. "story-hook = higher scores")
  Mental Models  — pre-computed reflections for frequent queries (fastest retrieval)

Plus:
  Directives     — hard rules enforced on every reflect() call
  Bank config    — missions, disposition traits, custom extraction instructions

Priority during reflect: Mental Models → Observations → Raw Facts
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
    return ConfigManager.hindsight()


def _get_base_url() -> str:
    return os.environ.get("HINDSIGHT_URL", _get_config()["url"])


def _get_bank_id() -> str:
    return _get_config().get("bank_id", "linkedin_writing_style")


def get_client() -> Hindsight:
    global _client
    if _client is None:
        _client = Hindsight(base_url=_get_base_url())
        logger.info("Initialized Hindsight client at %s", _get_base_url())
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
    """Create and fully configure the memory bank with missions, directives, and mental models."""
    client = get_client()
    bank_id = _get_bank_id()

    # --- Create bank (idempotent) ---
    try:
        await client.acreate_bank(
            bank_id=bank_id,
            # retain_mission: how Hindsight extracts facts from retained content
            retain_mission=(
                "Extract writing style preferences, tone guidance, formatting rules, "
                "hook pattern performance, quality scores, and generation outcomes. "
                "Distinguish between: (1) user feedback on post quality, and "
                "(2) agent actions — what hook/approach was used and what score it achieved."
            ),
            retain_custom_instructions=(
                "When content starts with 'I generated' or 'I wrote', treat it as an "
                "Experience Fact (agent action). When it describes user ratings or feedback, "
                "treat it as a World Fact. Extract numeric scores as structured entities."
            ),
            enable_observations=True,
            observations_mission=(
                "Synthesize patterns across agent experience and user feedback: "
                "which hook patterns correlate with higher scores, "
                "which source types produce better posts, "
                "optimal post length ranges, emoji and hashtag usage that users prefer, "
                "and writing approaches that consistently fail evaluation."
            ),
            reflect_mission=(
                "You are a LinkedIn writing coach with memory of this agent's past performance. "
                "Give specific, actionable writing guidance based on what has worked and failed. "
                "Cite patterns (e.g. 'story-hook averaged 8.5 score') not vague advice. "
                "Focus on style, structure, and tone — never on specific topics or companies."
            ),
            # Disposition: moderately skeptical (don't over-generalize from few examples),
            # low literalism (infer patterns), moderate empathy (user preferences matter)
            disposition_skepticism=4,
            disposition_literalism=2,
            disposition_empathy=4,
        )
        logger.info("Created Hindsight memory bank: %s", bank_id)
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            logger.debug("Memory bank %s already exists", bank_id)
        else:
            logger.warning("Could not create memory bank: %s", e)

    # --- Directives: hard rules enforced on every reflect() ---
    await _setup_directives(bank_id)

    # --- Mental Models: pre-computed reflections for frequent queries ---
    await _setup_mental_models(bank_id)

    # --- Seed style examples (idempotent — skips if already stored) ---
    await _seed_style_examples()


async def _run_sync(fn_name: str, *args, **kwargs):
    """Run a synchronous Hindsight SDK management method in an isolated thread.

    The SDK's management methods (list_directives, create_mental_model, etc.) are
    sync wrappers that call loop.run_until_complete() internally, which fails inside
    an already-running asyncio event loop (FastAPI / our pipeline).

    We create a FRESH Hindsight client inside the thread so the sync call gets its
    own event loop and doesn't corrupt the shared async client's aiohttp session.
    """
    import asyncio
    base_url = _get_base_url()

    def _in_thread():
        from hindsight_client import Hindsight as _Hindsight
        fresh = _Hindsight(base_url=base_url)
        method = getattr(fresh, fn_name)
        return method(*args, **kwargs)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _in_thread)


async def _setup_directives(bank_id: str) -> None:
    """Create directives if they don't exist yet."""
    try:
        response = await _run_sync("list_directives", bank_id=bank_id)
        existing_names = {d.name for d in (response.items or [])}
    except Exception:
        existing_names = set()

    directives = [
        {
            "name": "no_vendor_bias",
            "content": (
                "Never recommend or praise a specific company's product. "
                "Always write as an independent analyst. Every post must include "
                "at least one limitation, open challenge, or comparison to alternatives."
            ),
            "priority": 10,
        },
        {
            "name": "style_over_topics",
            "content": (
                "Guidance must focus on writing STYLE patterns only: tone, structure, "
                "hook type, length, emoji count, hashtag count. "
                "Never say 'posts about Company X perform well' — that is topic bias, not style guidance."
            ),
            "priority": 9,
        },
        {
            "name": "cite_patterns",
            "content": (
                "When giving guidance, cite observed patterns with evidence: "
                "e.g. 'story-hook posts averaged score 8.5 vs question-hook at 7.2'. "
                "Do not give vague advice like 'be engaging'."
            ),
            "priority": 8,
        },
    ]

    for d in directives:
        if d["name"] not in existing_names:
            try:
                await _run_sync(
                    "create_directive",
                    bank_id=bank_id,
                    name=d["name"],
                    content=d["content"],
                    priority=d["priority"],
                    tags=["core"],
                )
                logger.info("Created directive: %s", d["name"])
            except Exception as e:
                logger.debug("Directive %s may already exist: %s", d["name"], e)


async def _setup_mental_models(bank_id: str) -> None:
    """Create (or recreate) mental models aligned with the data we actually have.

    Two models were replaced:
    - "Hook Pattern Performance" → "Post Structure Templates" (queries style_example memories we have)
    - "Source Type Quality" → "Character Count & Density Patterns" (queries char_count data we have)
    Two are kept but refreshed:
    - "User Style Preferences" (feedback ratings provide signal)
    - "Quality Failure Patterns" (recreated clean — previous content was garbled tool-call JSON)
    """
    try:
        response = await _run_sync("list_mental_models", bank_id=bank_id)
        existing = {m.name: m for m in (response.items or [])}
    except Exception:
        existing = {}

    # Delete outdated / garbled models so they can be recreated clean
    models_to_delete = {"Hook Pattern Performance", "Source Type Quality", "Quality Failure Patterns"}
    for name in models_to_delete:
        if name in existing:
            try:
                await _run_sync("delete_mental_model", bank_id=bank_id, name=name)
                logger.info("Deleted stale mental model: %s", name)
            except Exception as e:
                logger.debug("Could not delete mental model %s: %s", name, e)

    # Refresh existing names after deletions
    try:
        response = await _run_sync("list_mental_models", bank_id=bank_id)
        existing_names = {m.name for m in (response.items or [])}
    except Exception:
        existing_names = set()

    mental_models = [
        {
            "name": "Post Structure Templates",
            "source_query": (
                "What formatting structures and opening patterns appear in high-quality example posts? "
                "Include: use of → arrows for sequential steps, > callout lines for key specs, "
                "numbered 'Here's how it works' sections, single-sentence emphasis lines, "
                "bold-statement openers with specific numbers, curiosity-gap openers with surprising results, "
                "and any structural patterns that earned high evaluation scores."
            ),
            "tags": ["structure", "templates", "style"],
        },
        {
            "name": "User Style Preferences",
            "source_query": (
                "What writing style does the user prefer for LinkedIn posts? "
                "Include: preferred tone (analytical vs conversational vs bold), "
                "optimal post length, emoji usage and placement, hashtag count, "
                "and any explicit feedback given via ratings. "
                "What style elements appear in posts the user rated positively?"
            ),
            "tags": ["style", "preferences"],
        },
        {
            "name": "Quality Failure Patterns",
            "source_query": (
                "What writing patterns, approaches, or content types consistently "
                "fail evaluation? What causes low bias scores, hallucination failures, "
                "or guardrail violations? What should be avoided in future posts?"
            ),
            "tags": ["failures", "quality"],
        },
        {
            "name": "Character Count & Density Patterns",
            "source_query": (
                "What post lengths and character counts correlate with passing evaluation vs failing? "
                "What is the optimal character range? What paragraph density (sentences per paragraph) "
                "appears in high-quality example posts? Are shorter or longer posts rated better?"
            ),
            "tags": ["length", "density", "quality"],
        },
    ]

    for m in mental_models:
        if m["name"] not in existing_names:
            try:
                await _run_sync(
                    "create_mental_model",
                    bank_id=bank_id,
                    name=m["name"],
                    source_query=m["source_query"],
                    tags=m["tags"],
                    trigger={"refresh_after_consolidation": True},
                )
                logger.info("Created mental model: %s", m["name"])
            except Exception as e:
                logger.debug("Mental model %s may already exist: %s", m["name"], e)

    # Explicitly refresh all models now so they have content immediately
    await _refresh_all_mental_models(bank_id)


async def _refresh_all_mental_models(bank_id: str) -> None:
    """Refresh all mental models so they have content from current memories."""
    try:
        response = await _run_sync("list_mental_models", bank_id=bank_id)
        for m in (response.items or []):
            try:
                await _run_sync("refresh_mental_model", bank_id=bank_id, name=m.name)
                logger.info("Refreshed mental model: %s", m.name)
            except Exception as e:
                logger.debug("Could not refresh mental model %s: %s", m.name, e)
    except Exception as e:
        logger.warning("Could not refresh mental models: %s", e)


_STYLE_EXAMPLES = [
    {
        "description": "LMCache — infrastructure/optimization problem-solution template",
        "source_type": "github",
        "post": """Your LLM inference is burning 50% of its compute on work it has already done.

If you are running RAG or Multi-Turn Chat, you are likely recomputing the KV Cache for the same documents over and over again.

I found the open-source library that solves this. It's called LMCache. It makes the KV cache Persistent and Shareable across different engine instances (vLLM, SGLang).

The "Cheat Code" for AI Infrastructure => Instead of the cache dying when a request finishes, LMCache offloads it to a shared layer (CPU/Disk/Network).

This unlocks architecture patterns that were previously impossible:
1. Instant RAG
Process a 100-page PDF once. Store the KV cache. Now any user query against that doc starts instantly (Zero Time-To-First-Token).

2. Disaggregated Serving
Run heavy "Prefill" on H100s. Stream the cache to cheaper L4s for "Decoding."

3. Context Sharing
Multiple users asking about the same context? Compute it once, serve everyone.

🚀 15x throughput gain in multi-round QA workloads.
⚡ 3-10x reduction in Time-To-First-Token (TTFT).

It integrates directly with vLLM and SGLang. Stop letting your GPUs do the same homework twice.

#MachineLearning #AIInfrastructure #LLM #MLOps #GenerativeAI""",
    },
]


async def _seed_style_examples() -> None:
    """Store curated style examples in Hindsight (idempotent — skips if already present)."""
    for ex in _STYLE_EXAMPLES:
        try:
            await store_style_example(
                example_post=ex["post"],
                description=ex["description"],
                source_type=ex["source_type"],
            )
        except Exception as e:
            logger.debug("Could not seed style example '%s': %s", ex["description"], e)


async def store_generation_experience(
    post_id: str,
    source: str,
    hook_pattern: str,
    overall_score: float,
    metric_scores: dict,
    passed: bool,
    attempt_number: int,
    char_count: int,
    issues: list[str],
) -> bool:
    """
    Store Experience Facts — the agent's own actions and outcomes.

    Written in first-person so Hindsight correctly classifies these as
    Experience Facts (vs World Facts from user feedback).
    These teach the agent: which approaches work, which fail, and why.
    """
    client = get_client()

    outcome = "passed evaluation" if passed else "failed evaluation"
    attempt_str = f"on attempt {attempt_number}" if attempt_number > 1 else "on the first attempt"
    scores_str = ", ".join(f"{k}={v:.1f}" for k, v in metric_scores.items())
    issues_str = f" Issues: {'; '.join(issues[:3])}." if issues else ""

    content = (
        f"I generated a LinkedIn post from a {source} source using the {hook_pattern} hook pattern. "
        f"The post was {char_count} characters long and {outcome} {attempt_str}. "
        f"Scores: overall={overall_score:.1f}, {scores_str}.{issues_str}"
    )

    tags = [
        f"hook:{hook_pattern}",
        f"source:{source}",
        "experience",
        "passed" if passed else "failed",
    ]
    if attempt_number > 1:
        tags.append("needed_retry")

    metadata = {
        "post_id": post_id,
        "source": source,
        "hook_pattern": hook_pattern,
        "overall_score": str(round(overall_score, 2)),
        "passed": str(passed).lower(),
        "attempt_number": str(attempt_number),
        "char_count": str(char_count),
    }

    # Quality tier for pattern matching (easier for Hindsight to cluster than raw floats)
    quality_tier = "high" if overall_score >= 8.0 else ("medium" if overall_score >= 7.0 else "low")
    char_bucket = (
        "short" if char_count < 1200 else
        "optimal" if char_count <= 2100 else
        "long"
    )

    # Entities use SDK format: {"text": value, "type": entity_type}
    # This builds the knowledge graph that Mental Models query over.
    entities = [
        {"text": hook_pattern, "type": "hook_pattern"},
        {"text": source, "type": "source_type"},
        {"text": quality_tier, "type": "quality_tier"},
        {"text": char_bucket, "type": "post_length"},
        {"text": "passed" if passed else "failed", "type": "outcome"},
    ]
    # Add failing metric entities so Hindsight learns which metrics each hook/source fails on
    for metric, score in metric_scores.items():
        threshold = {"answer_relevancy": 7.0, "faithfulness": 7.0, "hallucination": 8.0,
                     "bias": 5.5, "toxicity": 9.0, "linkedin_quality": 7.0}.get(metric, 7.0)
        if score < threshold:
            entities.append({"text": metric, "type": "failed_metric"})

    try:
        result = await client.aretain(
            bank_id=_get_bank_id(),
            content=content,
            context=f"Agent self-log: post generation outcome for source={source}",
            metadata=metadata,
            tags=tags,
            entities=entities,
            timestamp=datetime.now(),
        )
        logger.info(
            "Stored generation experience: hook=%s source=%s score=%.1f passed=%s",
            hook_pattern, source, overall_score, passed,
        )
        return result.success
    except Exception as e:
        logger.error("Failed to store generation experience: %s", e)
        return False


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

    # Put the user's feedback as primary content — Hindsight extracts from content
    # Post preview goes in context (background info, not the main fact to learn)
    sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "mixed")
    parts = [
        f"User gave a {rating}/5 star rating ({sentiment}) on a LinkedIn post.",
    ]
    if feedback_text:
        parts.append(f"User's exact feedback: \"{feedback_text}\"")
    if is_good_example:
        parts.append("User explicitly marked this as a good example to follow.")
    if rating >= 4 and not feedback_text:
        parts.append("User liked this post — style and format are good examples to follow.")
    elif rating <= 2 and not feedback_text:
        parts.append("User disliked this post — style and format should be avoided.")

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

    # Context: structural attributes only — NOT the topic/title.
    # Including the title teaches "this topic = good/bad" rather than "this style = good/bad",
    # causing Hindsight's reflect() to return topic-biased guidance (e.g. "Google posts work well")
    # instead of writing-style guidance (e.g. "use analytical language with clear structure").
    import re as _re
    char_count = len(post_text)
    hashtag_count = len(_re.findall(r'(?<!\w)#[A-Za-z]\w*', post_text))
    context = (
        f"Source type: {source} | "
        f"Post length: {char_count} chars | "
        f"Hashtag count: {hashtag_count}"
    )

    # Entities for feedback World Facts — enables the knowledge graph to link
    # ratings to source types and sentiment tiers for pattern extraction.
    rating_tier = "liked" if rating >= 4 else ("disliked" if rating <= 2 else "neutral")
    entities = [
        {"text": source, "type": "source_type"},
        {"text": rating_tier, "type": "user_sentiment"},
        {"text": str(rating), "type": "star_rating"},
    ]
    if is_good_example:
        entities.append({"text": "good_example", "type": "user_label"})

    try:
        result = await client.aretain(
            bank_id=_get_bank_id(),
            content=content,
            context=context,
            metadata=metadata,
            tags=tags,
            entities=entities,
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


async def store_failed_experience(
    source: str,
    hook_pattern: str,
    overall_score: float,
    metric_scores: dict,
    attempt_number: int,
    char_count: int,
    issues: list[str],
) -> bool:
    """
    Store Experience Facts for REJECTED posts.

    Agents must learn from failures, not just successes.
    Storing failed attempts teaches Hindsight which hook/source combinations
    consistently fail which metrics — enabling pattern-interrupt before the
    next generation attempt uses the same failing approach.
    """
    client = get_client()

    scores_str = ", ".join(f"{k}={v:.1f}" for k, v in metric_scores.items())
    issues_str = f" Issues: {'; '.join(issues[:3])}." if issues else ""
    attempt_str = f"on attempt {attempt_number}" if attempt_number > 1 else "on the first attempt"

    content = (
        f"I generated a LinkedIn post from a {source} source using the {hook_pattern} hook pattern. "
        f"The post was {char_count} characters long and failed evaluation {attempt_str}. "
        f"Scores: overall={overall_score:.1f}, {scores_str}.{issues_str} "
        f"This approach should be avoided or adjusted."
    )

    quality_tier = "low" if overall_score < 7.0 else "medium"
    entities = [
        {"text": hook_pattern, "type": "hook_pattern"},
        {"text": source, "type": "source_type"},
        {"text": quality_tier, "type": "quality_tier"},
        {"text": "failed", "type": "outcome"},
    ]
    for metric, score in metric_scores.items():
        threshold = {"answer_relevancy": 7.0, "faithfulness": 7.0, "hallucination": 8.0,
                     "bias": 5.5, "toxicity": 9.0, "linkedin_quality": 7.0}.get(metric, 7.0)
        if score < threshold:
            entities.append({"text": metric, "type": "failed_metric"})

    try:
        result = await client.aretain(
            bank_id=_get_bank_id(),
            content=content,
            context=f"Agent self-log: rejected post from {source} — failure case",
            metadata={
                "source": source,
                "hook_pattern": hook_pattern,
                "overall_score": str(round(overall_score, 2)),
                "passed": "false",
                "attempt_number": str(attempt_number),
            },
            tags=[f"hook:{hook_pattern}", f"source:{source}", "experience", "failed"],
            entities=entities,
            timestamp=datetime.now(),
        )
        logger.info(
            "Stored failed experience: hook=%s source=%s score=%.1f",
            hook_pattern, source, overall_score,
        )
        return result.success
    except Exception as e:
        logger.error("Failed to store failed experience in Hindsight: %s", e)
        return False


async def get_style_guidance(
    topic: str,
    source: str,
    content_summary: str | None = None,  # retained for caller compat, not used in query
) -> str:
    """
    Get autonomous style guidance from learned preferences.

    Uses official SDK: client.areflect(bank_id, query) -> ReflectResponse.text
    """
    client = get_client()

    # Query asks for writing STYLE patterns only — not topic recommendations.
    # We deliberately exclude the specific topic/summary from the query so Hindsight
    # does not retrieve topic-similar memories and respond "posts about X work well".
    # The `tags` filter scopes retrieval to feedback-tagged memories only.
    query = (
        "Based on all stored user feedback, what writing style patterns should I follow "
        "for LinkedIn posts? Focus only on: tone (analytical vs promotional), structure "
        "(hook type, paragraph length), length, emoji usage, hashtag count, and language "
        "choices to avoid. Do NOT mention specific topics, companies, or products."
    )

    try:
        hs_config = _get_config()
        result = await client.areflect(
            bank_id=_get_bank_id(),
            query=query,
            budget=hs_config.get("reflect_budget", "mid"),
            context=f"Source type being written about: {source}",
            tags=["feedback"],
            tags_match="any",
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


async def store_style_example(
    example_post: str,
    description: str,
    source_type: str = "github",
) -> bool:
    """Store a high-quality example post as a World Fact in Hindsight.

    These are external reference posts (not generated by the agent) that demonstrate
    the desired formatting style. Hindsight uses them to guide future generation.

    Idempotent: skips storage if a style_example with the same description already exists.
    """
    client = get_client()

    # Dedup check — avoid re-fragmenting the same example on every startup
    try:
        existing = client.list_memories(bank_id=_get_bank_id(), tags=["style_example"])
        if any(
            m.get("metadata", {}).get("description") == description
            for m in (existing.items or [])
        ):
            logger.debug("Style example already stored, skipping: %s", description)
            return True
    except Exception:
        pass  # If check fails, proceed to store

    content = (
        f"HIGH-QUALITY EXAMPLE POST ({description}):\n\n"
        f"{example_post}\n\n"
        f"Key style elements from this example: "
        f"short punchy sentences, → arrows for sequential steps, "
        f"> callout lines for key specs/features, specific concrete numbers, "
        f"single-sentence lines for emphasis."
    )
    try:
        result = await client.aretain(
            bank_id=_get_bank_id(),
            content=content,
            context=f"External example demonstrating ideal LinkedIn post formatting for {source_type} content",
            metadata={"type": "style_example", "source_type": source_type, "description": description},
            tags=["style_example", "good_example", "feedback", f"source:{source_type}"],
            entities=[
                {"text": "style_example", "type": "user_label"},
                {"text": source_type, "type": "source_type"},
                {"text": "high_quality", "type": "quality_tier"},
            ],
            timestamp=datetime.now(),
        )
        logger.info("Stored style example in Hindsight: %s, success=%s", description, result.success)
        return result.success
    except Exception as e:
        logger.error("Failed to store style example in Hindsight: %s", e)
        return False


async def check_health() -> dict:
    """Check if Hindsight server is reachable."""
    url = _get_base_url()
    try:
        async with httpx.AsyncClient() as http:
            response = await http.get(f"{url}/health", timeout=5.0)
            return {"status": "healthy", "code": response.status_code}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
