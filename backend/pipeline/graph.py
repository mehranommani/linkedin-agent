"""
LangGraph Pipeline
==================
Autonomous pipeline that fetches, filters, generates, evaluates, and saves posts.
Nodes are thin orchestrators that call MCP tools for all logic.

Flow:
  START → fetch_content → filter_and_score → pick_next_item
            ↓ (done) → finalize → END
            ↓ (process)
          extract_article → generate_post → evaluate_post
            ↓ (accept) → check_duplicate → save_post → pick_next_item
            ↓ (retry)  → generate_post (with issue feedback)
            ↓ (reject) → pick_next_item
"""
from __future__ import annotations

import asyncio
import json
import uuid
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END

from backend.pipeline.state import PipelineState
from backend.pipeline.edges import quality_gate_decision, should_continue, research_gate_decision
from backend.config import ConfigManager

# MCP tool functions (called directly, not via MCP protocol in-process)
from backend.mcp.tools.sources import fetch_source
from backend.mcp.tools.database import (
    insert_post, check_url_exists, log_pipeline_run,
    get_active_sources,
)
from backend.mcp.tools.llm import generate_post as llm_generate_post, batch_judge_relevance, research_content as llm_research_content
from backend.mcp.tools.evaluator import check_duplicate as eval_check_duplicate
from backend.mcp.resources.prompts import lessons_learned_prompt

logger = logging.getLogger(__name__)

# Deterministic keyword pre-filter — items must match at least one keyword in title+summary
# to be considered for LLM relevance scoring. Mirrors the original ContentScout filter.

async def _log(state: PipelineState, msg: str) -> list[str]:
    """Append a log message, broadcast via WebSocket if available, and return updated logs."""
    logs = list(state.get("logs", []))
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    logs.append(entry)
    broadcast_fn = state.get("broadcast_fn")
    if broadcast_fn is not None:
        try:
            await broadcast_fn({"type": "log", "message": entry, "timestamp": datetime.now().isoformat()})
        except Exception:
            pass  # Never fail the pipeline because of a broadcast error
    return logs


# ============================================================
# PIPELINE NODES
# ============================================================

async def fetch_content_node(state: PipelineState) -> dict:
    """Fetch content from all active sources concurrently."""
    config = state.get("run_config", {})
    limit = config.get("limit_per_source", 15)
    logs = await _log(state, "Fetching content from active sources...")

    # Get active source configs from DB
    sources_result = get_active_sources()
    active_sources = sources_result.get("sources", [])

    # Apply source type filter if the run config specifies sources
    source_types_filter = config.get("sources")  # list of source type names, e.g. ["github", "reddit"]
    if source_types_filter:
        filter_set = set(source_types_filter)
        active_sources = [s for s in active_sources if s.get("source_type") in filter_set]
        logger.info(f"Source filter applied: {filter_set} → {len(active_sources)} sources")
    else:
        logger.info(f"Active sources: {len(active_sources)}")

    # Group by source_type
    by_type: dict[str, list[dict]] = {}
    for src in active_sources:
        st = src["source_type"]
        if st not in by_type:
            by_type[st] = []
        by_type[st].append(src)

    # Fetch from each source type concurrently
    all_items = []

    async def _fetch_one(source_type: str, params: dict):
        try:
            result = await fetch_source(source_type=source_type, params=params, limit=limit)
            items = result.get("items", [])
            logger.info(f"Fetched {len(items)} items from {source_type}")
            return items
        except Exception as e:
            logger.error(f"Error fetching {source_type}: {e}")
            return []

    tasks = []
    for source_type, configs in by_type.items():
        for src_config in configs:
            params = src_config.get("params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            # Pass source ID for error tracking (used by RSS, etc.)
            params["_source_id"] = src_config.get("id", "")
            tasks.append(_fetch_one(source_type, params))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            all_items.extend(result)

    logs = await _log({**state, "logs": logs}, f"Fetched {len(all_items)} items from {len(by_type)} source types")
    return {"raw_items": all_items, "logs": logs, "step": "fetch_content"}


async def filter_and_score_node(state: PipelineState) -> dict:
    """Filter items by relevance and check for duplicates in DB."""
    items = state.get("raw_items", [])
    logs = await _log(state, f"Filtering {len(items)} items by relevance...")

    if not items:
        return {"filtered_items": [], "items_remaining": 0, "logs": logs, "step": "filter"}

    # Keep batch small enough for local Ollama judge to finish within timeout.
    # 30 items per batch is safe for qwen2.5:7b within 120s.
    batch_limit = 30

    # Batch relevance check via LLM
    batch_input = [
        {"title": item.get("title", ""), "summary": item.get("summary", "")[:150]}
        for item in items[:batch_limit]
    ]

    judgments_result = await batch_judge_relevance(items=batch_input)
    judgments = judgments_result.get("results", [])

    # Build lookup by index
    judgment_map = {j.get("index", 0): j for j in judgments}

    # Load existing post titles once for title-similarity dedup
    from backend.database import fetch_all as _fetch_all
    from backend.mcp.tools.evaluator import _jaccard_similarity
    existing_titles = [r[0] for r in _fetch_all("SELECT title FROM posts ORDER BY created_at DESC LIMIT 500") if r[0]]

    def _repo_name(title: str) -> str:
        """For 'org/repo' style titles (GitHub), return just the repo name portion."""
        return title.split("/")[-1] if "/" in title else title

    def _is_title_duplicate(candidate: str, existing_list: list[str]) -> bool:
        """Return True if candidate is semantically too similar to any existing title.

        Uses two strategies:
        - Exact repo-name match for GitHub 'org/repo' style titles
        - Trigram Jaccard >= 0.4 for all other titles
        """
        candidate_repo = _repo_name(candidate).lower().strip()
        for existing in existing_list:
            # Strategy 1: repo-name substring match (catches forks / org renames)
            existing_repo = _repo_name(existing).lower().strip()
            if len(candidate_repo) >= 5 and (
                candidate_repo in existing_repo or existing_repo in candidate_repo
            ):
                return True
            # Strategy 2: trigram Jaccard on full title
            if _jaccard_similarity(candidate, existing) >= 0.4:
                return True
        return False

    filtered = []
    skipped_irrelevant = 0
    skipped_existing = 0

    for i, item in enumerate(items[:batch_limit]):
        judgment = judgment_map.get(i + 1, {"is_relevant": True, "score": 7.0})

        if not judgment.get("is_relevant", False):
            skipped_irrelevant += 1
            continue

        # Check if URL already in DB (exact match)
        url = item.get("url", "")
        if url:
            exists_result = check_url_exists(url)
            if exists_result.get("exists", False):
                skipped_existing += 1
                continue

        # Check title similarity against existing posts (catches forks/derivatives)
        candidate_title = item.get("title", "")
        if candidate_title and _is_title_duplicate(candidate_title, existing_titles):
            logger.debug("Skipping '%s' — similar title already posted", candidate_title)
            skipped_existing += 1
            continue

        item["relevance_score"] = judgment.get("score", 7.0)
        filtered.append(item)

    # Sort by relevance score descending
    filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    logs = await _log(
        {**state, "logs": logs},
        f"Filtered: {len(filtered)} passed, {skipped_irrelevant} irrelevant, {skipped_existing} existing"
    )

    return {
        "filtered_items": filtered,
        "items_remaining": len(filtered),
        "current_item_index": 0,
        "logs": logs,
        "step": "filter",
    }


async def pick_next_item_node(state: PipelineState) -> dict:
    """Pick the next item to process from filtered items."""
    filtered = state.get("filtered_items", [])
    idx = state.get("current_item_index", 0)
    logs = state.get("logs", [])

    if idx >= len(filtered):
        return {
            "current_item": None,
            "items_remaining": 0,
            "logs": logs,
            "step": "pick_next",
        }

    item = filtered[idx]
    logs = await _log(state, f"Processing [{idx+1}/{len(filtered)}]: {item.get('title', '')[:60]}...")

    return {
        "current_item": item,
        "current_item_index": idx + 1,
        "items_remaining": len(filtered) - idx - 1,
        "generation_attempts": 0,
        "current_post": None,
        "extracted_content": "",
        "research_brief": None,
        "research_attempts": 0,
        "research_evaluation": None,
        "research_failed": False,
        "logs": logs,
        "step": "pick_next",
    }


async def extract_article_node(state: PipelineState) -> dict:
    """Extract article content from the current item's URL."""
    item = state.get("current_item")
    if not item:
        return state

    logs = await _log(state, f"Extracting article content...")
    url = item.get("url", "")

    content = ""
    if url:
        try:
            from backend.mcp.tools.sources import extract_article
            result = await extract_article(url)
            if result.get("success"):
                content = result.get("content", "")
        except Exception as e:
            logger.warning(f"Article extraction failed: {e}")

    return {"extracted_content": content, "logs": logs, "step": "extract"}


async def research_content_node(state: PipelineState) -> dict:
    """Research Agent: extract a structured brief from source content.

    Runs typed, extractive questions per content type — no open-ended prompts.
    Produces a ResearchBrief that the Writer uses as its factual foundation.
    Temperature starts low (0.3) and escalates on retry.
    """
    item = state.get("current_item")
    if not item:
        return state

    attempt = state.get("research_attempts", 0) + 1
    logs = await _log(state, f"Researching content (attempt {attempt})...")

    # Temperature escalates on retry to encourage different extraction strategy
    temps = [0.3, 0.5, 0.7]
    temp = temps[min(attempt - 1, len(temps) - 1)]

    # Inject issues from previous failed attempt
    previous_issues: list[str] = []
    prev_eval = state.get("research_evaluation")
    if prev_eval and not prev_eval.get("passed", False):
        previous_issues = prev_eval.get("issues", [])

    content = state.get("extracted_content") or item.get("summary", "")

    result = await llm_research_content(
        title=item.get("title", ""),
        content=content,
        source=item.get("source", "github"),
        source_url=item.get("url", ""),
        previous_issues=previous_issues or None,
        temperature=temp,
    )

    ct = result.get("content_type", "unknown")
    conf = result.get("confidence", 0.0)
    angle_preview = result.get("angle", "")[:60]
    logs = await _log(
        {**state, "logs": logs},
        f"Research: type={ct} confidence={conf:.2f} angle='{angle_preview}...'"
    )

    return {
        "research_brief": result if result.get("success") else None,
        "research_attempts": attempt,
        "research_evaluation": None,  # reset for evaluate step
        "logs": logs,
        "step": "research",
    }


async def evaluate_research_node(state: PipelineState) -> dict:
    """Research Evaluator: validate the research brief quality.

    Two-stage: programmatic guardrails (instant) + LLM judge (1 call).
    Sets research_failed=True and clears brief if retries exhausted.
    """
    from backend.evaluation.evaluator import evaluate_research_brief

    brief = state.get("research_brief")
    attempts = state.get("research_attempts", 0)

    if not brief:
        # No brief — go straight to fallback
        logs = await _log(state, "No research brief produced — using degraded generation mode")
        return {
            "research_evaluation": {"passed": False, "overall": 0.0, "issues": ["no brief produced"]},
            "research_failed": True,
            "logs": logs,
            "step": "evaluate_research",
        }

    logs = await _log(state, "Evaluating research brief (programmatic + LLM)...")

    evaluation = await evaluate_research_brief(brief)

    passed = evaluation.get("passed", False)
    overall = evaluation.get("overall", 0.0)
    stage = evaluation.get("stage", "")
    status = "PASSED" if passed else f"FAILED (overall={overall:.1f}, stage={stage})"

    logs = await _log(
        {**state, "logs": logs},
        f"Research eval: {status} | "
        f"specificity={evaluation.get('specificity', 0):.1f} "
        f"hook_viability={evaluation.get('hook_viability', 0):.1f} "
        f"evidence_grounding={evaluation.get('evidence_grounding', 0):.1f}"
    )

    # If we've exhausted retries and still failing → fallback mode
    research_failed = not passed and attempts >= 2

    if research_failed:
        logs = await _log(
            {**state, "logs": logs},
            "Research retries exhausted — generating without brief (degraded mode)"
        )

    return {
        "research_evaluation": evaluation,
        "research_failed": research_failed,
        "research_brief": brief if not research_failed else None,
        "logs": logs,
        "step": "evaluate_research",
    }


async def generate_post_node(state: PipelineState) -> dict:
    """Generate a LinkedIn post for the current item."""
    from backend.memory.hindsight_client import get_style_guidance

    item = state.get("current_item")
    if not item:
        return state

    attempt = state.get("generation_attempts", 0) + 1
    logs = await _log(state, f"Generating post (attempt {attempt})...")

    # Get config
    gen_config = ConfigManager.generation()
    temps = gen_config.get("temperatures", [0.7, 0.85, 1.0])
    temp = temps[min(attempt - 1, len(temps) - 1)]

    # Get lessons from previous runs
    lessons = lessons_learned_prompt()

    # Get autonomous style guidance from Hindsight (learned from all feedback)
    # Uses 4-strategy retrieval + LLM reasoning for topic-aware guidance
    feedback_patterns = await get_style_guidance(
        topic=item.get("title", ""),
        source=item.get("source", ""),
        content_summary=item.get("summary", "")
    )

    # Fetch recently used hook patterns to force variety across posts
    from backend.database import fetch_all as _fetch_all_hooks
    recent_hooks = [
        r[0] for r in _fetch_all_hooks(
            "SELECT hook_pattern_used FROM posts "
            "WHERE hook_pattern_used IS NOT NULL AND hook_pattern_used != '' "
            "ORDER BY created_at DESC LIMIT 8"
        ) if r[0]
    ]

    # Build feedback from the previous LangGraph evaluation cycle (empty on first attempt)
    previous_feedback = ""
    current_evaluation = state.get("current_evaluation")
    if current_evaluation:
        issues = current_evaluation.get("issues", [])
        if issues:
            previous_feedback = "Issues from last attempt: " + "; ".join(issues[:5])

    url = item.get("url", "")
    banned = ConfigManager.banned_phrases()
    constraints_parts = ["No markdown code blocks, star bullets (* item), or badge syntax ([![)."]
    if banned:
        constraints_parts.append(f"Never use these phrases: {', '.join(banned[:10])}.")
    if recent_hooks:
        constraints_parts.append(f"Avoid hook patterns already used recently: {', '.join(recent_hooks[:5])}.")
    constraints = " ".join(constraints_parts)

    research_brief = state.get("research_brief") if not state.get("research_failed") else None
    brief = research_brief or {}
    angle = brief.get("angle") or item.get("title", "")
    evidence_list = brief.get("evidence") or []
    source_excerpt = (state.get("extracted_content") or item.get("summary", ""))[:2000]
    style = ((feedback_patterns or "") + ("\n\n" + lessons if lessons else "")).strip()

    from backend.dspy_modules.post_generator import get_post_generator
    import dspy as _dspy

    post_text = ""
    hook_used = ""
    logger.info("DSPy generating post (attempt=%d, angle=%s)", attempt, angle[:80])
    try:
        pred = await _dspy.asyncify(get_post_generator())(
            angle=angle,
            evidence=evidence_list,
            repo_url=url,
            source_excerpt=source_excerpt,
            style_context=style,
            previous_feedback=previous_feedback,
            constraints=constraints,
        )
        post_text = pred.post or ""
        logger.info("DSPy generation succeeded: %d chars", len(post_text))
    except Exception as exc:
        logger.warning("DSPy post generation failed (%s) — using direct LLM fallback", exc)
        seed = hash(url + str(attempt)) % (2 ** 31)
        fallback = await llm_generate_post(
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            source=item.get("source", ""),
            source_url=url,
            content=state.get("extracted_content"),
            research_brief=research_brief,
            trending_topics=[t.get("name", "") for t in state.get("trending_topics", [])],
            avoid_phrases=banned,
            previous_issues=[previous_feedback] if previous_feedback else [],
            lessons_learned=lessons,
            feedback_patterns=feedback_patterns or None,
            avoid_hook_patterns=recent_hooks or None,
            temperature=temp,
            seed=seed,
        )
        post_text = fallback.get("post_text", "")
        hook_used = fallback.get("hook_pattern_used", "")

    # Ensure source URL is present
    if url and "http" not in post_text:
        post_text = post_text.rstrip() + f"\n\n{url}"

    # Apply LinkedIn Unicode bold formatting (hook line + key metrics)
    from backend.utils.linkedin_format import apply_linkedin_bold
    post_text = apply_linkedin_bold(post_text)

    # Normalize hook_pattern_used to a valid pattern name.
    # The LLM sometimes returns the post's first line instead of the pattern name,
    # especially after apply_linkedin_bold() has already formatted the text.
    _valid_hooks = {"discovery", "pattern-interrupt", "curiosity-gap", "bold-statement",
                    "story-hook", "contrarian", "question-hook"}
    if hook_used.lower().strip() not in _valid_hooks:
        hook_used = "unknown"

    post_data = {
        "post_text": post_text,
        "hook": hook_used,
        "body": "",
        "takeaway": "",
        "hashtags": [],
        "url": url,
        "char_count": len(post_text),
        "issues": [],
        "quality_score": 0,
        "faithfulness_score": 0,
        "passed_quality_gate": False,
        "hook_pattern_used": hook_used,
        "seo_keywords_used": [],
    }

    logs = await _log(
        {**state, "logs": logs},
        f"Generated post: {post_data['char_count']} chars (attempt={attempt}, hook={hook_used[:60]})"
    )

    return {
        "current_post": post_data,
        "generation_attempts": attempt,
        "logs": logs,
        "step": "generate",
    }


async def evaluate_post_node(state: PipelineState) -> dict:
    """Evaluate the generated post using the 6-metric evaluation engine."""
    from backend.evaluation.evaluator import run_full_evaluation

    post = state.get("current_post")
    item = state.get("current_item")
    if not post or not item:
        return state

    logs = await _log(state, "Evaluating post (6-metric v2 engine)...")
    post_text = post.get("post_text", "")
    source_content = state.get("extracted_content") or item.get("summary", "")
    source_title = item.get("title", "")

    evaluation = await run_full_evaluation(
        post_text=post_text,
        source_content=source_content,
        source_title=source_title,
        pipeline_run_id=state.get("run_id", ""),
    )

    # Map v2 scores back to post data for backward compatibility
    post["quality_score"] = evaluation.get("linkedin_quality", 0)
    post["faithfulness_score"] = evaluation.get("faithfulness", 0)
    post["issues"] = evaluation.get("issues", [])
    post["passed_quality_gate"] = evaluation.get("passed", False)
    post["overall_score"] = evaluation.get("overall_score", 0)

    passed = evaluation.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    logs = await _log(
        {**state, "logs": logs},
        f"Evaluation: overall={evaluation.get('overall_score', 0):.1f} "
        f"[rel={evaluation.get('answer_relevancy', 0):.1f} "
        f"faith={evaluation.get('faithfulness', 0):.1f} "
        f"halluc={evaluation.get('hallucination', 0):.1f} "
        f"bias={evaluation.get('bias', 0):.1f} "
        f"tox={evaluation.get('toxicity', 0):.1f} "
        f"lq={evaluation.get('linkedin_quality', 0):.1f}] → {status}"
    )

    return {
        "current_post": post,
        "current_evaluation": evaluation,
        "logs": logs,
        "step": "evaluate",
    }


async def check_duplicate_node(state: PipelineState) -> dict:
    """Check if the accepted post is a duplicate of existing content."""
    post = state.get("current_post")
    if not post:
        return state

    logs = await _log(state, "Checking for duplicates...")
    result = eval_check_duplicate(post_text=post.get("post_text", ""))

    if result.get("is_duplicate", False):
        logs = await _log(
            {**state, "logs": logs},
            f"DUPLICATE detected (similarity={result['similarity']:.2f}). Rejecting."
        )
        rejected = list(state.get("rejected_items", []))
        item = state.get("current_item", {})
        rejected.append({**item, "reason": "duplicate", "similarity": result["similarity"]})
        return {"rejected_items": rejected, "logs": logs, "step": "dedup_reject"}

    logs = await _log({**state, "logs": logs}, "No duplicate found. Saving post.")
    return {"logs": logs, "step": "dedup_pass"}


async def save_post_node(state: PipelineState) -> dict:
    """Save the accepted post to the database."""
    from backend.evaluation.evaluator import _persist_evaluation

    post = state.get("current_post")
    item = state.get("current_item")
    if not post or not item:
        return state

    logs = await _log(state, "Saving post to database...")

    result = insert_post(
        title=item.get("title", ""),
        original_url=item.get("url", ""),
        source=item.get("source", ""),
        post_text=post.get("post_text", ""),
        relevance_score=item.get("relevance_score", 0),
        quality_score=post.get("quality_score", 0),
        faithfulness_score=post.get("faithfulness_score", 0),
        trending_boost=item.get("trending_boost", 0),
        source_summary=item.get("summary"),
        image_url=item.get("image_url"),
        extracted_content=state.get("extracted_content"),
        hashtags=post.get("hashtags"),
        generation_attempts=state.get("generation_attempts", 1),
        pipeline_run_id=state.get("run_id"),
        hook_pattern_used=post.get("hook_pattern_used"),
    )

    post_id = result.get("id", "")

    # Persist the v2 detailed evaluation now that we have a post_id
    evaluation = state.get("current_evaluation")
    if evaluation and post_id:
        evaluation["post_id"] = post_id
        _persist_evaluation(evaluation)

    # Store Experience Fact in Hindsight — teaches the agent what worked
    if evaluation:
        try:
            from backend.memory.hindsight_client import store_generation_experience
            hook_pattern = post.get("hook_pattern_used", "unknown")
            await store_generation_experience(
                post_id=post_id,
                source=item.get("source", "unknown"),
                hook_pattern=hook_pattern or "unknown",
                overall_score=evaluation.get("overall_score", 0.0),
                metric_scores={
                    k: evaluation.get(k, 0.0)
                    for k in ("answer_relevancy", "faithfulness", "hallucination", "bias", "toxicity", "linkedin_quality")
                },
                passed=True,
                attempt_number=state.get("generation_attempts", 1),
                char_count=len(post.get("post_text", "")),
                issues=evaluation.get("issues", []),
            )
        except Exception as e:
            logger.debug("Could not store generation experience in Hindsight: %s", e)

    accepted = list(state.get("accepted_posts", []))
    accepted.append({
        "post_id": post_id,
        "title": item.get("title", ""),
        "quality_score": post.get("quality_score", 0),
    })

    logs = await _log(
        {**state, "logs": logs},
        f"Saved post (id={post_id}). Total accepted: {len(accepted)}"
    )

    return {"accepted_posts": accepted, "logs": logs, "step": "save"}


async def reject_post_node(state: PipelineState) -> dict:
    """Record a rejected item and store failure experience in Hindsight."""
    item = state.get("current_item", {})
    post = state.get("current_post", {})
    evaluation = state.get("current_evaluation", {})
    logs = await _log(state, f"Rejecting: {item.get('title', '')[:40]}... (issues: {len(post.get('issues', []))})")

    rejected = list(state.get("rejected_items", []))
    rejected.append({
        **item,
        "reason": "quality_fail",
        "issues": post.get("issues", []),
        "quality_score": post.get("quality_score", 0),
    })

    # Store failure Experience Fact in Hindsight — agents learn from failures too
    if evaluation:
        try:
            from backend.memory.hindsight_client import store_failed_experience
            await store_failed_experience(
                source=item.get("source", "unknown"),
                hook_pattern=post.get("hook_pattern_used", "unknown") or "unknown",
                overall_score=evaluation.get("overall_score", 0.0),
                metric_scores={
                    k: evaluation.get(k, 0.0)
                    for k in ("answer_relevancy", "faithfulness", "hallucination", "bias", "toxicity", "linkedin_quality")
                },
                attempt_number=state.get("generation_attempts", 1),
                char_count=len(post.get("post_text", "")),
                issues=evaluation.get("issues", []),
            )
        except Exception as e:
            logger.debug("Could not store failure experience in Hindsight: %s", e)

    return {"rejected_items": rejected, "logs": logs, "step": "reject"}


async def finalize_node(state: PipelineState) -> dict:
    """Finalize the pipeline run: log results and common issues."""
    accepted = state.get("accepted_posts", [])
    rejected = state.get("rejected_items", [])
    run_id = state.get("run_id", "")

    # Collect common issues
    all_issues: dict[str, int] = {}
    for item in rejected:
        for issue in item.get("issues", []):
            all_issues[issue] = all_issues.get(issue, 0) + 1

    sorted_issues = sorted(all_issues.items(), key=lambda x: x[1], reverse=True)
    common_issues = [i[0] for i in sorted_issues[:10]]

    avg_quality = 0.0
    if accepted:
        avg_quality = sum(p.get("quality_score", 0) for p in accepted) / len(accepted)

    total = len(accepted) + len(rejected)
    pass_rate = (len(accepted) / total * 100) if total > 0 else 0

    # Log pipeline run
    log_pipeline_run(
        run_id=run_id,
        status="completed",
        items_fetched=len(state.get("raw_items", [])),
        items_filtered=len(state.get("filtered_items", [])),
        posts_accepted=len(accepted),
        posts_rejected=len(rejected),
        avg_quality=round(avg_quality, 2),
        pass_rate=round(pass_rate, 1),
        common_issues=common_issues,
        config=state.get("run_config"),
    )

    logs = await _log(state, f"Pipeline complete: {len(accepted)} accepted, {len(rejected)} rejected ({pass_rate:.0f}% pass rate)")
    return {"logs": logs, "step": "finalize"}


# ============================================================
# GRAPH DEFINITION
# ============================================================

def _after_dedup_decision(state: PipelineState) -> str:
    """Route after duplicate check."""
    if state.get("step") == "dedup_reject":
        return "pick_next"
    return "save"


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("fetch_content", fetch_content_node)
    graph.add_node("filter_and_score", filter_and_score_node)
    graph.add_node("pick_next_item", pick_next_item_node)
    graph.add_node("extract_article", extract_article_node)
    graph.add_node("research_content", research_content_node)
    graph.add_node("evaluate_research", evaluate_research_node)
    graph.add_node("generate_post", generate_post_node)
    graph.add_node("evaluate_post", evaluate_post_node)
    graph.add_node("check_duplicate", check_duplicate_node)
    graph.add_node("save_post", save_post_node)
    graph.add_node("reject_post", reject_post_node)
    graph.add_node("finalize", finalize_node)

    # Entry point
    graph.set_entry_point("fetch_content")

    # Edges
    graph.add_edge("fetch_content", "filter_and_score")
    graph.add_edge("filter_and_score", "pick_next_item")

    # After picking next item: continue or done
    graph.add_conditional_edges(
        "pick_next_item",
        should_continue,
        {"process": "extract_article", "done": "finalize"},
    )

    graph.add_edge("extract_article", "research_content")
    graph.add_edge("research_content", "evaluate_research")

    # Research gate: pass → generate, retry → re-research, fallback → generate (degraded)
    graph.add_conditional_edges(
        "evaluate_research",
        research_gate_decision,
        {"pass": "generate_post", "retry": "research_content", "fallback": "generate_post"},
    )
    graph.add_edge("generate_post", "evaluate_post")

    # Quality gate: accept, retry, or reject
    graph.add_conditional_edges(
        "evaluate_post",
        quality_gate_decision,
        {"accept": "check_duplicate", "retry": "generate_post", "reject": "reject_post"},
    )

    # After duplicate check: save or skip
    graph.add_conditional_edges(
        "check_duplicate",
        _after_dedup_decision,
        {"save": "save_post", "pick_next": "pick_next_item"},
    )

    graph.add_edge("save_post", "pick_next_item")
    graph.add_edge("reject_post", "pick_next_item")
    graph.add_edge("finalize", END)

    return graph.compile()


# Pre-built pipeline instance
pipeline = build_pipeline()


async def run_pipeline(
    max_posts: int = 10,
    max_age_days: int = 7,
    limit_per_source: int = 15,
    max_retries: int = 3,
    sources: list[str] | None = None,
    broadcast_fn=None,
    timeout_seconds: int | None = None,
) -> PipelineState:
    """Run the complete pipeline and return final state."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Scale timeout with the number of posts requested.
    # Budget: 10 min base + 6 min per post (DSPy generation + 4-parallel eval + retries).
    if timeout_seconds is None:
        timeout_seconds = 600 + max_posts * 360

    # When filtering to 1-2 source types, fetch more per source to ensure enough candidates
    # after dedup/relevance filtering to meet max_posts. With 50% typical pass rate,
    # we need at least 2×max_posts filtered items, so fetch 3× to be safe.
    if sources and len(sources) <= 2:
        limit_per_source = max(limit_per_source, max_posts * 3)

    # Log pipeline start
    log_pipeline_run(
        run_id=run_id,
        status="running",
        config={
            "max_posts": max_posts,
            "max_age_days": max_age_days,
            "limit_per_source": limit_per_source,
            "max_retries": max_retries,
        },
    )

    initial_state: PipelineState = {
        "run_id": run_id,
        "run_config": {
            "max_posts": max_posts,
            "max_age_days": max_age_days,
            "limit_per_source": limit_per_source,
            "max_retries": max_retries,
            "sources": sources,
        },
        "raw_items": [],
        "trending_topics": [],
        "filtered_items": [],
        "items_remaining": 0,
        "current_item": None,
        "current_item_index": 0,
        "extracted_content": "",
        "research_brief": None,
        "research_attempts": 0,
        "research_evaluation": None,
        "research_failed": False,
        "current_post": None,
        "generation_attempts": 0,
        "accepted_posts": [],
        "rejected_items": [],
        "step": "init",
        "logs": [],
        "error": None,
        "broadcast_fn": broadcast_fn,
    }

    try:
        logger.info(f"Invoking pipeline graph with run_id={run_id}")
        result = await asyncio.wait_for(pipeline.ainvoke(initial_state), timeout=timeout_seconds)
        logger.info(f"Pipeline completed. Accepted: {len(result.get('accepted_posts', []))}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Pipeline timed out after {timeout_seconds}s (run_id={run_id})")
        # Salvage: count posts already saved to DB during this run
        try:
            from backend.database import fetch_all as _fetch_run_posts
            saved = _fetch_run_posts(
                "SELECT quality_score FROM posts WHERE pipeline_run_id = ?", [run_id]
            )
            n_saved = len(saved)
            avg_q = round(sum(r[0] for r in saved) / n_saved, 2) if n_saved else None
            log_pipeline_run(
                run_id=run_id,
                status="completed" if n_saved > 0 else "failed",
                items_fetched=len(initial_state.get("raw_items", [])),
                items_filtered=len(initial_state.get("filtered_items", [])),
                posts_accepted=n_saved,
                avg_quality=avg_q,
            )
            if n_saved:
                logger.info(
                    "Timeout salvage: %d posts already saved (avg quality %.2f), "
                    "marking run as completed", n_saved, avg_q
                )
        except Exception as salvage_err:
            logger.warning("Timeout salvage failed: %s", salvage_err)
            log_pipeline_run(run_id=run_id, status="failed")
        initial_state["error"] = f"Pipeline timed out after {timeout_seconds}s"
        initial_state["logs"].append(f"[ERROR] Pipeline timed out after {timeout_seconds}s")
        return initial_state
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}\n{tb}")
        log_pipeline_run(run_id=run_id, status="failed")
        initial_state["error"] = str(e)
        initial_state["logs"].append(f"[ERROR] Pipeline failed: {e}")
        return initial_state
