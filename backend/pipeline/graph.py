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
from backend.pipeline.edges import quality_gate_decision, should_continue
from backend.config import ConfigManager

# MCP tool functions (called directly, not via MCP protocol in-process)
from backend.mcp.tools.sources import fetch_source
from backend.mcp.tools.database import (
    insert_post, check_url_exists, log_pipeline_run,
    get_active_sources,
)
from backend.mcp.tools.llm import generate_post as llm_generate_post, batch_judge_relevance
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

    # Batch relevance check via LLM
    batch_input = [
        {"title": item.get("title", ""), "summary": item.get("summary", "")[:150]}
        for item in items[:50]  # Limit batch size
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

    for i, item in enumerate(items[:50]):
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

    # Build rich, actionable retry feedback from the previous evaluation
    previous_issues = []
    current_evaluation = state.get("current_evaluation")
    if current_evaluation:
        details = current_evaluation.get("metric_details", {})
        fh = details.get("faithfulness_hallucination", {})
        lq = details.get("linkedin_quality", {})

        fabricated = fh.get("fabricated_content", [])
        if fabricated:
            previous_issues.append(
                f"CRITICAL — Remove fabricated claims not found in the source: {'; '.join(fabricated[:3])}"
            )
        unsupported = fh.get("unsupported_claims", [])
        if unsupported:
            previous_issues.append(
                f"Unsupported claims — only state facts from the source: {'; '.join(unsupported[:2])}"
            )
        fh_reason = fh.get("hallucination_reasoning", "")
        if fh_reason and not fabricated:
            previous_issues.append(f"Hallucination issue: {fh_reason[:200]}")
        faith_reason = fh.get("faithfulness_reasoning", "")
        if faith_reason and not unsupported:
            previous_issues.append(f"Faithfulness issue: {faith_reason[:200]}")
        improvements = lq.get("improvements", [])
        if improvements:
            previous_issues.extend(improvements[:3])

        # Fallback: include threshold failures if no detailed reasoning found
        if not previous_issues:
            previous_issues = current_evaluation.get("issues", [])

    # Seed for reproducibility but variety across attempts
    url = item.get("url", "")
    seed = hash(url + str(attempt)) % (2**31)

    banned = ConfigManager.banned_phrases()

    result = await llm_generate_post(
        title=item.get("title", ""),
        summary=item.get("summary", ""),
        source=item.get("source", ""),
        source_url=item.get("url", ""),
        content=state.get("extracted_content"),
        trending_topics=[t.get("name", "") for t in state.get("trending_topics", [])],
        avoid_phrases=banned,
        previous_issues=previous_issues,
        lessons_learned=lessons,
        feedback_patterns=feedback_patterns or None,
        temperature=temp,
        seed=seed,
    )

    post_text = result.get("post_text", "")
    # If the LLM omitted the source URL from the assembled text, append it.
    if url and "http" not in post_text:
        post_text = post_text.rstrip() + f"\n\n{url}"

    post_data = {
        "post_text": post_text,
        "hook": result.get("hook", ""),
        "body": result.get("body", ""),
        "takeaway": result.get("takeaway", ""),
        "hashtags": result.get("hashtags", []),
        "url": result.get("url", url),
        "char_count": len(post_text),
        "issues": [],
        "quality_score": 0,
        "faithfulness_score": 0,
        "passed_quality_gate": False,
        "hook_pattern_used": result.get("hook_pattern_used", ""),
        "seo_keywords_used": result.get("seo_keywords_used", []),
    }

    hook_used = post_data.get("hook_pattern_used") or "unknown"
    logs = await _log(
        {**state, "logs": logs},
        f"Generated post: {post_data['char_count']} chars (temp={temp}, hook={hook_used})"
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

    graph.add_edge("extract_article", "generate_post")
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
    timeout_seconds: int = 1800,
) -> PipelineState:
    """Run the complete pipeline and return final state."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

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
