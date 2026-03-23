"""
MCP Evaluator Tools
===================
Tools for quality evaluation: relevance judging, quality scoring,
faithfulness checking, duplicate detection.
All use LLM-as-judge via the llm tools.
"""
from __future__ import annotations

import json
import hashlib
from typing import Any

from backend.mcp.server import mcp_server
from backend.mcp.tools.llm import _ollama_chat, _ollama_generate
from backend.database import fetch_all
from backend.config import ConfigManager


def _ngrams(text: str, n: int = 3) -> set[str]:
    """Generate character n-grams for similarity comparison."""
    text = text.lower().strip()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    a = _ngrams(text1, n)
    b = _ngrams(text2, n)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@mcp_server.tool()
async def judge_relevance(
    title: str,
    summary: str,
    source: str = "",
) -> dict:
    """Judge if content is relevant for a LinkedIn AI/ML audience.

    Returns is_relevant (bool), score (0-10), and reasoning.
    """
    schema = {
        "type": "object",
        "properties": {
            "is_relevant": {"type": "boolean"},
            "score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["is_relevant", "score", "reasoning"],
    }

    prompt = f"""You are an AI content relevance judge for a LinkedIn page focused on AI, ML, Data Science, GenAI, and Agentic AI.

Rate this content's relevance on a 0-10 scale:
- 9-10: Core AI/ML breakthrough, major tool release, seminal research
- 7-8: Relevant AI-adjacent content useful for AI practitioners
- 5-6: Somewhat related to AI/tech
- 0-4: Not relevant to AI audience

Title: {title}
Source: {source}
Summary: {summary[:500]}

Return JSON with: is_relevant (true if score >= 7), score (0-10), reasoning (1 sentence)."""

    raw = await _ollama_generate(prompt=prompt, temperature=0.3, format_schema=schema)

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return {"is_relevant": True, "score": 7.0, "reasoning": "parse_fallback"}


@mcp_server.tool()
async def judge_quality(
    post_text: str,
    original_title: str = "",
) -> dict:
    """Judge the quality of a generated LinkedIn post.

    Returns quality_score (0-10), issues list, strengths list, passed bool.
    """
    gen_config = ConfigManager.generation()
    min_chars = gen_config.get("min_char_count", 1200)
    max_chars = gen_config.get("max_char_count", 3000)
    eval_config = ConfigManager.evaluation()
    min_quality = eval_config.get("min_quality", 7.0)

    schema = {
        "type": "object",
        "properties": {
            "quality_score": {"type": "number"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "strengths": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["quality_score", "issues", "strengths"],
    }

    banned = ConfigManager.banned_phrases()
    banned_ctx = f"\nBanned phrases (automatic fail): {', '.join(banned[:10])}" if banned else ""

    prompt = f"""You are a LinkedIn content quality judge. Evaluate this post strictly.

Score on 0-10 scale considering:
1. Professional tone (not overly casual, salesy, or clickbaity)
2. Clear structure (hook, body with analysis, takeaway)
3. Specific technical insights (not vague platitudes)
4. Engagement potential for AI/ML professionals
5. Grammar and readability
6. Appropriate length ({min_chars}-{max_chars} chars)
7. Includes source URL and relevant hashtags
8. Avoids clichés and buzzwords{banned_ctx}

Original topic: {original_title}

Post to evaluate:
---
{post_text}
---

Return JSON with: quality_score (0-10), issues (list of problems), strengths (list of positives)."""

    raw = await _ollama_generate(prompt=prompt, temperature=0.3, format_schema=schema)

    try:
        data = json.loads(raw)
        score = float(data.get("quality_score", 0))

        # Additional programmatic checks
        issues = list(data.get("issues", []))
        char_count = len(post_text)

        if char_count < min_chars:
            issues.append(f"Too short: {char_count} chars (min {min_chars})")
        if char_count > max_chars:
            issues.append(f"Too long: {char_count} chars (max {max_chars})")

        # Check banned phrases
        post_lower = post_text.lower()
        for phrase in banned:
            if phrase in post_lower:
                issues.append(f"Contains banned phrase: '{phrase}'")
                score = min(score, 5.0)
                break

        passed = score >= min_quality and char_count >= min_chars

        return {
            "quality_score": score,
            "issues": issues,
            "strengths": data.get("strengths", []),
            "passed": passed,
            "char_count": char_count,
        }
    except json.JSONDecodeError:
        return {
            "quality_score": 5.0,
            "issues": ["LLM judge parse failed"],
            "strengths": [],
            "passed": False,
            "char_count": len(post_text),
        }


@mcp_server.tool()
async def judge_faithfulness(
    post_text: str,
    source_content: str,
) -> dict:
    """Judge how faithfully a post represents the source material.

    Returns faithfulness_score (0-10) and any hallucinations found.
    """
    schema = {
        "type": "object",
        "properties": {
            "faithfulness_score": {"type": "number"},
            "hallucinations": {"type": "array", "items": {"type": "string"}},
            "accurate_claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["faithfulness_score", "hallucinations"],
    }

    prompt = f"""You are a factual accuracy judge. Compare the LinkedIn post against the source material.

Score faithfulness 0-10:
- 9-10: All claims supported by source, no fabrication
- 7-8: Mostly accurate with minor embellishments
- 5-6: Some unsupported claims
- 0-4: Significant hallucinations or misrepresentation

Source material:
---
{source_content[:3000]}
---

LinkedIn post:
---
{post_text}
---

Return JSON with: faithfulness_score (0-10), hallucinations (list of unsupported claims), accurate_claims (list of verified claims)."""

    raw = await _ollama_generate(prompt=prompt, temperature=0.3, format_schema=schema)

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return {"faithfulness_score": 5.0, "hallucinations": ["parse_failed"]}


@mcp_server.tool()
def check_duplicate(post_text: str) -> dict:
    """Check if a post is too similar to existing posts using Jaccard n-gram similarity.

    Returns is_duplicate (bool), similarity score, and similar_post_id if found.
    """
    threshold = ConfigManager.evaluation().get("duplicate_threshold", 0.85)

    # Get recent posts for comparison
    rows = fetch_all("""
        SELECT id, post_text FROM posts
        ORDER BY created_at DESC LIMIT 100
    """)

    max_sim = 0.0
    similar_id = None

    for post_id, existing_text in rows:
        if existing_text:
            sim = _jaccard_similarity(post_text, existing_text)
            if sim > max_sim:
                max_sim = sim
                similar_id = post_id

    return {
        "is_duplicate": max_sim >= threshold,
        "similarity": round(max_sim, 3),
        "similar_post_id": similar_id if max_sim >= threshold else None,
        "threshold": threshold,
    }


@mcp_server.tool()
async def get_trending_topics(limit: int = 10) -> dict:
    """Detect trending topics across recent content items.

    Analyzes titles from recent posts to identify emerging themes.
    """
    rows = fetch_all("""
        SELECT title, source FROM posts
        WHERE created_at >= current_timestamp - INTERVAL 7 DAY
        ORDER BY created_at DESC LIMIT 200
    """)

    if not rows:
        return {"topics": []}

    titles_text = "\n".join(f"- [{row[1]}] {row[0]}" for row in rows[:50])

    schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "mentions": {"type": "integer"},
                        "momentum": {"type": "number"},
                    },
                    "required": ["name", "mentions", "momentum"],
                },
            },
        },
        "required": ["topics"],
    }

    prompt = f"""Analyze these recent AI/ML content titles and identify the top {limit} trending topics.

Titles:
{titles_text}

For each topic, estimate:
- name: Short topic name (2-4 words)
- mentions: How many titles reference this topic
- momentum: 0.0-1.0 how "hot" this topic is (1.0 = extremely trending)

Return JSON with a "topics" array, sorted by momentum descending."""

    raw = await _ollama_generate(prompt=prompt, temperature=0.3, format_schema=schema)

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return {"topics": []}


@mcp_server.tool()
async def run_detailed_evaluation(
    post_text: str,
    source_content: str,
    source_title: str,
    post_id: str = "",
    pipeline_run_id: str = "",
) -> dict:
    """Run the full 6-metric evaluation engine on a post.

    Returns all 6 metric scores, overall score, pass/fail, issues, strengths.
    Metrics: answer_relevancy, faithfulness, hallucination, bias, toxicity, linkedin_quality.
    """
    from backend.evaluation.evaluator import run_full_evaluation

    return await run_full_evaluation(
        post_text=post_text,
        source_content=source_content,
        source_title=source_title,
        post_id=post_id,
        pipeline_run_id=pipeline_run_id,
    )


@mcp_server.tool()
def get_post_evaluation(post_id: str) -> dict:
    """Fetch the latest detailed evaluation for a post.

    Returns all 6 metric scores with per-metric reasoning and details.
    """
    from backend.evaluation.evaluator import get_evaluation_for_post

    result = get_evaluation_for_post(post_id)
    if result is None:
        return {"found": False, "message": f"No evaluation found for post {post_id}"}
    return {"found": True, "evaluation": result}
