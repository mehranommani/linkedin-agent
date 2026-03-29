"""
Six-Metric Evaluation Engine
=============================
DeepEval/RAGAS-equivalent metrics for LinkedIn post quality.

Metrics:
  1. answer_relevancy  — Does post address source content's key points?
  2. faithfulness       — Are claims supported by source material?
  3. hallucination      — Any fabricated facts/stats/quotes?
  4. bias               — Balanced, not promotional/vendor-biased?
  5. toxicity           — Professional tone, no inappropriate content?
  6. linkedin_quality   — Hook strength + readability + SEO + structure

Performance: Batched into 4 parallel LLM calls for speed.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend.mcp.tools.llm import _judge_chat as _ollama_chat
from backend.config import ConfigManager

logger = logging.getLogger(__name__)


# ============================================================
# METRIC JUDGE PROMPTS
# ============================================================

_RELEVANCY_PROMPT = """You are a content relevancy judge. Evaluate how well the LinkedIn post addresses the key points from the source article.

SOURCE TITLE: {source_title}
SOURCE CONTENT: {source_content}

LINKEDIN POST:
{post_text}

Score 0-10 where:
- 10: Post perfectly captures and addresses all key points from the source
- 7-9: Post addresses most key points with good insight
- 4-6: Post is somewhat related but misses important points
- 0-3: Post barely relates to the source content

Respond in JSON with: score (float 0-10), reasoning (string), key_points_covered (list of strings), key_points_missed (list of strings)."""

_FAITHFULNESS_HALLUCINATION_PROMPT = """You are a factual accuracy judge evaluating a LinkedIn post written about a technical article or repository.

SOURCE TITLE: {source_title}
SOURCE CONTENT: {source_content}

LINKEDIN POST:
{post_text}

IMPORTANT CONTEXT: LinkedIn posts are allowed to add professional analysis, implications, and broader context beyond the source. What matters is:
- FAITHFULNESS: Are concrete factual claims (names, numbers, dates, specific features, version numbers) accurate and traceable to the source?
  Score LOW only if the post contradicts the source or invents specific facts.
  Score HIGH if the post accurately reflects the source's key points, even if it adds commentary.
- HALLUCINATION: Does the post invent specific facts, statistics, direct quotes, or product claims that are NOT in the source and could be misleading?
  Score LOW only for invented specific facts (e.g. "achieves 95% accuracy", "used by 10,000 companies" when source doesn't say this).
  Score HIGH if all specific claims are traceable to the source (general analysis/implications are acceptable).

Respond in JSON with:
- faithfulness_score (float 0-10): 10 = all concrete facts match source
- hallucination_score (float 0-10): 10 = no invented facts/stats/quotes
- faithfulness_reasoning (string)
- hallucination_reasoning (string)
- unsupported_claims (list of strings): only specific invented facts, not analytical commentary
- fabricated_content (list of strings): specific fabricated stats/quotes/numbers not in source"""

_BIAS_TOXICITY_PROMPT = """You are a professional content tone judge. Evaluate TWO aspects of this LinkedIn post.

LINKEDIN POST:
{post_text}

Evaluate:
1. BIAS: Is the post balanced and objective? Or does it show vendor/promotional/ideological bias?
2. TOXICITY: Is the tone fully professional? Any condescending, aggressive, or inappropriate language?

Respond in JSON with:
- bias_score (float 0-10): 10 = perfectly balanced and objective
- toxicity_score (float 0-10): 10 = fully professional and appropriate
- bias_reasoning (string)
- toxicity_reasoning (string)
- bias_indicators (list of strings): specific biased language found
- toxicity_indicators (list of strings): specific unprofessional language found"""

_LINKEDIN_QUALITY_PROMPT = """You are a LinkedIn content quality expert. Evaluate the overall quality of this post for the LinkedIn platform.

LINKEDIN POST:
{post_text}

Evaluate these aspects:
- HOOK: Does the opening grab attention? (pattern-interrupt, curiosity gap, bold statement, etc.)
- READABILITY: Is it easy to scan? Short paragraphs, clear language, accessible vocabulary?
- STRUCTURE: Hook → Analysis → Takeaway → CTA? Proper formatting?
- SEO: Does it naturally include relevant AI/ML/Data Science keywords?
- ENGAGEMENT: Would professionals want to like, comment, or share?
- EMOJI USAGE: Are emojis used appropriately (1-3 max, relevant positions)?
- HASHTAGS: 4-6 relevant, mix of broad and niche?

Respond in JSON with:
- linkedin_quality_score (float 0-10): 10 = perfect LinkedIn post
- hook_score (float 0-10)
- readability_score (float 0-10)
- structure_score (float 0-10)
- seo_score (float 0-10)
- engagement_score (float 0-10)
- reasoning (string)
- strengths (list of strings)
- improvements (list of strings)"""


# ============================================================
# METRIC SCHEMAS (for structured JSON output)
# ============================================================

_RELEVANCY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_points_covered": {"type": "array", "items": {"type": "string"}},
        "key_points_missed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "reasoning"],
}

_FAITH_HALLUC_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness_score": {"type": "number"},
        "hallucination_score": {"type": "number"},
        "faithfulness_reasoning": {"type": "string"},
        "hallucination_reasoning": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "fabricated_content": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithfulness_score", "hallucination_score"],
}

_BIAS_TOXICITY_SCHEMA = {
    "type": "object",
    "properties": {
        "bias_score": {"type": "number"},
        "toxicity_score": {"type": "number"},
        "bias_reasoning": {"type": "string"},
        "toxicity_reasoning": {"type": "string"},
        "bias_indicators": {"type": "array", "items": {"type": "string"}},
        "toxicity_indicators": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["bias_score", "toxicity_score"],
}

_LINKEDIN_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "linkedin_quality_score": {"type": "number"},
        "hook_score": {"type": "number"},
        "readability_score": {"type": "number"},
        "structure_score": {"type": "number"},
        "seo_score": {"type": "number"},
        "engagement_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["linkedin_quality_score", "reasoning"],
}


# ============================================================
# INDIVIDUAL METRIC EVALUATORS
# ============================================================

async def evaluate_relevancy(
    post_text: str, source_content: str, source_title: str, temperature: float = 0.2
) -> dict[str, Any]:
    """Evaluate answer relevancy of post against source."""
    prompt = _RELEVANCY_PROMPT.format(
        source_title=source_title,
        source_content=source_content[:3000],
        post_text=post_text,
    )
    try:
        raw = await _ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            format_schema=_RELEVANCY_SCHEMA,
        )
        data = json.loads(raw)
        return {
            "answer_relevancy": _clamp(data.get("score", 0)),
            "details": data,
        }
    except Exception as e:
        logger.error("Relevancy evaluation failed: %s", e)
        return {"answer_relevancy": 0.0, "details": {"error": str(e)}}


async def evaluate_faithfulness_hallucination(
    post_text: str, source_content: str, source_title: str, temperature: float = 0.2
) -> dict[str, Any]:
    """Evaluate faithfulness and hallucination (combined call)."""
    prompt = _FAITHFULNESS_HALLUCINATION_PROMPT.format(
        source_title=source_title,
        source_content=source_content[:3000],
        post_text=post_text,
    )
    try:
        raw = await _ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            format_schema=_FAITH_HALLUC_SCHEMA,
        )
        data = json.loads(raw)
        return {
            "faithfulness": _clamp(data.get("faithfulness_score", 0)),
            "hallucination": _clamp(data.get("hallucination_score", 0)),
            "details": data,
        }
    except Exception as e:
        logger.error("Faithfulness/hallucination evaluation failed: %s", e)
        return {"faithfulness": 0.0, "hallucination": 0.0, "details": {"error": str(e)}}


async def evaluate_bias_toxicity(
    post_text: str, temperature: float = 0.2
) -> dict[str, Any]:
    """Evaluate bias and toxicity (combined call)."""
    prompt = _BIAS_TOXICITY_PROMPT.format(post_text=post_text)
    try:
        raw = await _ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            format_schema=_BIAS_TOXICITY_SCHEMA,
        )
        data = json.loads(raw)
        return {
            "bias": _clamp(data.get("bias_score", 0)),
            "toxicity": _clamp(data.get("toxicity_score", 0)),
            "details": data,
        }
    except Exception as e:
        logger.error("Bias/toxicity evaluation failed: %s", e)
        return {"bias": 0.0, "toxicity": 0.0, "details": {"error": str(e)}}


async def evaluate_linkedin_quality(
    post_text: str, temperature: float = 0.2
) -> dict[str, Any]:
    """Evaluate LinkedIn-specific quality (hook, readability, SEO, structure)."""
    prompt = _LINKEDIN_QUALITY_PROMPT.format(post_text=post_text)
    try:
        raw = await _ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            format_schema=_LINKEDIN_QUALITY_SCHEMA,
        )
        data = json.loads(raw)
        return {
            "linkedin_quality": _clamp(data.get("linkedin_quality_score", 0)),
            "details": data,
        }
    except Exception as e:
        logger.error("LinkedIn quality evaluation failed: %s", e)
        return {"linkedin_quality": 0.0, "details": {"error": str(e)}}


# ============================================================
# PROGRAMMATIC GUARDRAILS (no LLM needed)
# ============================================================

def check_programmatic_guardrails(post_text: str) -> dict[str, Any]:  # noqa: C901
    """Run fast, non-LLM checks on the post."""
    issues: list[str] = []
    gen_config = ConfigManager.generation()

    # Character count check
    char_count = len(post_text)
    min_chars = gen_config.get("min_char_count", 1200)
    max_chars = gen_config.get("max_char_count", 3000)
    if char_count < min_chars:
        issues.append(f"Too short ({char_count} chars, minimum {min_chars})")
    if char_count > max_chars:
        issues.append(f"Too long ({char_count} chars, maximum {max_chars})")

    # Banned phrases check
    banned = ConfigManager.banned_phrases()
    text_lower = post_text.lower()
    found_banned = [p for p in banned if p.lower() in text_lower]
    if found_banned:
        issues.append(f"Contains banned phrases: {', '.join(found_banned[:3])}")

    # Hashtag count — only count actual hashtags (word boundary + #word), not
    # markdown headers, GitHub issue refs, or # inside code comments
    hashtag_count = len(re.findall(r'(?<!\w)#[A-Za-z]\w*', post_text))
    min_tags = gen_config.get("min_hashtags", 4)
    max_tags = gen_config.get("max_hashtags", 6)
    if hashtag_count < min_tags:
        issues.append(f"Too few hashtags ({hashtag_count}, minimum {min_tags})")
    if hashtag_count > max_tags + 2:
        issues.append(f"Too many hashtags ({hashtag_count}, maximum {max_tags})")

    # URL check
    if "http" not in post_text:
        issues.append("Missing source URL")

    # Code / script check
    code_patterns = ["```", "def ", "import ", "class ", "print(", "return ", "```python", "```js"]
    found_code = [p for p in code_patterns if p in post_text]
    if found_code:
        issues.append("Post contains code or programming syntax — rewrite in plain English without any code")

    # Markdown formatting check
    import re as _re
    badge_pattern = _re.search(r'\[!\[', post_text)
    if badge_pattern:
        issues.append("Post contains markdown badge syntax ([![...]) — remove all markdown formatting")
    markdown_patterns = ["**", "__", "~~"]
    found_md = [p for p in markdown_patterns if p in post_text]
    if found_md:
        issues.append(f"Post contains markdown formatting ({', '.join(found_md)}) — use plain text only")

    # No text after hashtags check
    hashtag_positions = [m.start() for m in re.finditer(r'(?<!\w)#[A-Za-z]\w*', post_text)]
    if hashtag_positions:
        last_hashtag_end = max(m.end() for m in re.finditer(r'(?<!\w)#[A-Za-z]\w*', post_text))
        text_after_hashtags = post_text[last_hashtag_end:].strip()
        if text_after_hashtags:
            issues.append("Post has text after hashtags — hashtags must be the very last thing in the post")

    # Star/dash bullets check
    lines = post_text.split('\n')
    bullet_lines = [l.strip() for l in lines if l.strip().startswith('* ') or l.strip().startswith('- ')]
    if bullet_lines:
        issues.append("Post uses star (*) or dash (-) bullets — use plain paragraphs instead")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "char_count": char_count,
        "hashtag_count": hashtag_count,
    }


# ============================================================
# HELPER
# ============================================================

def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    """Clamp a value between low and high."""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.0
