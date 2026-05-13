"""
DSPy Post Generator
===================
Defines the LinkedInPostSignature and wraps it with dspy.Refine for
automatic guardrail-level retry with LLM-generated feedback between attempts.

Usage:
    from backend.dspy_modules.post_generator import get_post_generator
    import dspy

    generator = get_post_generator()
    pred = dspy.asyncify(generator)(angle=..., evidence=..., ...)
    post_text = pred.post
"""
from __future__ import annotations

import logging
import os
from typing import List

import dspy

from backend.evaluation.metrics import check_programmatic_guardrails

logger = logging.getLogger(__name__)

_COMPILED_PATH = "compiled/post_generator.json"


class LinkedInPostSignature(dspy.Signature):
    """Generate a LinkedIn post about a software project for a professional audience.

    The post must:
    - FIRST LINE: 5-18 words maximum, standalone complete thought (the LinkedIn scroll-stopper)
    - Use blank lines between every paragraph (LinkedIn is scanned, not read)
    - Each paragraph: 1-3 sentences max
    - Open with a pattern-interrupt hook: bold claim, surprising stat, or discovery
    - Have 3-5 short paragraphs of concrete insight (not marketing fluff)
    - End with the source URL on its own line, then 4-6 hashtags on the last line
    - Be 1200-2200 characters total (optimal 1400-1800)
    - NOT start with 'What if you could', 'What if you', or 'Introducing'
    - NOT use markdown code blocks (```), star bullets (* item), or badge syntax ([![)
    """

    angle: str = dspy.InputField(
        desc="The specific surprising fact or technical insight to highlight"
    )
    evidence: List[str] = dspy.InputField(
        desc="Concrete data points, benchmarks, and technical details from the source"
    )
    repo_url: str = dspy.InputField(
        desc="Source URL that must appear in the post"
    )
    source_excerpt: str = dspy.InputField(
        desc="Key technical content from the source (README excerpt, description)"
    )
    style_context: str = dspy.InputField(
        desc="Writing style guidance and lessons from successful past posts"
    )
    previous_feedback: str = dspy.InputField(
        desc="Issues from the previous generation attempt to fix (empty on first try)"
    )
    constraints: str = dspy.InputField(
        desc="Banned phrases and formatting rules to strictly follow"
    )
    post: str = dspy.OutputField(
        desc="The complete LinkedIn post, ready to publish"
    )


def post_quality_reward(args: dict, pred: dspy.Prediction, trace=None) -> float:
    """Returns 1.0 if post passes all programmatic guardrails, 0.0 otherwise."""
    try:
        result = check_programmatic_guardrails(pred.post)
        if not result["passed"]:
            logger.debug("DSPy Refine: guardrail failed — %s", result.get("issues", []))
        return 1.0 if result["passed"] else 0.0
    except Exception as exc:
        logger.warning("post_quality_reward error: %s", exc)
        return 0.0


def _build_post_generator() -> dspy.Refine:
    """Build the Refine-wrapped post generator, loading compiled instructions if available."""
    base = dspy.Predict(LinkedInPostSignature)
    if os.path.exists(_COMPILED_PATH):
        try:
            base.load(_COMPILED_PATH)
            logger.info("Loaded compiled DSPy post generator from %s", _COMPILED_PATH)
        except Exception as exc:
            logger.warning("Could not load compiled DSPy program (%s) — using default instructions", exc)

    return dspy.Refine(
        module=base,
        N=3,
        reward_fn=post_quality_reward,
        threshold=1.0,
    )


# Lazily-initialized singleton
_post_generator: dspy.Refine | None = None


def get_post_generator() -> dspy.Refine:
    """Return (and lazily create) the module-level post generator singleton."""
    global _post_generator
    if _post_generator is None:
        _post_generator = _build_post_generator()
    return _post_generator
