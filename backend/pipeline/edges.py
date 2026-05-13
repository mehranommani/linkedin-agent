"""
Pipeline Edge Functions
======================
Conditional routing functions for the LangGraph pipeline.
"""
from __future__ import annotations

from backend.pipeline.state import PipelineState
from backend.config import ConfigManager


def quality_gate_decision(state: PipelineState) -> str:
    """Decide what to do after evaluation: accept, retry, or reject.

    Uses the v2 six-metric evaluation. Only retries if the failure is
    in a 'fixable' metric (relevancy, faithfulness, linkedin_quality).
    Bias/toxicity failures are not retryable.
    """
    post = state.get("current_post")
    if not post:
        return "reject"

    max_retries = state.get("run_config", {}).get("max_retries", 3)
    passed = post.get("passed_quality_gate", False)

    if passed:
        return "accept"

    # Check if failure is in fixable metrics (worth retrying)
    evaluation = state.get("current_evaluation", {})
    eval_config = ConfigManager.evaluation_v2()
    thresholds = eval_config.get("thresholds", {})

    # All metrics are retryable except toxicity (cannot be fixed by rephrasing alone)
    fixable_metrics = {"answer_relevancy", "faithfulness", "hallucination", "linkedin_quality", "bias"}
    non_fixable_failures = []

    for metric, threshold in thresholds.items():
        score = evaluation.get(metric, 0)
        if score < threshold and metric not in fixable_metrics:
            non_fixable_failures.append(metric)

    # Only toxicity failures are truly non-fixable — reject immediately
    if non_fixable_failures:
        return "reject"

    if state.get("generation_attempts", 0) < max_retries:
        return "retry"

    return "reject"


def should_continue(state: PipelineState) -> str:
    """Decide whether to process more items or finish."""
    max_posts = state.get("run_config", {}).get("max_posts", 10)
    accepted = state.get("accepted_posts", [])
    remaining = state.get("items_remaining", 0)

    if remaining <= 0 or len(accepted) >= max_posts:
        return "done"
    return "process"


def research_gate_decision(state: PipelineState) -> str:
    """Decide what to do after research evaluation: pass, retry, or fallback.

    - pass:    research brief passed → proceed to generate_post
    - retry:   failed but attempts < 2 → retry research with higher temperature
    - fallback: failed after 2 attempts → degraded mode, generate without brief
    """
    evaluation = state.get("research_evaluation") or {}
    attempts = state.get("research_attempts", 0)

    if evaluation.get("passed", False):
        return "pass"

    if attempts < 2:
        return "retry"

    return "fallback"
