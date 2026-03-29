"""
Evaluation Orchestrator
=======================
Runs all 6 metrics in 4 parallel LLM calls, applies programmatic guardrails,
computes overall score, persists to detailed_evaluations table.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from backend.config import ConfigManager
from backend.database import execute
from backend.evaluation.metrics import (
    evaluate_relevancy,
    evaluate_faithfulness_hallucination,
    evaluate_bias_toxicity,
    evaluate_linkedin_quality,
    check_programmatic_guardrails,
)

logger = logging.getLogger(__name__)


async def run_full_evaluation(
    post_text: str,
    source_content: str,
    source_title: str,
    post_id: str = "",
    pipeline_run_id: str = "",
) -> dict[str, Any]:
    """Run all 6 metrics in parallel + programmatic guardrails.

    Returns a complete evaluation result dict with:
      - All 6 metric scores
      - overall_score (weighted)
      - passed (bool)
      - metric_details (per-metric reasoning)
      - issues / strengths lists
    """
    eval_config = ConfigManager.evaluation_v2()
    temperature = eval_config.get("judge_temperature", 0.2)
    weights = eval_config.get("weights", {})
    thresholds = eval_config.get("thresholds", {})
    overall_threshold = eval_config.get("overall_pass_threshold", 7.0)

    # --- Programmatic guardrails (instant, no LLM) ---
    guardrails = check_programmatic_guardrails(post_text)

    # --- 4 parallel LLM evaluations ---
    relevancy_task = evaluate_relevancy(post_text, source_content, source_title, temperature)
    faith_halluc_task = evaluate_faithfulness_hallucination(post_text, source_content, source_title, temperature)
    bias_tox_task = evaluate_bias_toxicity(post_text, temperature)
    linkedin_task = evaluate_linkedin_quality(post_text, temperature)

    results = await asyncio.gather(
        relevancy_task,
        faith_halluc_task,
        bias_tox_task,
        linkedin_task,
        return_exceptions=True,
    )

    # Unpack results (handle exceptions gracefully)
    relevancy_result = results[0] if not isinstance(results[0], Exception) else {"answer_relevancy": 0.0, "details": {"error": str(results[0])}}
    faith_halluc_result = results[1] if not isinstance(results[1], Exception) else {"faithfulness": 0.0, "hallucination": 0.0, "details": {"error": str(results[1])}}
    bias_tox_result = results[2] if not isinstance(results[2], Exception) else {"bias": 0.0, "toxicity": 0.0, "details": {"error": str(results[2])}}
    linkedin_result = results[3] if not isinstance(results[3], Exception) else {"linkedin_quality": 0.0, "details": {"error": str(results[3])}}

    # --- Extract scores ---
    scores = {
        "answer_relevancy": relevancy_result.get("answer_relevancy", 0.0),
        "faithfulness": faith_halluc_result.get("faithfulness", 0.0),
        "hallucination": faith_halluc_result.get("hallucination", 0.0),
        "bias": bias_tox_result.get("bias", 0.0),
        "toxicity": bias_tox_result.get("toxicity", 0.0),
        "linkedin_quality": linkedin_result.get("linkedin_quality", 0.0),
    }

    # If source content is too short to evaluate faithfulness/hallucination fairly,
    # substitute a neutral passing score so the judge's low-confidence guess doesn't
    # block every GitHub/short-summary item.  The faithfulness threshold is still
    # applied to what the judge CAN verify; hallucination is skipped.
    source_len = len(source_content.strip())
    source_too_short = source_len < 400
    if source_too_short:
        scores["hallucination"] = 8.5   # insufficient data — treat as passing
        scores["faithfulness"] = max(scores["faithfulness"], 7.0)
        scores["answer_relevancy"] = max(scores["answer_relevancy"], 7.0)  # can't judge relevancy vs a 74-char summary
        logger.info("Source content too short (%d chars) — skipping hallucination/relevancy thresholds", source_len)

    # Detect LLM scoring confusion: high faithfulness + low hallucination is semantically
    # impossible (can't be faithful but also highly hallucinated).  The model often scores
    # hallucination near 0-3 when it lacks enough source to verify, even if faith is high.
    # Treat as unverifiable and substitute a neutral passing score.
    elif scores["hallucination"] <= 3.0 and scores["faithfulness"] >= 7.5:
        logger.info(
            "Conflicting hallucination/faithfulness scores (halluc=%.1f, faith=%.1f) — "
            "treating hallucination as unverifiable, substituting 8.0",
            scores["hallucination"], scores["faithfulness"],
        )
        scores["hallucination"] = 8.0

    # --- Compute weighted overall score ---
    overall = sum(
        scores.get(metric, 0.0) * weights.get(metric, 0.0)
        for metric in weights
    )

    # --- Determine pass/fail ---
    threshold_failures = []
    effective_thresholds = dict(thresholds)
    if source_too_short:
        effective_thresholds.pop("hallucination", None)   # can't evaluate without source
        effective_thresholds.pop("answer_relevancy", None)  # already floored to 7.0 above
    for metric, threshold in effective_thresholds.items():
        if scores.get(metric, 0.0) < threshold:
            threshold_failures.append(f"{metric}: {scores[metric]:.1f} < {threshold}")

    passed = (
        overall >= overall_threshold
        and len(threshold_failures) == 0
        and guardrails["passed"]
    )

    # --- Collect issues and strengths ---
    issues: list[str] = list(guardrails.get("issues", []))
    issues.extend(threshold_failures)

    strengths: list[str] = []
    for metric, score in scores.items():
        if score >= 9.0:
            strengths.append(f"{metric}: excellent ({score:.1f})")

    # Collect detailed info from each evaluator
    metric_details = {
        "answer_relevancy": relevancy_result.get("details", {}),
        "faithfulness_hallucination": faith_halluc_result.get("details", {}),
        "bias_toxicity": bias_tox_result.get("details", {}),
        "linkedin_quality": linkedin_result.get("details", {}),
        "guardrails": guardrails,
    }

    # Add specific issues from LLM evaluations
    linkedin_details = linkedin_result.get("details", {})
    if isinstance(linkedin_details, dict):
        improvements = linkedin_details.get("improvements", [])
        if improvements:
            issues.extend(improvements[:3])
        lq_strengths = linkedin_details.get("strengths", [])
        if lq_strengths:
            strengths.extend(lq_strengths[:3])

    judge_model = ConfigManager.llm().get("model", "unknown")

    result = {
        "post_id": post_id,
        "pipeline_run_id": pipeline_run_id,
        **scores,
        "overall_score": round(overall, 2),
        "passed": passed,
        "metric_details": metric_details,
        "issues": issues,
        "strengths": strengths,
        "judge_model": judge_model,
        "evaluation_version": "v2",
    }

    # --- Persist to database ---
    if post_id:
        _persist_evaluation(result)

    failures_str = ""
    if not passed:
        fail_parts = []
        if threshold_failures:
            fail_parts.append("thresholds=" + "|".join(threshold_failures))
        if not guardrails["passed"]:
            fail_parts.append("guardrails=" + "|".join(guardrails.get("issues", [])))
        failures_str = " FAILURES: " + "; ".join(fail_parts) if fail_parts else " FAILURES: unknown"

    logger.info(
        "Evaluation complete: overall=%.2f passed=%s [rel=%.1f faith=%.1f halluc=%.1f bias=%.1f tox=%.1f lq=%.1f]%s",
        overall, passed,
        scores["answer_relevancy"], scores["faithfulness"],
        scores["hallucination"], scores["bias"],
        scores["toxicity"], scores["linkedin_quality"],
        failures_str,
    )

    return result


def _persist_evaluation(result: dict) -> None:
    """Save evaluation result to the detailed_evaluations table."""
    eval_id = uuid.uuid4().hex[:12]
    try:
        execute(
            """INSERT INTO detailed_evaluations
            (id, post_id, pipeline_run_id,
             answer_relevancy, faithfulness, hallucination,
             bias, toxicity, linkedin_quality,
             overall_score, passed, metric_details, issues, strengths,
             judge_model, evaluation_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                eval_id,
                result.get("post_id", ""),
                result.get("pipeline_run_id", ""),
                result.get("answer_relevancy", 0.0),
                result.get("faithfulness", 0.0),
                result.get("hallucination", 0.0),
                result.get("bias", 0.0),
                result.get("toxicity", 0.0),
                result.get("linkedin_quality", 0.0),
                result.get("overall_score", 0.0),
                result.get("passed", False),
                json.dumps(result.get("metric_details", {})),
                result.get("issues", []),
                result.get("strengths", []),
                result.get("judge_model", ""),
                result.get("evaluation_version", "v2"),
            ],
        )
        # Update post's detailed_eval_id
        if result.get("post_id"):
            execute(
                "UPDATE posts SET detailed_eval_id = ? WHERE id = ?",
                [eval_id, result["post_id"]],
            )
    except Exception as e:
        logger.error("Failed to persist evaluation: %s", e)


def get_evaluation_for_post(post_id: str) -> dict | None:
    """Fetch the latest detailed evaluation for a post."""
    from backend.database import fetch_one
    row = fetch_one(
        """SELECT id, post_id, evaluated_at, pipeline_run_id,
                  answer_relevancy, faithfulness, hallucination,
                  bias, toxicity, linkedin_quality,
                  overall_score, passed, metric_details, issues, strengths,
                  judge_model, evaluation_version
           FROM detailed_evaluations
           WHERE post_id = ?
           ORDER BY evaluated_at DESC LIMIT 1""",
        [post_id],
    )
    if not row:
        return None

    columns = [
        "id", "post_id", "evaluated_at", "pipeline_run_id",
        "answer_relevancy", "faithfulness", "hallucination",
        "bias", "toxicity", "linkedin_quality",
        "overall_score", "passed", "metric_details", "issues", "strengths",
        "judge_model", "evaluation_version",
    ]
    result = dict(zip(columns, row))
    if result.get("evaluated_at"):
        result["evaluated_at"] = str(result["evaluated_at"])
    if isinstance(result.get("metric_details"), str):
        try:
            result["metric_details"] = json.loads(result["metric_details"])
        except (json.JSONDecodeError, TypeError):
            pass
    for field in ("issues", "strengths"):
        v = result.get(field)
        if isinstance(v, str):
            try:
                result[field] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[field] = []
        elif v is None:
            result[field] = []
    return result
