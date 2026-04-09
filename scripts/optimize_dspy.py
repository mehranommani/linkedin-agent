"""
DSPy MIPROv2 Offline Optimizer
===============================
Automatically optimizes the LinkedIn post generation instructions using
existing high-quality posts from the database as training signal.

Usage:
    python scripts/optimize_dspy.py

Output:
    compiled/post_generator.json  (loaded automatically at server startup)

Requirements:
    - At least 20 posts with overall_score >= 7.5 in the database
    - Ollama running locally (or set CEREBRAS_API_KEY in .env for cloud)
    - ~10-20 minutes runtime for light auto mode

The optimizer uses MIPROv2 in zero-shot mode: it analyzes the signature
and dataset properties, proposes candidate instructions via Bayesian
optimization, and selects the best-performing wording — no human prompts
needed.
"""
from __future__ import annotations

import json
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dspy
from dspy.teleprompt import MIPROv2

from backend.database import fetch_all
from backend.dspy_modules.post_generator import LinkedInPostSignature, post_quality_reward
from backend.mcp.tools.llm import _load_env_vars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def configure_lm() -> str:
    """Configure DSPy LM from .env — same priority as main.py startup."""
    env = _load_env_vars()

    if key := env.get("CEREBRAS_API_KEY"):
        dspy.configure(lm=dspy.LM(
            "openai/qwen-3-235b-a22b-instruct-2507",
            api_key=key,
            api_base="https://api.cerebras.ai/v1",
            max_tokens=2000,
        ))
        return "Cerebras (qwen-3-235b)"

    if key := (env.get("GROQ_API_KEY") or env.get("GROQ_API_KEY_1")):
        dspy.configure(lm=dspy.LM(
            "openai/llama-3.3-70b-versatile",
            api_key=key,
            api_base="https://api.groq.com/openai/v1",
            max_tokens=2000,
        ))
        return "Groq (llama-3.3-70b)"

    dspy.configure(lm=dspy.LM(
        "ollama_chat/qwen2.5:14b",
        api_base="http://localhost:11434",
        max_tokens=2000,
    ))
    return "Ollama (qwen2.5:14b)"


def load_training_examples() -> list[dspy.Example]:
    """Load high-quality posts from DB as DSPy training examples."""
    rows = fetch_all(
        """SELECT
               p.title,
               p.post_text,
               p.source_summary,
               p.original_url,
               COALESCE(d.overall_score, p.quality_score) AS score
           FROM posts p
           LEFT JOIN detailed_evaluations d ON p.detailed_eval_id = d.id
           WHERE p.post_text IS NOT NULL
             AND p.post_text != ''
             AND COALESCE(d.overall_score, p.quality_score) >= 7.5
           ORDER BY score DESC
           LIMIT 100"""
    )
    if not rows:
        logger.warning("No qualifying posts found — try lowering the score threshold.")
        return []

    examples = []
    for title, post_text, source_summary, original_url, score in rows:
        example = dspy.Example(
            angle=title or "",
            evidence=[],
            repo_url=original_url or "",
            source_excerpt=(source_summary or "")[:500],
            style_context="",
            previous_feedback="",
            constraints="",
            post=post_text,
        ).with_inputs("angle", "evidence", "repo_url", "source_excerpt",
                      "style_context", "previous_feedback", "constraints")
        examples.append(example)

    logger.info("Loaded %d training examples (score >= 7.5)", len(examples))
    return examples


def main() -> None:
    lm_name = configure_lm()
    logger.info("Using LM: %s", lm_name)

    examples = load_training_examples()
    if len(examples) < 10:
        logger.error(
            "Need at least 10 qualifying posts to optimize. Found %d. "
            "Run the pipeline to generate more posts first.",
            len(examples),
        )
        sys.exit(1)

    logger.info("Starting MIPROv2 zero-shot optimization (%d examples)...", len(examples))

    optimizer = MIPROv2(
        metric=lambda example, pred, trace=None: post_quality_reward(None, pred),
        auto="light",               # ~10 trials, fast (~10-20 min)
        max_bootstrapped_demos=0,   # zero-shot — no labeled examples needed
        max_labeled_demos=0,
        num_threads=1,              # conservative: avoid rate limit bursts
    )

    program = dspy.Predict(LinkedInPostSignature)
    compiled = optimizer.compile(
        program,
        trainset=examples,
        requires_permission_to_run=False,
    )

    os.makedirs("compiled", exist_ok=True)
    compiled.save("compiled/post_generator.json")
    logger.info("Saved compiled/post_generator.json — restart the server to load it.")

    # Print the optimized instruction for inspection
    try:
        with open("compiled/post_generator.json") as f:
            data = json.load(f)
        instruction = data.get("predict", {}).get("signature", {}).get("instructions", "")
        if instruction:
            logger.info("Optimized instruction:\n%s", instruction)
    except Exception:
        pass


if __name__ == "__main__":
    main()
