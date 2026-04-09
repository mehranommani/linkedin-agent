"""
Pipeline State
==============
TypedDict defining the state that flows through the LangGraph pipeline.
"""
from __future__ import annotations

from typing import TypedDict, Any, Callable, Awaitable


class PipelineState(TypedDict, total=False):
    """State flowing through the LangGraph pipeline."""

    # --- Run Config ---
    run_id: str
    run_config: dict[str, Any]  # max_posts, max_age_days, limit_per_source, etc.

    # --- Fetch Phase ---
    raw_items: list[dict]          # All items fetched from sources
    trending_topics: list[dict]    # Detected trending topics

    # --- Filter Phase ---
    filtered_items: list[dict]     # Items that passed relevance filter
    items_remaining: int           # Count of items left to process

    # --- Processing Loop ---
    current_item: dict | None      # Item being processed right now
    current_item_index: int        # Index in filtered_items
    extracted_content: str         # Article content for current item

    # --- Research Phase (between extract and generate) ---
    research_brief: dict | None     # ResearchBrief from research_content_node
    research_attempts: int          # Retry counter for research phase
    research_evaluation: dict | None  # Research brief evaluation result
    research_failed: bool           # True if research retries exhausted → degraded mode

    # --- Generation ---
    current_post: dict | None      # Generated post data
    generation_attempts: int       # Retry counter for current item
    current_evaluation: dict | None  # Detailed v2 evaluation result

    # --- Results ---
    accepted_posts: list[dict]     # Posts that passed quality gate
    rejected_items: list[dict]     # Items that failed after max retries

    # --- Tracking ---
    step: str                      # Current pipeline step name
    logs: list[str]                # Log messages for WebSocket streaming
    error: str | None              # Error message if pipeline fails

    # --- WebSocket broadcasting (optional, injected when called via WebSocket) ---
    broadcast_fn: Callable[[dict], Awaitable[None]] | None
