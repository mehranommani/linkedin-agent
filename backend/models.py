"""
Pydantic Models
===============
Shared data models for API, pipeline, and MCP tools.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# --- Content Items (from sources) ---

class ContentItem(BaseModel):
    """Universal content item from any source."""
    title: str
    url: str
    source: str  # github, hackernews, reddit, rss, arxiv, devto, producthunt, papers
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    image_url: str = ""
    timestamp: float = 0.0
    relevance_score: float = 0.0
    trending_boost: float = 0.0
    extracted_content: str | None = None


# --- Posts ---

class PostCreate(BaseModel):
    """Input for creating a new post in the database."""
    title: str
    original_url: str
    source: str
    source_summary: str | None = None
    image_url: str | None = None
    source_metrics: dict[str, Any] | None = None
    extracted_content: str | None = None
    post_text: str
    hashtags: list[str] = Field(default_factory=list)
    relevance_score: float
    quality_score: float
    faithfulness_score: float = 0.0
    trending_boost: float = 0.0
    content_fingerprint: str | None = None
    generation_attempts: int = 1
    pipeline_run_id: str | None = None


class PostResponse(BaseModel):
    """Post as returned by the API."""
    id: str
    created_at: datetime | None = None
    title: str
    original_url: str
    source: str
    image_url: str | None = None
    post_text: str
    char_count: int = 0
    hashtags: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    quality_score: float = 0.0
    faithfulness_score: float = 0.0
    trending_boost: float = 0.0
    final_score: float = 0.0
    is_used: bool = False
    generation_attempts: int = 1
    pipeline_run_id: str | None = None


class PostListResponse(BaseModel):
    """Paginated post listing."""
    posts: list[PostResponse]
    total: int
    page: int
    per_page: int
    pages: int


class PostStats(BaseModel):
    """Dashboard summary statistics."""
    total_posts: int = 0
    used_posts: int = 0
    avg_relevance: float = 0.0
    avg_quality: float = 0.0
    avg_score: float = 0.0
    avg_length: int = 0
    source_distribution: list[SourceStat] = Field(default_factory=list)


class SourceStat(BaseModel):
    source: str
    count: int
    avg_quality: float = 0.0
    avg_score: float = 0.0


# Fix forward reference
PostStats.model_rebuild()


# --- LinkedIn Post (structured LLM output) ---

class LinkedInPost(BaseModel):
    """Structured output from the LLM post generator."""
    hook: str = Field(description="Opening line that grabs attention (1-2 sentences)")
    body: str = Field(description="Main content with insights and analysis (3-5 paragraphs)")
    takeaway: str = Field(description="Key takeaway or call-to-action (1-2 sentences)")
    hashtags: list[str] = Field(description="4-6 relevant hashtags")
    url: str = Field(description="Source URL")
    hook_pattern_used: str = Field(default="", description="Which hook pattern was used")
    seo_keywords_used: list[str] = Field(default_factory=list, description="SEO keywords woven in")
    emoji_placement: str = Field(default="", description="Where emojis were placed")

    @property
    def full_text(self) -> str:
        """Assemble the complete LinkedIn post."""
        tags = " ".join(f"#{t.lstrip('#')}" for t in self.hashtags)
        return f"{self.hook}\n\n{self.body}\n\n{self.takeaway}\n\n{self.url}\n\n{tags}"


# --- Evaluation ---

class EvaluationResult(BaseModel):
    """Result of evaluating a single post (v1 — kept for backward compat)."""
    post_id: str
    relevance_score: float = 0.0
    quality_score: float = 0.0
    faithfulness_score: float = 0.0
    confidence_score: float = 0.0
    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    judge_model: str = ""
    evaluation_type: str = "inline"  # 'inline' or 'batch'


class DetailedEvaluationResult(BaseModel):
    """Six-metric evaluation result (v2 — DeepEval/RAGAS equivalent)."""
    post_id: str = ""
    pipeline_run_id: str = ""
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    hallucination: float = 0.0
    bias: float = 0.0
    toxicity: float = 0.0
    linkedin_quality: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    metric_details: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    judge_model: str = ""
    evaluation_version: str = "v2"


class EvaluationReport(BaseModel):
    """Aggregated evaluation report."""
    total_posts: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_relevance: float = 0.0
    avg_quality: float = 0.0
    avg_confidence: float = 0.0
    avg_char_count: float = 0.0
    results: list[EvaluationResult] = Field(default_factory=list)
    common_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# --- Human Feedback ---

class HumanFeedbackCreate(BaseModel):
    """Input for submitting human feedback on a post."""
    rating: int = Field(ge=1, le=5, description="Star rating 1-5")
    text_feedback: str = Field(default="", description="Optional text feedback")
    is_good_example: bool = Field(default=False, description="Mark as a good example for few-shot")


class HumanFeedbackResponse(BaseModel):
    """Human feedback as returned by the API."""
    id: str
    post_id: str
    created_at: datetime | None = None
    rating: int
    text_feedback: str = ""
    is_good_example: bool = False
    feedback_weight: float = 1.0
    feedback_source: str = "human"


# --- Sources ---

class SourceConfig(BaseModel):
    """Source configuration from the database."""
    id: str
    source_type: str
    name: str
    url: str | None = None
    category: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    priority: int = 5
    total_fetched: int = 0
    total_accepted: int = 0
    avg_quality: float = 0.0
    last_fetched_at: datetime | None = None


class SourceCreate(BaseModel):
    """Input for creating a new source."""
    source_type: str
    name: str
    url: str | None = None
    category: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    priority: int = 5


# --- Pipeline ---

class PipelineRunConfig(BaseModel):
    """Configuration for a pipeline run."""
    max_posts: int = 10
    max_age_days: int = 7
    limit_per_source: int = 15
    max_retries: int = 3
    sources: list[str] | None = None  # None = all active sources


class PipelineRunStatus(BaseModel):
    """Status of a pipeline run."""
    run_id: str
    status: str = "running"  # running, completed, cancelled, failed
    started_at: datetime | None = None
    completed_at: datetime | None = None
    items_fetched: int = 0
    items_filtered: int = 0
    posts_accepted: int = 0
    posts_rejected: int = 0
    current_step: str = ""
    progress: float = 0.0  # 0-100


# --- Trending ---

class TrendingTopic(BaseModel):
    """A trending topic detected across sources."""
    topic: str
    mentions: int = 0
    momentum: float = 0.0
    sources: list[str] = Field(default_factory=list)


# --- WebSocket Messages ---

class WSMessage(BaseModel):
    """WebSocket message for real-time pipeline updates."""
    type: str  # status, log, result, error
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
