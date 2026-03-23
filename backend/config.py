"""
ConfigManager
=============
Reads configuration from DuckDB config table.
Provides typed access to all dynamic settings.
"""
from __future__ import annotations

import json
from typing import Any

from backend.database import fetch_one, execute


class ConfigManager:
    """Reads/writes config from the DuckDB config table."""

    _cache: dict[str, Any] = {}

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a config value. Caches after first read."""
        if key in cls._cache:
            return cls._cache[key]

        row = fetch_one("SELECT value FROM config WHERE key = ?", [key])
        if row is None:
            return default

        value = row[0]
        # DuckDB returns JSON as string or dict depending on version
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass

        cls._cache[key] = value
        return value

    @classmethod
    def set(cls, key: str, value: Any, description: str | None = None) -> None:
        """Set a config value in the database."""
        execute(
            """
            INSERT INTO config (key, value, description, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                description = COALESCE(excluded.description, config.description),
                updated_at = datetime('now')
            """,
            [key, json.dumps(value), description],
        )
        cls._cache[key] = value

    @classmethod
    def refresh(cls, key: str | None = None) -> None:
        """Clear cache. If key given, clear only that key."""
        if key:
            cls._cache.pop(key, None)
        else:
            cls._cache.clear()

    # --- Typed Accessors ---

    @classmethod
    def generation(cls) -> dict:
        return cls.get("generation", {
            "model": "qwen2.5:14b",
            "max_retries": 3,
            "temperatures": [0.7, 0.85, 1.0],
            "min_char_count": 1200,
            "max_char_count": 3000,
            "optimal_min": 1400,
            "optimal_max": 2100,
            "min_hashtags": 4,
            "max_hashtags": 6,
        })

    @classmethod
    def evaluation(cls) -> dict:
        return cls.get("evaluation", {
            "min_relevance": 7.0,
            "min_quality": 7.0,
            "min_faithfulness": 6.0,
            "duplicate_threshold": 0.85,
        })

    @classmethod
    def banned_phrases(cls) -> list[str]:
        return cls.get("banned_phrases", [])

    @classmethod
    def skip_repos(cls) -> list[str]:
        return cls.get("skip_repos", [])

    @classmethod
    def pipeline(cls) -> dict:
        return cls.get("pipeline", {
            "max_posts_per_run": 10,
            "max_age_days": 7,
            "limit_per_source": 15,
            "concurrent_sources": True,
        })

    @classmethod
    def llm(cls) -> dict:
        return cls.get("llm", {
            "backend": "ollama",
            "ollama_url": "http://localhost:11434",
            "model": "qwen2.5:14b",
            "timeout": 120,
        })

    @classmethod
    def evaluation_v2(cls) -> dict:
        return cls.get("evaluation_v2", {
            "weights": {
                "answer_relevancy": 0.20,
                "faithfulness": 0.20,
                "hallucination": 0.15,
                "bias": 0.10,
                "toxicity": 0.10,
                "linkedin_quality": 0.25,
            },
            "thresholds": {
                "answer_relevancy": 7.0,
                "faithfulness": 7.0,
                "hallucination": 8.0,
                "bias": 7.0,
                "toxicity": 9.0,
                "linkedin_quality": 7.0,
            },
            "overall_pass_threshold": 7.0,
            "judge_temperature": 0.2,
        })

    @classmethod
    def feedback_config(cls) -> dict:
        return cls.get("feedback_config", {
            "decay_lambda": 0.02,
            "max_feedback_patterns": 10,
            "good_example_weight": 2.0,
        })

    @classmethod
    def hindsight(cls) -> dict:
        return cls.get("hindsight", {
            "url": "http://localhost:8888",
            "bank_id": "linkedin_writing_style",
            "reflect_budget": "mid",
            "recall_max_tokens": 4096,
            "timeout": 180,
        })

    @classmethod
    def linkedin_seo_keywords(cls) -> list[str]:
        return cls.get("linkedin_seo_keywords", [
            "AI", "Machine Learning", "Deep Learning", "LLM",
            "GenAI", "Generative AI", "Data Science", "MLOps",
            "AI Engineering", "NLP", "Computer Vision", "RAG",
            "Fine-tuning", "Prompt Engineering", "AI Agents",
            "Agentic AI", "Neural Networks", "Transformers",
        ])

    @classmethod
    def recruiter_keywords(cls) -> list[str]:
        return cls.get("recruiter_keywords", [
            "AI Engineer", "ML Engineer", "Data Scientist",
            "Machine Learning Engineer", "AI Researcher",
            "NLP Engineer", "MLOps Engineer", "Data Engineer",
            "Prompt Engineer", "AI Architect", "Deep Learning",
            "LLM", "RAG", "Fine-tuning", "Python", "PyTorch",
            "TensorFlow", "Hugging Face", "LangChain",
        ])

    @classmethod
    def hook_patterns(cls) -> list[str]:
        return cls.get("hook_patterns", [
            "pattern-interrupt", "curiosity-gap", "bold-statement",
            "story-hook", "contrarian", "question-hook",
        ])

    @classmethod
    def emoji_style(cls) -> str:
        return cls.get("emoji_style", "light")
