"""
Seed Config & Sources
=====================
Populates the config and sources tables with initial data
migrated from the hardcoded values in the old agent.py.

Usage: python scripts/seed_config.py
"""

import sys
import json
import uuid
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import get_connection


def _id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:12]


def seed_sources(conn):
    """Migrate all hardcoded sources into the sources table."""

    existing = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    if existing > 0:
        print(f"  Sources table already has {existing} rows. Skipping.")
        return

    sources = []

    # --- RSS Feeds ---
    rss_feeds = [
        # AI Labs (priority 1)
        ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "ai_lab", 1),
        ("OpenAI Blog", "https://openai.com/blog/rss.xml", "ai_lab", 1),
        ("Anthropic Blog", "https://www.anthropic.com/rss.xml", "ai_lab", 1),
        ("Google AI Blog", "https://ai.googleblog.com/feeds/posts/default", "ai_lab", 1),
        ("DeepMind Blog", "https://deepmind.google/blog/rss.xml", "ai_lab", 1),
        ("NVIDIA AI Blog", "https://blogs.nvidia.com/feed/", "ai_lab", 1),
        ("Meta AI Blog", "https://ai.meta.com/blog/rss/", "ai_lab", 1),

        # AI Startups (priority 2)
        ("Stability AI", "https://stability.ai/blog/rss.xml", "startup", 2),
        ("Cohere Blog", "https://cohere.com/blog/rss.xml", "startup", 2),
        ("Mistral AI", "https://mistral.ai/feed.xml", "startup", 2),
        ("Replicate Blog", "https://replicate.com/blog/rss.xml", "startup", 2),
        ("Modal Blog", "https://modal.com/blog/rss.xml", "startup", 2),
        ("Together AI", "https://www.together.ai/blog/rss.xml", "startup", 2),
        ("Groq Blog", "https://wow.groq.com/feed/", "startup", 2),
        ("Perplexity Blog", "https://www.perplexity.ai/blog/rss.xml", "startup", 2),
        ("Anyscale Blog", "https://www.anyscale.com/blog/rss.xml", "startup", 2),
        ("W&B Blog", "https://www.weights-biases.com/blog/rss", "startup", 2),
        ("LangChain Blog", "https://www.langchain.com/blog/rss.xml", "startup", 2),
        ("LlamaIndex Blog", "https://blog.llamaindex.ai/feed", "startup", 2),
        ("Unsloth Blog", "https://unsloth.ai/blog/rss.xml", "startup", 2),

        # Cloud Platforms (priority 3)
        ("AWS ML Blog", "https://aws.amazon.com/blogs/machine-learning/feed/", "cloud", 3),
        ("Google Cloud AI", "https://cloud.google.com/blog/products/ai-machine-learning/rss", "cloud", 3),
        ("Azure AI Blog", "https://azure.microsoft.com/en-us/blog/tag/ai/feed/", "cloud", 3),
        ("Databricks AI", "https://www.databricks.com/blog/category/artificial-intelligence/feed", "cloud", 3),

        # Tech News (priority 4)
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "news", 4),
        ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "news", 4),
        ("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss", "news", 4),
        ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "news", 4),
        ("Ars Technica AI", "https://arstechnica.com/tag/artificial-intelligence/feed/", "news", 4),
        ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", "news", 4),
        ("MarkTechPost", "https://www.marktechpost.com/feed/", "news", 4),
        ("The Decoder", "https://the-decoder.com/feed/", "news", 4),

        # Research (priority 2)
        ("BAIR Blog", "https://bair.berkeley.edu/blog/feed.xml", "research", 2),
        ("Lilian Weng", "https://lilianweng.github.io/index.xml", "research", 2),
        ("Karpathy Blog", "https://karpathy.github.io/feed.xml", "research", 2),
        ("Colah Blog", "https://colah.github.io/rss.xml", "research", 2),
        ("Distill", "https://distill.pub/rss.xml", "research", 2),
        ("Stanford AI", "https://ai.stanford.edu/blog/feed.xml", "research", 2),
        ("Allen AI", "https://blog.allenai.org/feed", "research", 2),
        ("EleutherAI", "https://www.eleuther.ai/blog/rss.xml", "research", 2),

        # Developer (priority 3)
        ("Meta Engineering", "https://engineering.fb.com/feed/", "developer", 3),
        ("Netflix Tech", "https://netflixtechblog.com/feed", "developer", 3),
        ("Google AI Blog", "https://blog.google/technology/ai/rss/", "developer", 3),
        ("GitHub Blog", "https://github.blog/feed/", "developer", 3),
        ("Sourcegraph Blog", "https://sourcegraph.com/blog/rss.xml", "developer", 3),

        # Newsletters (priority 3)
        ("DeepLearning.ai Batch", "https://www.deeplearning.ai/the-batch/feed/", "newsletter", 3),
        ("Import AI", "https://jack-clark.net/feed/", "newsletter", 3),
        ("Last Week in AI", "https://lastweekin.ai/feed", "newsletter", 3),
        ("The Rundown AI", "https://therundownai.com/feed", "newsletter", 3),
        ("AI Weekly", "https://www.aiweekly.co/feed", "newsletter", 3),
    ]

    for name, url, category, priority in rss_feeds:
        sources.append({
            "id": _id(f"rss-{url}"),
            "source_type": "rss",
            "name": name,
            "url": url,
            "category": category,
            "params": json.dumps({"feed_url": url}),
            "priority": priority,
        })

    # --- Reddit Subreddits ---
    subreddits = [
        ("LocalLLaMA", "ai_community", 2),
        ("MachineLearning", "research", 2),
        ("ClaudeAI", "ai_community", 3),
        ("ChatGPT", "ai_community", 3),
        ("ollama", "ai_community", 3),
        ("AgenticAI", "ai_community", 2),
        ("ArtificialIntelligence", "ai_community", 3),
        ("GPT4All", "ai_community", 4),
        ("MistralAI", "ai_community", 4),
    ]

    for sub, category, priority in subreddits:
        sources.append({
            "id": _id(f"reddit-{sub}"),
            "source_type": "reddit",
            "name": f"r/{sub}",
            "url": f"https://www.reddit.com/r/{sub}/",
            "category": category,
            "params": json.dumps({"subreddit": sub, "min_score": 100}),
            "priority": priority,
        })

    # --- GitHub ---
    sources.append({
        "id": _id("github-trending"),
        "source_type": "github",
        "name": "GitHub Trending",
        "url": "https://github.com/trending",
        "category": "developer",
        "params": json.dumps({
            "languages": ["python", "jupyter-notebook"],
            "since": "weekly",
        }),
        "priority": 1,
    })
    sources.append({
        "id": _id("github-search"),
        "source_type": "github",
        "name": "GitHub Search (AI/ML)",
        "url": "https://api.github.com/search/repositories",
        "category": "developer",
        "params": json.dumps({
            "queries": [
                "language:python topic:machine-learning",
                "language:python topic:llm",
                "language:python topic:generative-ai",
                "language:python topic:deep-learning",
                "language:jupyter-notebook topic:data-science",
            ]
        }),
        "priority": 2,
    })

    # --- Hacker News ---
    sources.append({
        "id": _id("hackernews"),
        "source_type": "hackernews",
        "name": "Hacker News Top",
        "url": "https://hacker-news.firebaseio.com/v0/topstories.json",
        "category": "news",
        "params": json.dumps({"min_score": 50, "limit": 15}),
        "priority": 2,
    })

    # --- Product Hunt ---
    sources.append({
        "id": _id("producthunt"),
        "source_type": "producthunt",
        "name": "Product Hunt",
        "url": "https://www.producthunt.com",
        "category": "startup",
        "params": json.dumps({"limit": 10}),
        "priority": 4,
    })

    # --- Arxiv ---
    sources.append({
        "id": _id("arxiv-ai"),
        "source_type": "arxiv",
        "name": "Arxiv AI/ML",
        "url": "https://arxiv.org",
        "category": "research",
        "params": json.dumps({
            "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
            "max_age_days": 7,
        }),
        "priority": 2,
    })

    # --- Dev.to ---
    sources.append({
        "id": _id("devto"),
        "source_type": "devto",
        "name": "Dev.to AI/ML",
        "url": "https://dev.to",
        "category": "developer",
        "params": json.dumps({"tags": ["ai", "machinelearning", "llm", "python"]}),
        "priority": 4,
    })

    # --- Papers with Code ---
    sources.append({
        "id": _id("papers-with-code"),
        "source_type": "papers",
        "name": "Papers with Code",
        "url": "https://paperswithcode.com",
        "category": "research",
        "params": json.dumps({"limit": 10}),
        "priority": 3,
    })

    # Insert all sources
    for src in sources:
        conn.execute(
            """
            INSERT INTO sources (id, source_type, name, url, category, params, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            [src["id"], src["source_type"], src["name"], src["url"],
             src["category"], src["params"], src["priority"]],
        )

    count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    print(f"  Seeded {count} sources")


def seed_config(conn):
    """Seed the config table with initial settings."""

    existing = conn.execute("SELECT COUNT(*) FROM config").fetchone()[0]
    print(f"  Config table has {existing} existing rows. Upserting...")

    configs = [
        (
            "generation",
            {
                "model": "qwen2.5:7b",
                "max_retries": 3,
                "temperatures": [0.7, 0.85, 1.0],
                "min_char_count": 1200,
                "max_char_count": 3000,
                "optimal_min": 1400,
                "optimal_max": 2100,
                "min_hashtags": 4,
                "max_hashtags": 6,
            },
            "Post generation settings",
        ),
        (
            "evaluation",
            {
                "min_relevance": 7.0,
                "min_quality": 7.0,
                "min_faithfulness": 6.0,
                "duplicate_threshold": 0.85,
            },
            "Quality gate thresholds (v1 — legacy)",
        ),
        (
            "evaluation_v2",
            {
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
            },
            "Six-metric evaluation engine (v2 — DeepEval/RAGAS equivalent)",
        ),
        (
            "banned_phrases",
            [
                "imagine a world", "imagine an era", "picture this",
                "in today's fast-paced", "in today's world", "in the ever-evolving",
                "ever wondered", "have you ever", "just came across", "just found",
                "stumbled upon", "excited to share", "happy to announce", "i'm thrilled",
                "[hook]", "[cta]", "[insert", "game changer", "game-changer", "revolutionary",
            ],
            "Phrases to avoid in generated posts",
        ),
        (
            "skip_repos",
            ["langchain", "dify", "ollama", "transformers", "pytorch",
             "tensorflow", "autogpt", "gpt4all", "llama", "whisper"],
            "GitHub repos to skip (too generic/well-known)",
        ),
        (
            "pipeline",
            {
                "max_posts_per_run": 10,
                "max_age_days": 7,
                "limit_per_source": 15,
                "concurrent_sources": True,
            },
            "Pipeline run defaults",
        ),
        (
            "llm",
            {
                "backend": "ollama",
                "ollama_url": "http://localhost:11434",
                "model": "qwen2.5:7b",
                "timeout": 180,
            },
            "LLM backend configuration",
        ),
        (
            "linkedin_seo_keywords",
            [
                "AI", "Machine Learning", "Deep Learning", "LLM",
                "GenAI", "Generative AI", "Data Science", "MLOps",
                "AI Engineering", "NLP", "Computer Vision", "RAG",
                "Fine-tuning", "Prompt Engineering", "AI Agents",
                "Agentic AI", "Neural Networks", "Transformers",
            ],
            "SEO keywords to weave into LinkedIn posts",
        ),
        (
            "recruiter_keywords",
            [
                "AI Engineer", "ML Engineer", "Data Scientist",
                "Machine Learning Engineer", "AI Researcher",
                "NLP Engineer", "MLOps Engineer", "Data Engineer",
                "Prompt Engineer", "AI Architect", "Deep Learning",
                "LLM", "RAG", "Fine-tuning", "Python", "PyTorch",
                "TensorFlow", "Hugging Face", "LangChain",
            ],
            "Recruiter-targeted keywords for LinkedIn search visibility",
        ),
        (
            "hook_patterns",
            [
                "pattern-interrupt", "curiosity-gap", "bold-statement",
                "story-hook", "contrarian", "question-hook",
            ],
            "Enabled hook patterns for post generation",
        ),
        (
            "emoji_style",
            "light",
            "Emoji usage style: light (1-3), moderate (3-5), none",
        ),
        (
            "feedback_config",
            {
                "decay_lambda": 0.02,
                "max_feedback_patterns": 10,
                "good_example_weight": 2.0,
            },
            "Human feedback loop and anti-overfitting settings",
        ),
        (
            "hindsight",
            {
                "url": "http://localhost:8888",
                "bank_id": "linkedin_writing_style",
                "reflect_budget": "mid",
                "recall_max_tokens": 4096,
                "timeout": 180,
            },
            "Hindsight agent memory configuration",
        ),
    ]

    for key, value, desc in configs:
        conn.execute(
            """
            INSERT INTO config (key, value, description, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            [key, json.dumps(value), desc],
        )

    count = conn.execute("SELECT COUNT(*) FROM config").fetchone()[0]
    print(f"  Seeded {count} config entries")


def main():
    print("Seeding database...")
    conn = get_connection()
    seed_sources(conn)
    seed_config(conn)
    print("\nDone!")

    # Summary
    print("\nDatabase summary:")
    for table in ["posts", "sources", "config", "evaluations", "pipeline_runs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")


if __name__ == "__main__":
    main()
