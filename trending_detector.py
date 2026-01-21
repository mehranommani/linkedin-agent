"""
Trending AI Topics Detector
============================
Fetches real-time trending topics from multiple sources to help
prioritize content that's currently hot in the AI community.

Usage:
    from trending_detector import TrendingDetector
    detector = TrendingDetector()
    hot_topics = detector.get_trending_topics()
"""

import re
import requests
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Set
from dataclasses import dataclass
import feedparser


@dataclass
class TrendingTopic:
    topic: str
    mentions: int
    sources: List[str]
    momentum: float  # How fast it's rising


class TrendingDetector:
    """Detects trending AI topics from multiple sources."""

    # High-signal sources for trend detection
    TREND_SOURCES = [
        # Hacker News (AI/ML)
        "https://hnrss.org/newest?q=AI+OR+LLM+OR+machine+learning",
        # Reddit (ML subreddit)
        "https://www.reddit.com/r/MachineLearning/hot/.rss",
        "https://www.reddit.com/r/LocalLLaMA/hot/.rss",
        # Product Hunt (AI category)
        "https://www.producthunt.com/feed?category=artificial-intelligence",
        # GitHub Trending (via RSS bridge if available)
        "https://rsshub.app/github/trending/daily/python?since=daily",
    ]

    # Known AI tool/library names to track
    TRACKABLE_ENTITIES = {
        # Models
        "gpt-4", "gpt-4o", "gpt-5", "claude", "claude-3", "claude-4",
        "llama", "llama-3", "mistral", "mixtral", "gemini", "gemma",
        "qwen", "phi-3", "deepseek", "grok", "command-r",

        # Frameworks/Libraries
        "langchain", "llamaindex", "crewai", "autogen", "dspy",
        "vllm", "ollama", "lmstudio", "openrouter", "groq",
        "instructor", "outlines", "guidance", "marvin",

        # Techniques
        "rag", "agentic", "mcp", "function calling", "tool use",
        "fine-tuning", "lora", "qlora", "dpo", "rlhf", "cot",
        "chain of thought", "reasoning", "multimodal",

        # Infrastructure
        "vector database", "pinecone", "weaviate", "qdrant", "chroma",
        "milvus", "pgvector", "faiss",
    }

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        }

    def fetch_recent_mentions(self) -> Counter:
        """Fetch and count mentions from trend sources."""
        mentions = Counter()

        for source_url in self.TREND_SOURCES:
            try:
                response = requests.get(
                    source_url,
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code != 200:
                    continue

                feed = feedparser.parse(response.content)

                for entry in feed.entries[:20]:
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    text = f"{title} {summary}"

                    # Count entity mentions
                    for entity in self.TRACKABLE_ENTITIES:
                        if entity in text:
                            mentions[entity] += 1

                    # Also extract potential new trending terms
                    self._extract_emerging_topics(text, mentions)

            except Exception:
                continue

        return mentions

    def _extract_emerging_topics(self, text: str, mentions: Counter):
        """Extract potential new trending topics from text."""
        # Pattern for new tools/libraries (often CamelCase or with version numbers)
        patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',  # CamelCase words
            r'\b([a-z]+[-_][a-z]+)\b',  # hyphenated/underscored names
            r'\b([A-Z]{2,})\b',  # Acronyms
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                match_lower = match.lower()
                # Filter out common non-AI terms
                if len(match) > 2 and match_lower not in {'the', 'and', 'for', 'are'}:
                    if any(kw in text for kw in ['ai', 'ml', 'model', 'llm', 'neural']):
                        mentions[match_lower] += 0.5  # Lower weight for unverified

    def get_trending_topics(self, top_n: int = 15) -> List[TrendingTopic]:
        """Get the current trending AI topics."""
        print("🔥 Detecting trending AI topics...")

        mentions = self.fetch_recent_mentions()

        # Get top topics
        trending = []
        for topic, count in mentions.most_common(top_n):
            trending.append(TrendingTopic(
                topic=topic,
                mentions=int(count),
                sources=[],  # Could track which sources mentioned it
                momentum=count / 10.0  # Simplified momentum score
            ))

        return trending

    def boost_relevance_score(
        self,
        title: str,
        summary: str,
        base_score: float,
        trending_topics: List[TrendingTopic]
    ) -> float:
        """Boost relevance score if content matches trending topics."""
        text = f"{title} {summary}".lower()
        boost = 0.0

        for topic in trending_topics:
            if topic.topic in text:
                # More mentions = higher boost
                boost += min(topic.mentions * 0.2, 2.0)

        return base_score + boost


def get_trending_keywords() -> Dict[str, float]:
    """
    Quick function to get trending keywords with their weights.
    Can be imported and used to dynamically update TOPIC_WEIGHTS.
    """
    detector = TrendingDetector()
    trending = detector.get_trending_topics(top_n=20)

    weights = {}
    for topic in trending:
        # Convert mentions to a weight between 1.0 and 2.0
        weight = min(1.0 + (topic.mentions * 0.1), 2.0)
        weights[topic.topic] = weight

    return weights


if __name__ == "__main__":
    # Test the detector
    detector = TrendingDetector()
    topics = detector.get_trending_topics()

    print("\n📈 Current Trending AI Topics:")
    print("-" * 40)
    for i, topic in enumerate(topics, 1):
        print(f"{i:2}. {topic.topic:<20} ({topic.mentions} mentions)")
