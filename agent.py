"""
LinkedIn AI Content Agent v2
============================
Clean, efficient multi-source content agent.
Fetches from RSS, GitHub, Product Hunt, Hacker News, Reddit.
Generates varied, natural LinkedIn posts.

Run: python agent_v2.py
"""

import os
import re
import time
import hashlib
import pandas as pd
import requests
import feedparser
import html2text
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Set, Optional
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linkedin_content_bank.csv")
MODEL_NAME = "qwen2.5:14b"
MAX_RETRIES = 2

# ============================================================
# UNIFIED CONTENT MODEL
# ============================================================

@dataclass
class ContentItem:
    """Universal content item from any source."""
    title: str
    url: str
    source: str  # rss, github, producthunt, hackernews, reddit
    summary: str = ""
    metrics: dict = field(default_factory=dict)  # stars, votes, points, score, etc.
    image_url: str = ""
    timestamp: float = 0.0  # Unix timestamp for freshness calculation

    @property
    def score(self) -> float:
        """Calculate relevance score based on metrics, source, and freshness."""
        base = 5.0

        # 1. Metric-based scoring (ADJUSTED THRESHOLDS - more generous)
        if 'stars' in self.metrics:
            # GitHub: 1000 stars = +1 point (was 5000)
            base += min(5.0, self.metrics['stars'] / 1000)
        if 'votes' in self.metrics:
            base += min(3.0, self.metrics['votes'] / 100)
        if 'points' in self.metrics:
            # HN: 50 points = +1 point (was 200)
            base += min(4.0, self.metrics['points'] / 50)
        if 'upvotes' in self.metrics:
            # Reddit: 100 upvotes = +1 point (was 500)
            base += min(3.0, self.metrics['upvotes'] / 100)

        # 2. RSS BOOST - curated sources get bonus
        if self.source == "rss":
            base += 1.5

        # 3. FRESHNESS BOOST - newer content scores higher
        if self.timestamp > 0:
            age_hours = (datetime.now().timestamp() - self.timestamp) / 3600
            if age_hours < 24:
                base += 1.0  # Very fresh: < 24 hours
            elif age_hours < 48:
                base += 0.5  # Fresh: < 48 hours

        return min(12.0, base)  # Increased cap to accommodate boosts

# ============================================================
# MULTI-SOURCE SCOUT
# ============================================================

class ContentScout:
    """Unified scout for all content sources."""

    AI_KEYWORDS = [
        # Core AI/ML
        'ai', 'llm', 'machine learning', 'deep learning', 'neural', 'artificial intelligence',
        'ml', 'mlops', 'automl', 'tensorflow', 'pytorch', 'keras', 'scikit', 'sklearn',

        # LLMs & Models
        'gpt', 'gpt-4', 'gpt4', 'gpt-3', 'gpt-3.5', 'gpt-neo', 'gpt-neox', 'gptj',
        'claude', 'claude2', 'anthropic', 'openai', 'gemini', 'bard', 'grok',
        'llama', 'mistral', 'bloom', 'falcon', 'phi-', 'qwen', 'deepseek',
        'ollama', 'huggingface', 'transformers', 'tokenizer',

        # GenAI & Diffusion
        'generative', 'genai', 'diffusion', 'stable diffusion', 'dalle', 'midjourney',
        'text-to-image', 'image generation', 'whisper', 'speech-to-text', 'tts',

        # Agentic AI & RAG
        'agentic', 'agent', 'multi-agent', 'autonomous agent', 'ai agent',
        'rag', 'retrieval augmented', 'langchain', 'langgraph', 'langsmith',
        'llamaindex', 'crewai', 'autogen', 'dify', 'flowise',
        'mcp', 'model context protocol', 'tool use', 'function calling',

        # Vector & Embeddings
        'embedding', 'vector', 'vector db', 'vector database', 'vectorstore',
        'pinecone', 'weaviate', 'chroma', 'faiss', 'milvus', 'qdrant', 'vectara', 'lancedb',
        'semantic search', 'similarity search', 'contextual search',

        # NLP & Text
        'nlp', 'natural language', 'chatbot', 'conversational', 'copilot',
        'prompt engineering', 'zero-shot', 'few-shot', 'fine-tun', 'lora', 'qlora',
        'instruction tuning', 'rlhf', 'dpo',

        # Data Science
        'data science', 'datascience', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'jupyter', 'notebook', 'analytics', 'data analysis', 'visualization',
        'statistics', 'predictive', 'forecasting', 'time series',

        # Computer Vision
        'computer vision', 'cv', 'object detection', 'image classification',
        'yolo', 'opencv', 'segmentation', 'ocr',

        # MLOps & Infrastructure
        'model serving', 'inference', 'triton', 'vllm', 'tgi',
        'weights', 'checkpoint', 'quantization', 'gguf', 'ggml',

        # Research & Benchmarks
        'benchmark', 'evaluation', 'leaderboard', 'arxiv', 'paper'
    ]

    SKIP_REPOS = {'langchain', 'dify', 'ollama', 'transformers', 'pytorch',
                  'tensorflow', 'autogpt', 'gpt4all', 'llama', 'whisper'}

    def __init__(self, max_age_days: int = 7):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        self.seen = set()
        self.max_age_days = max_age_days
        self.cutoff_date = datetime.now() - timedelta(days=max_age_days)

    def _is_recent(self, timestamp: float = None, date_struct: tuple = None) -> bool:
        """Check if content is within the max_age_days window."""
        try:
            if timestamp:
                content_date = datetime.fromtimestamp(timestamp)
            elif date_struct:
                content_date = datetime(*date_struct[:6])
            else:
                return True  # If no date info, include it
            return content_date >= self.cutoff_date
        except:
            return True  # On error, include it

    def _is_ai_related(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.AI_KEYWORDS)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def fetch_all(self, seen_links: Set[str], limit_per_source: int = 5, sources: List[str] = None) -> List[ContentItem]:
        """Fetch from specified sources (filtered by date).

        Args:
            seen_links: Set of already processed URLs
            limit_per_source: Max items per source
            sources: List of source names to fetch from. If None, fetch from all.
                     Valid: github, hackernews, reddit, producthunt, papers, arxiv, devto, rss
        """
        items = []

        # Default to all sources if none specified
        all_sources = ['github', 'hackernews', 'reddit', 'producthunt', 'papers', 'arxiv', 'devto', 'rss']
        active_sources = sources if sources else all_sources

        print(f"\n📡 Gathering hot topics from last {self.max_age_days} days...")
        print(f"   Active sources: {', '.join(active_sources)}")

        # GitHub Trending
        if 'github' in active_sources:
            items.extend(self._fetch_github(limit_per_source))

        # Hacker News
        if 'hackernews' in active_sources:
            items.extend(self._fetch_hackernews(limit_per_source))

        # Reddit
        if 'reddit' in active_sources:
            items.extend(self._fetch_reddit(limit_per_source))

        # Product Hunt AI
        if 'producthunt' in active_sources:
            items.extend(self._fetch_producthunt(limit_per_source))

        # Papers with Code
        if 'papers' in active_sources:
            items.extend(self._fetch_papers_with_code(limit_per_source))

        # Arxiv AI/ML
        if 'arxiv' in active_sources:
            items.extend(self._fetch_arxiv(limit_per_source))

        # Dev.to AI articles
        if 'devto' in active_sources:
            items.extend(self._fetch_devto(limit_per_source))

        # RSS Feeds (expanded)
        if 'rss' in active_sources:
            items.extend(self._fetch_rss(seen_links, limit_per_source))

        # Filter duplicates and already seen
        unique = []
        for item in items:
            h = self._hash(item.title)
            if h not in self.seen and item.url not in seen_links:
                self.seen.add(h)
                unique.append(item)

        # Sort by score
        unique.sort(key=lambda x: x.score, reverse=True)

        print(f"📊 Found {len(unique)} unique items")
        return unique

    def _fetch_github(self, limit: int) -> List[ContentItem]:
        """Fetch trending GitHub repos from multiple languages."""
        items = []
        seen_repos = set()  # Avoid duplicates across languages
        try:
            # Map max_age_days to GitHub's since parameter
            if self.max_age_days <= 1:
                since = "daily"
            elif self.max_age_days <= 7:
                since = "weekly"
            else:
                since = "monthly"

            # Fetch from Python and Jupyter Notebook trending
            languages = ["python", "jupyter-notebook"]

            for lang in languages:
                url = f"https://github.com/trending/{lang}?since={since}"
                resp = requests.get(url, headers=self.headers, timeout=15)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.content, 'lxml')
                for article in soup.find_all('article', class_='Box-row')[:25]:
                    try:
                        h2 = article.find('h2')
                        if not h2:
                            continue
                        link = h2.find('a')
                        if not link:
                            continue

                        full_name = link.get('href', '').strip('/')
                        name = full_name.split('/')[-1].lower()

                        # Skip duplicates across languages
                        if full_name in seen_repos:
                            continue
                        seen_repos.add(full_name)

                        # Skip famous repos
                        if any(skip in name for skip in self.SKIP_REPOS):
                            continue

                        desc = article.find('p', class_='col-9')
                        description = desc.get_text(strip=True) if desc else ""

                        if not self._is_ai_related(f"{name} {description}"):
                            continue

                        # Get stars
                        stars_elem = article.find('a', href=lambda x: x and '/stargazers' in x)
                        stars = 0
                        if stars_elem:
                            stars_text = stars_elem.get_text(strip=True).replace(',', '')
                            stars = int(stars_text) if stars_text.isdigit() else 0

                        # Skip if too famous (>50k stars)
                        if stars > 50000:
                            continue

                        items.append(ContentItem(
                            title=f"{full_name}: {description[:80]}",
                            url=f"https://github.com/{full_name}",
                            source="github",
                            summary=description,
                            metrics={'stars': stars},
                            image_url=f"https://opengraph.githubassets.com/1/{full_name}",
                            timestamp=datetime.now().timestamp()
                        ))
                    except:
                        continue

            print(f"   🐙 GitHub ({since}): {len(items)} repos")
        except Exception as e:
            print(f"   ⚠️ GitHub error: {e}")
        return items[:limit]

    def _fetch_hackernews(self, limit: int) -> List[ContentItem]:
        """Fetch AI stories from Hacker News (filtered by date)."""
        items = []
        try:
            for endpoint in ['topstories', 'showstories']:
                resp = requests.get(f"https://hacker-news.firebaseio.com/v0/{endpoint}.json", timeout=10)
                if resp.status_code != 200:
                    continue

                for story_id in resp.json()[:50]:  # Check more to filter by date
                    try:
                        item_resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=5)
                        if item_resp.status_code != 200:
                            continue

                        data = item_resp.json()
                        if not data or data.get('type') != 'story':
                            continue

                        # Filter by date (HN uses Unix timestamp)
                        if not self._is_recent(timestamp=data.get('time')):
                            continue

                        title = data.get('title', '')
                        if not self._is_ai_related(title):
                            continue

                        items.append(ContentItem(
                            title=title,
                            url=data.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                            source="hackernews",
                            summary=f"HN Discussion: {data.get('descendants', 0)} comments",
                            metrics={'points': data.get('score', 0), 'comments': data.get('descendants', 0)},
                            timestamp=data.get('time', 0)
                        ))

                        if len(items) >= limit * 2:
                            break
                    except:
                        continue

            print(f"   📰 Hacker News: {len(items)} stories")
        except Exception as e:
            print(f"   ⚠️ HN error: {e}")
        return items[:limit]

    def _fetch_reddit(self, limit: int) -> List[ContentItem]:
        """Fetch hot posts from AI subreddits (filtered by date)."""
        items = []
        subreddits = ['LocalLLaMA', 'MachineLearning', 'ClaudeAI', 'ChatGPT', 'ollama', 'AgeniticsAI', 'ArtificialIntelligence', 'GPT4All', 'MistralAI']

        try:
            for sub in subreddits:
                try:
                    resp = requests.get(
                        f"https://www.reddit.com/r/{sub}/hot.json?limit=20",  # Get more to filter
                        headers={'User-Agent': 'LinkedInAgent/1.0'},
                        timeout=10
                    )
                    if resp.status_code != 200:
                        continue

                    for child in resp.json().get('data', {}).get('children', []):
                        post = child.get('data', {})
                        if post.get('stickied') or post.get('score', 0) < 100:
                            continue

                        # Filter by date (Reddit uses created_utc)
                        if not self._is_recent(timestamp=post.get('created_utc')):
                            continue

                        items.append(ContentItem(
                            title=post.get('title', ''),
                            url=f"https://reddit.com{post.get('permalink', '')}",
                            source="reddit",
                            summary=f"r/{sub} • {post.get('num_comments', 0)} comments",
                            metrics={'upvotes': post.get('score', 0), 'comments': post.get('num_comments', 0)},
                            timestamp=post.get('created_utc', 0)
                        ))
                except:
                    continue

            print(f"   🔴 Reddit: {len(items)} posts")
        except Exception as e:
            print(f"   ⚠️ Reddit error: {e}")
        return items[:limit]

    def _fetch_producthunt(self, limit: int) -> List[ContentItem]:
        """Fetch AI products from Product Hunt."""
        items = []
        try:
            # Product Hunt doesn't have a public API, use their RSS/feed
            resp = requests.get(
                "https://www.producthunt.com/feed?category=artificial-intelligence",
                headers=self.headers,
                timeout=15
            )
            if resp.status_code == 200:
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:15]:
                    title = entry.get('title', '')
                    if not self._is_ai_related(title):
                        continue

                    date_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                    if not self._is_recent(date_struct=date_struct):
                        continue

                    ts = 0.0
                    if date_struct:
                        try:
                            ts = datetime(*date_struct[:6]).timestamp()
                        except:
                            pass

                    items.append(ContentItem(
                        title=title,
                        url=entry.get('link', ''),
                        source="producthunt",
                        summary=entry.get('summary', '')[:300],
                        metrics={'votes': 100},  # PH doesn't expose votes in RSS
                        timestamp=ts
                    ))

            print(f"   🚀 Product Hunt: {len(items)} products")
        except Exception as e:
            print(f"   ⚠️ Product Hunt error: {e}")
        return items[:limit]

    def _fetch_papers_with_code(self, limit: int) -> List[ContentItem]:
        """Fetch trending papers from Papers with Code via web scraping."""
        items = []
        try:
            # Scrape the trending page directly
            resp = requests.get(
                "https://paperswithcode.com/",
                headers=self.headers,
                timeout=15
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find paper cards on the homepage
                paper_cards = soup.select('.paper-card, .row.infinite-item')[:20]

                for card in paper_cards:
                    # Get title
                    title_elem = card.select_one('h1 a, .item-content h1 a, .paper-card-title a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    href = title_elem.get('href', '')

                    if not title or not href:
                        continue

                    # Build full URL
                    paper_url = f"https://paperswithcode.com{href}" if href.startswith('/') else href

                    # Get abstract/summary if available
                    abstract_elem = card.select_one('.item-strip-abstract, .paper-abstract')
                    abstract = abstract_elem.get_text(strip=True)[:400] if abstract_elem else ''

                    if not self._is_ai_related(f"{title} {abstract}"):
                        continue

                    items.append(ContentItem(
                        title=title,
                        url=paper_url,
                        source="papers",
                        summary=abstract,
                        metrics={},
                        timestamp=datetime.now().timestamp()
                    ))

            print(f"   📄 Papers with Code: {len(items)} papers")
        except Exception as e:
            print(f"   ⚠️ Papers with Code error: {e}")
        return items[:limit]

    def _fetch_arxiv(self, limit: int) -> List[ContentItem]:
        """Fetch latest AI/ML papers from Arxiv."""
        items = []
        try:
            # Arxiv API for cs.AI and cs.LG categories
            categories = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
            resp = requests.get(
                f"http://export.arxiv.org/api/query?search_query={categories}&sortBy=submittedDate&sortOrder=descending&max_results=30",
                headers=self.headers,
                timeout=15
            )
            if resp.status_code == 200:
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:20]:
                    title = entry.get('title', '').replace('\n', ' ')
                    summary = entry.get('summary', '').replace('\n', ' ')

                    # Check relevance
                    if not self._is_ai_related(f"{title} {summary}"):
                        continue

                    # Check date
                    date_struct = entry.get('published_parsed')
                    if not self._is_recent(date_struct=date_struct):
                        continue

                    ts = 0.0
                    if date_struct:
                        try:
                            ts = datetime(*date_struct[:6]).timestamp()
                        except:
                            pass

                    items.append(ContentItem(
                        title=title,
                        url=entry.get('link', ''),
                        source="arxiv",
                        summary=summary[:400],
                        metrics={},
                        timestamp=ts
                    ))

            print(f"   🔬 Arxiv: {len(items)} papers")
        except Exception as e:
            print(f"   ⚠️ Arxiv error: {e}")
        return items[:limit]

    def _fetch_devto(self, limit: int) -> List[ContentItem]:
        """Fetch AI/ML articles from Dev.to."""
        items = []
        try:
            # Dev.to API for AI-related articles
            for tag in ['ai', 'machinelearning', 'llm', 'openai', 'chatgpt']:
                resp = requests.get(
                    f"https://dev.to/api/articles?tag={tag}&top=7&per_page=10",
                    headers=self.headers,
                    timeout=10
                )
                if resp.status_code != 200:
                    continue

                for article in resp.json()[:5]:
                    title = article.get('title', '')

                    # Check date
                    pub_date = article.get('published_at')
                    if pub_date:
                        try:
                            article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            if article_date.replace(tzinfo=None) < self.cutoff_date:
                                continue
                            ts = article_date.timestamp()
                        except:
                            ts = datetime.now().timestamp()
                    else:
                        ts = datetime.now().timestamp()

                    items.append(ContentItem(
                        title=title,
                        url=article.get('url', ''),
                        source="devto",
                        summary=article.get('description', '')[:300],
                        metrics={'reactions': article.get('positive_reactions_count', 0)},
                        timestamp=ts
                    ))

            # Dedupe by URL
            seen_urls = set()
            unique_items = []
            for item in items:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    unique_items.append(item)
            items = unique_items

            print(f"   📝 Dev.to: {len(items)} articles")
        except Exception as e:
            print(f"   ⚠️ Dev.to error: {e}")
        return items[:limit]

    def _fetch_rss(self, seen_links: Set[str], limit: int) -> List[ContentItem]:
        """Fetch from RSS feeds (filtered by date)."""
        items = []
        feeds = [
            # === PRIMARY AI LABS (Highest Priority) ===
            "https://huggingface.co/blog/feed.xml",
            "https://openai.com/blog/rss.xml",
            "https://www.anthropic.com/rss.xml",
            "https://ai.googleblog.com/feeds/posts/default",
            "https://deepmind.google/blog/rss.xml",
            "https://blogs.nvidia.com/feed/",
            "https://ai.meta.com/blog/rss/",

            # === AI STARTUP BLOGS (Tool & Library Announcements) ===
            "https://stability.ai/blog/rss.xml",
            "https://cohere.com/blog/rss.xml",
            "https://mistral.ai/feed.xml",
            "https://replicate.com/blog/rss.xml",
            "https://modal.com/blog/rss.xml",
            "https://www.together.ai/blog/rss.xml",
            "https://wow.groq.com/feed/",
            "https://www.perplexity.ai/blog/rss.xml",
            "https://www.anyscale.com/blog/rss.xml",
            "https://www.weights-biases.com/blog/rss",
            "https://www.langchain.com/blog/rss.xml",
            "https://blog.llamaindex.ai/feed",
            "https://unsloth.ai/blog/rss.xml",

            # === CLOUD AI PLATFORMS ===
            "https://aws.amazon.com/blogs/machine-learning/feed/",
            "https://cloud.google.com/blog/products/ai-machine-learning/rss",
            "https://azure.microsoft.com/en-us/blog/tag/ai/feed/",
            "https://www.databricks.com/blog/category/artificial-intelligence/feed",

            # === TECH NEWS (AI Focus) ===
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://venturebeat.com/category/ai/feed/",
            "https://www.wired.com/feed/tag/ai/latest/rss",
            "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
            "https://arstechnica.com/tag/artificial-intelligence/feed/",
            "https://www.technologyreview.com/topic/artificial-intelligence/feed",
            "https://www.marktechpost.com/feed/",
            "https://the-decoder.com/feed/",

            # === RESEARCH & ACADEMIC ===
            "https://bair.berkeley.edu/blog/feed.xml",
            "https://lilianweng.github.io/index.xml",
            "https://karpathy.github.io/feed.xml",
            "https://colah.github.io/rss.xml",
            "https://distill.pub/rss.xml",
            "https://ai.stanford.edu/blog/feed.xml",
            "https://blog.allenai.org/feed",
            "https://www.eleuther.ai/blog/rss.xml",

            # === DEVELOPER & ENGINEERING ===
            "https://engineering.fb.com/feed/",
            "https://netflixtechblog.com/feed",
            "https://blog.google/technology/ai/rss/",
            "https://github.blog/feed/",
            "https://sourcegraph.com/blog/rss.xml",

            # === NEWSLETTERS & CURATED ===
            "https://www.deeplearning.ai/the-batch/feed/",
            "https://jack-clark.net/feed/",
            "https://lastweekin.ai/feed",
            "https://therundownai.com/feed",
            "https://www.aiweekly.co/feed",
        ]

        try:
            for feed_url in feeds:
                try:
                    resp = requests.get(feed_url, headers=self.headers, timeout=10)
                    if resp.status_code != 200:
                        continue

                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries[:10]:  # Check more to filter by date
                        if entry.link in seen_links:
                            continue

                        # Filter by date (RSS uses published_parsed or updated_parsed)
                        date_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                        if not self._is_recent(date_struct=date_struct):
                            continue

                        title = entry.get('title', '')
                        summary = entry.get('summary', '')

                        if not self._is_ai_related(f"{title} {summary}"):
                            continue

                        # Convert date_struct to Unix timestamp
                        ts = 0.0
                        if date_struct:
                            try:
                                ts = datetime(*date_struct[:6]).timestamp()
                            except:
                                pass

                        items.append(ContentItem(
                            title=title,
                            url=entry.link,
                            source="rss",
                            summary=summary[:300],
                            timestamp=ts
                        ))
                except:
                    continue

            print(f"   📰 RSS: {len(items)} articles")
        except Exception as e:
            print(f"   ⚠️ RSS error: {e}")
        return items[:limit]

# ============================================================
# CONTENT EXTRACTOR
# ============================================================

class ContentExtractor:
    """Extracts article content for fact-grounding."""

    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def extract(self, url: str) -> Optional[str]:
        """Extract main content from URL."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.content, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                tag.decompose()

            content = soup.find('article') or soup.find('main') or soup.body
            if not content:
                return None

            converter = html2text.HTML2Text()
            converter.ignore_links = False
            text = converter.handle(str(content))
            return text[:10000]
        except:
            return None

# ============================================================
# SMART POST WRITER - LinkedIn SEO Optimized Template
# ============================================================

class SmartWriter:
    """Generates LinkedIn posts using the Insight & SEO template."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatOllama(model=model_name, temperature=0.85)
        self.verifier = PostVerifier()

    def write(self, item: ContentItem, content: str = "", max_retries: int = 2) -> Optional[str]:
        """Generate a LinkedIn post using the SEO-optimized template."""

        for attempt in range(max_retries + 1):
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", self._get_user_prompt(item, content))
            ])

            try:
                chain = prompt | self.llm | StrOutputParser()
                result = chain.invoke({})
                post = self._clean_output(result)

                # Ensure URL is in the post
                if item.url not in post and 'http' not in post:
                    post = post.rstrip() + f"\n\n🔗 {item.url}"

                # Verify and retry if needed
                is_valid, issues = self.verifier.verify(post)
                if is_valid:
                    return post
                elif attempt < max_retries:
                    print(f"   🔄 Retry {attempt + 1}: {issues}")
                    continue
                else:
                    return post

            except Exception as e:
                print(f"   ⚠️ Write error: {e}")
                if attempt == max_retries:
                    return None
        return None

    def _get_system_prompt(self) -> str:
        """The single, clean LinkedIn template."""

        return """You are an AI/ML thought leader writing LinkedIn posts that drive engagement and are optimized for recruiter search algorithms.

Follow this template structure, but be CREATIVE with your writing and emoji choices.

══════════════════════════════════════════════════════════════
THE "INSIGHT & SEO" LINKEDIN TEMPLATE
══════════════════════════════════════════════════════════════

SECTION 1 - HOOK (1-2 sentences)
Write a controversial statement, surprising insight, or strong opinion that stops the scroll.
Use a relevant emoji at the end (choose based on topic - could be ⚠️ 🔥 ⚡ 🎯 💥 🧠 etc.)
Make it specific and bold. Challenge conventional thinking.

SECTION 2 - THE CORE INSIGHT (3-4 sentences)
Explain the current problem or state of affairs.
Why does this matter RIGHT NOW? What's at stake?
Add depth and context. Use a relevant emoji to start.

SECTION 3 - THE SOLUTION / BREAKDOWN
Explain how this tool/technology/approach addresses the problem.
List 3-5 bullet points with **bold headers**:
• **Point 1:** Detailed explanation of the benefit or feature
• **Point 2:** Another key advantage with specifics
• **Point 3:** Technical detail that matters to practitioners
• **Point 4:** (optional) Additional insight
• **Point 5:** (optional) Practical application

Each bullet should be 1-2 sentences, not just a few words.

SECTION 4 - THE RESULT/IMPACT (2-3 sentences)
What's the measurable outcome? Efficiency gain? Time saved? Accuracy improvement?
Be specific with numbers or comparisons when possible.
Use a relevant emoji (🚀 📈 ✅ 💪 etc.)

SECTION 5 - CALL TO ACTION
Ask a specific question that invites debate or sharing experiences.
Use either/or format OR ask about their experience.
Examples:
- "Are you prioritizing X or Y in your pipeline?"
- "Have you tried this approach? What results did you see?"
- "Which matters more in production: A or B?"

SECTION 6 - URL & HASHTAGS
Include the source URL on its own line.
Then exactly 5 relevant hashtags on the final line.

══════════════════════════════════════════════════════════════
SEO KEYWORDS - NATURAL INTEGRATION (CRITICAL)
══════════════════════════════════════════════════════════════

DO NOT list keywords separately. Instead, naturally weave technical terms throughout your post:
- Use specific technology names in context (e.g., "built with PyTorch" not "Tech Stack: PyTorch")
- Mention frameworks, languages, and tools as part of your explanation
- Include industry terms that recruiters search for (RAG, LLM, fine-tuning, embeddings, etc.)
- Make it sound like natural conversation, not keyword stuffing

GOOD: "This Python library leverages transformer architectures and vector databases to enable real-time RAG pipelines..."
BAD: "🛠️ Tech Stack: Python, Transformers, Vector DB, RAG"

══════════════════════════════════════════════════════════════
FORMAT & LENGTH REQUIREMENTS
══════════════════════════════════════════════════════════════

• LENGTH: 1800-2500 characters (THIS IS CRITICAL - write substantial content)
• Use **text** for bold (will be converted to Unicode bold)
• Use vertical spacing between sections (blank lines for readability)
• Mobile-friendly: paragraphs of 2-3 sentences max
• Choose emojis that match the TOPIC (AI: 🧠, speed: ⚡, warning: ⚠️, etc.)
• Keywords should flow naturally in sentences, not be listed

══════════════════════════════════════════════════════════════
BANNED PHRASES
══════════════════════════════════════════════════════════════

- "Imagine...", "Picture this...", "In today's world..."
- "Just came across...", "Ever wondered...", "Excited to share..."
- "Game changer", "Game-changer", "Revolutionary", "Groundbreaking"
- "What do you think?" (too generic)
- Star counts, upvotes, or GitHub metrics
- Section headers in the output

══════════════════════════════════════════════════════════════
OUTPUT
══════════════════════════════════════════════════════════════

Write ONLY the final LinkedIn post. No explanations, no meta-commentary.
Make it substantial (1800-2500 chars). Each section should have real depth."""

    def _get_user_prompt(self, item: ContentItem, content: str) -> str:
        """Build the user prompt with source material."""

        return f"""Write a LinkedIn post about this topic using the Insight & SEO template.

SOURCE MATERIAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title: {item.title}
URL: {item.url}
Source: {item.source}
Summary: {item.summary[:1000] if item.summary else 'N/A'}

Additional Context:
{content[:2500] if content else 'N/A'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL REQUIREMENTS:
1. LENGTH: Write 1800-2500 characters. Posts under 1500 chars will be REJECTED.
2. Follow template: Hook → Insight → Breakdown (3-5 bullets) → Result → CTA → URL → Hashtags
3. Hook must be bold and controversial (challenge conventional wisdom)
4. Each bullet point should be 1-2 full sentences, not just keywords
5. NATURALLY integrate technical keywords throughout (Python, RAG, LLM, etc.) - DO NOT list them separately
6. Include the URL before hashtags
7. CTA: specific either/or question or experience-based question
8. Exactly 5 hashtags at the end
9. Use relevant emojis creatively (not the same ones every time)
10. Write in simple, clear language that your audience understands

Write the LinkedIn post now (ONLY the post, make it SUBSTANTIAL):"""

    def _clean_output(self, text: str) -> str:
        """Clean and format the output."""
        # Remove broken patterns and meta-commentary
        text = re.sub(r'^[\]\[\)\(\*\s]+', '', text.strip())
        text = re.sub(r'\*\*\s*\*\*', '', text)
        text = re.sub(r'\n+---\n+.*$', '', text, flags=re.DOTALL)
        text = re.sub(r'\n+Note:.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\n+\(Note:.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'^(Here\'s the|Here is the).*?:\s*\n', '', text, flags=re.IGNORECASE)

        # Remove ALL markdown headers (###, ##, #) - they don't render on LinkedIn
        text = re.sub(r'^#{1,3}\s+.*$', '', text, flags=re.MULTILINE)

        # Remove template section headers if model includes them
        text = re.sub(r'^\s*(SECTION \d+|HOOK|INSIGHT|SOLUTION|RESULT|TECH|CTA|HASHTAG|CALL TO ACTION|THE SOLUTION|THE RESULT|THE IMPACT)[\s:]*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove "Tech Stack:" lines - keywords should be integrated naturally
        text = re.sub(r'^🛠️?\s*Tech Stack:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^Tech Stack:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Convert **bold** to Unicode bold
        def to_bold(match):
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            bold = "𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗"
            return match.group(1).translate(str.maketrans(chars, bold))

        text = re.sub(r'\*\*(.*?)\*\*', to_bold, text)
        text = re.sub(r'\*\*', '', text)

        # Clean up multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove any leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Remove empty lines at start
        text = re.sub(r'^\n+', '', text)

        return text.strip()

# ============================================================
# POST VERIFIER - LinkedIn SEO Template Standards
# ============================================================

class PostVerifier:
    """Verifies posts follow the Insight & SEO template."""

    BANNED_PHRASES = [
        "imagine a world", "imagine an era", "picture this",
        "in today's fast-paced", "in today's world", "in the ever-evolving",
        "ever wondered", "have you ever", "just came across", "just found",
        "stumbled upon", "excited to share", "happy to announce", "i'm thrilled",
        "[hook]", "[cta]", "[insert", "game changer", "game-changer", "revolutionary",
    ]

    def verify(self, post: str) -> tuple[bool, List[str]]:
        """Verify post follows template structure."""
        issues = []
        post_lower = post.lower()

        # Check banned phrases
        for phrase in self.BANNED_PHRASES:
            if phrase in post_lower:
                issues.append(f"Banned: '{phrase}'")
                break

        # Check length (1500-3000 for substantial posts)
        if len(post) < 1200:
            issues.append(f"Too short: {len(post)} chars (need 1200+)")
        elif len(post) > 3000:
            issues.append(f"Too long: {len(post)} chars")

        # Check hashtags (need 4-6)
        hashtags = re.findall(r'#\w+', post)
        if len(hashtags) < 4:
            issues.append(f"Need more hashtags: {len(hashtags)}")
        elif len(hashtags) > 6:
            issues.append(f"Too many hashtags: {len(hashtags)} (max 6)")

        # Check has link
        if 'http' not in post:
            issues.append("Missing URL")

        return len(issues) == 0, issues

# ============================================================
# MAIN AGENT
# ============================================================

def load_seen_links() -> Set[str]:
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            return set(df['original_link'].dropna().tolist())
        except:
            pass
    return set()


def save_post(data: dict):
    import csv
    df = pd.DataFrame([data])
    exists = os.path.exists(CSV_FILE)
    # Use QUOTE_ALL to properly handle multiline content in posts
    df.to_csv(CSV_FILE, mode='a', header=not exists, index=False, quoting=csv.QUOTE_ALL)


def run_agent(max_posts: int = 10, max_age_days: int = 7, sources: List[str] = None):
    print("=" * 60)
    print("🚀 LinkedIn Content Agent v2")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Filter: Last {max_age_days} days")
    if sources:
        print(f"   Sources: {', '.join(sources)}")
    else:
        print(f"   Sources: All")
    print("=" * 60)

    seen_links = load_seen_links()

    # Initialize with date filter
    scout = ContentScout(max_age_days=max_age_days)
    extractor = ContentExtractor()
    writer = SmartWriter(MODEL_NAME)

    # Fetch content from specified sources (filtered by date)
    items = scout.fetch_all(seen_links, limit_per_source=5, sources=sources)

    if not items:
        print("💤 No new content found.")
        return

    print(f"\n🏆 Processing top {min(max_posts, len(items))} items...")

    generated = 0
    for item in items[:max_posts]:
        print(f"\n{'='*50}")
        print(f"📌 [{item.source.upper()}] {item.title[:50]}...")
        print(f"   Score: {item.score:.1f} | {item.metrics}")

        # Extract content for RSS articles
        content = ""
        if item.source == "rss":
            print("   📄 Extracting content...")
            content = extractor.extract(item.url) or ""

        # Generate post (with built-in retries and verification)
        print("   ✍️  Writing...")
        post = writer.write(item, content)

        if not post:
            print("   ⏩ Skip: Generation failed")
            continue

        # Final verification check for reporting
        is_valid, issues = writer.verifier.verify(post)
        if not is_valid:
            print(f"   ⚠️  Minor issues: {issues}")

        # Save
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": item.title[:100],
            "original_link": item.url,
            "image_url": item.image_url,
            "linkedin_post": post,
            "relevance_score": round(item.score, 2),
            "quality_score": 7.5,
            "trending_boost": 1.0,
            "final_score": round(item.score + 1.0, 2),
            "verified": is_valid,
            "source": item.source,
        }
        save_post(entry)
        print(f"   💾 Saved! (Valid: {is_valid})")
        generated += 1

        time.sleep(2)

    print("\n" + "=" * 60)
    print(f"🎉 Done! Generated {generated} posts.")
    print(f"📄 Results: {CSV_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    # Usage: python agent.py [max_posts] [max_age_days] [sources]
    # Example: python agent.py 10 7 github,hackernews,reddit
    # Sources: github, hackernews, reddit, producthunt, papers, arxiv, devto, rss
    max_posts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_age_days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    sources = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    run_agent(max_posts=max_posts, max_age_days=max_age_days, sources=sources)
