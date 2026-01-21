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
import random
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

CSV_FILE = "/Users/mehran/Desktop/Linkedin Agent/linkedin_content_bank.csv"
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
        'ai', 'llm', 'gpt', 'claude', 'openai', 'anthropic', 'machine learning',
        'rag', 'langchain', 'agent', 'transformer', 'embedding', 'vector',
        'fine-tun', 'neural', 'deep learning', 'nlp', 'chatbot', 'copilot',
        'ollama', 'llama', 'mistral', 'diffusion', 'huggingface'
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

    def fetch_all(self, seen_links: Set[str], limit_per_source: int = 5) -> List[ContentItem]:
        """Fetch from all sources (filtered by date)."""
        items = []

        print(f"\n📡 Gathering hot topics from last {self.max_age_days} days...")

        # GitHub Trending
        items.extend(self._fetch_github(limit_per_source))

        # Hacker News
        items.extend(self._fetch_hackernews(limit_per_source))

        # Reddit
        items.extend(self._fetch_reddit(limit_per_source))

        # RSS Feeds
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
        """Fetch trending GitHub repos."""
        items = []
        try:
            url = "https://github.com/trending/python?since=daily"
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                return items

            soup = BeautifulSoup(resp.content, 'lxml')
            for article in soup.find_all('article', class_='Box-row')[:20]:
                try:
                    h2 = article.find('h2')
                    if not h2:
                        continue
                    link = h2.find('a')
                    if not link:
                        continue

                    full_name = link.get('href', '').strip('/')
                    name = full_name.split('/')[-1].lower()

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
                        timestamp=datetime.now().timestamp()  # Trending = fresh today
                    ))
                except:
                    continue

            print(f"   🐙 GitHub: {len(items)} repos")
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
        subreddits = ['LocalLLaMA', 'MachineLearning', 'ClaudeAI', 'ChatGPT', 'ollama']

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

    def _fetch_rss(self, seen_links: Set[str], limit: int) -> List[ContentItem]:
        """Fetch from RSS feeds (filtered by date)."""
        items = []
        feeds = [
            "https://huggingface.co/blog/feed.xml",
            "https://openai.com/blog/rss.xml",
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://venturebeat.com/category/ai/feed/",
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
# SMART POST WRITER
# ============================================================

class SmartWriter:
    """Generates varied LinkedIn posts with different styles."""

    # Different writing styles - NOT all posts need "My Take"
    STYLES = [
        "insight",      # Share insight/analysis
        "discovery",    # "Just discovered..." casual
        "breakdown",    # Technical breakdown
        "question",     # Lead with question
        "story",        # Mini story format
        "list",         # Listicle style
    ]

    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatOllama(model=model_name, temperature=0.85)
        self.verifier = PostVerifier()

    def write(self, item: ContentItem, content: str = "", max_retries: int = 2) -> Optional[str]:
        """Generate a post with randomly selected style. Retries if verification fails."""

        for attempt in range(max_retries + 1):
            style = random.choice(self.STYLES)

            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt(style)),
                ("user", self._get_user_prompt(item, content))
            ])

            try:
                chain = prompt | self.llm | StrOutputParser()
                result = chain.invoke({
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "summary": item.summary,
                    "metrics": str(item.metrics)
                })
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
                    return post  # Return anyway on last attempt

            except Exception as e:
                print(f"   ⚠️ Write error: {e}")
                if attempt == max_retries:
                    return None
        return None

    def _get_system_prompt(self, style: str) -> str:
        """Get style-specific system prompt."""

        base = """You write engaging LinkedIn posts about AI/tech.
Write naturally like a tech professional sharing insights - NOT like marketing copy.

CRITICAL REQUIREMENTS:
1. LENGTH: Write 1400-1800 characters (about 250-300 words). Posts under 1000 chars will be rejected.
2. URL: You MUST include the full source URL in the post body (copy it exactly as provided)
3. HASHTAGS: End with exactly 5 relevant hashtags on the last line

BANNED:
- "Imagine...", "As an engineer...", placeholder brackets like [hook], "Feel free to..."
- Generic filler text - every sentence must add value
- Mentioning star counts, upvote counts, or point counts (e.g. "10,000 stars", "500 upvotes")
- Social proof metrics - focus on the CONTENT value, not popularity numbers"""

        style_instructions = {
            "insight": """
Style: Share a key insight or analysis.
Start with a bold statement about what you learned or noticed.
Explain WHY it matters. Add your perspective. End with a thought-provoking question.""",

            "discovery": """
Style: Casual discovery share.
Start like "Just came across..." or "Found something interesting..."
Keep it conversational. Share what caught your attention and why others should care.""",

            "breakdown": """
Style: Technical breakdown.
Lead with the core technical innovation.
Break down 3-4 key technical aspects with bullet points.
Explain practical implications.""",

            "question": """
Style: Lead with a provocative question.
Start with a question that makes people think.
Then provide context and your thoughts. End asking for others' opinions.""",

            "story": """
Style: Mini narrative.
Start with a specific moment or scenario.
Build to the insight or tool. Make it relatable.""",

            "list": """
Style: Quick valuable list.
Start with "X things about [topic]:" or "Why [topic] matters:"
Use numbered points. Keep each point punchy and valuable."""
        }

        return base + style_instructions.get(style, style_instructions["insight"])

    def _get_user_prompt(self, item: ContentItem, content: str) -> str:
        """Build user prompt based on content type."""

        source_context = {
            "github": "This is a trending GitHub repository gaining attention in the AI community.",
            "hackernews": "This is generating discussion on Hacker News.",
            "reddit": "This is a hot topic in AI Reddit communities.",
            "rss": "This is a recent article from a tech publication."
        }

        return f"""Write a LinkedIn post about:

Title: {item.title}
Source: {item.source}
Context: {source_context.get(item.source, '')}
URL: {item.url}
Summary: {item.summary[:500] if item.summary else 'N/A'}
Additional content: {content[:1500] if content else 'N/A'}

REMEMBER: Do NOT mention any numbers like stars, points, or upvotes. Focus on WHAT it does and WHY it matters.

Write a natural, engaging post now:"""

    def _clean_output(self, text: str) -> str:
        """Clean and format the output."""
        # Remove broken patterns
        text = re.sub(r'^[\]\[\)\(\*\s]+', '', text.strip())
        text = re.sub(r'\*\*\s*\*\*', '', text)

        # Convert **bold** to Unicode bold
        def to_bold(match):
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            bold = "𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗"
            return match.group(1).translate(str.maketrans(chars, bold))

        text = re.sub(r'\*\*(.*?)\*\*', to_bold, text)
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

# ============================================================
# POST VERIFIER
# ============================================================

class PostVerifier:
    """Light verification - not overly strict."""

    BANNED = ["imagine a world", "imagine an era", "as an ml engineer", "[hook]", "[cta]"]

    def verify(self, post: str) -> tuple[bool, List[str]]:
        """Verify post quality. Returns (is_valid, issues)."""
        issues = []
        post_lower = post.lower()

        # Check banned phrases
        for phrase in self.BANNED:
            if phrase in post_lower:
                issues.append(f"Contains: '{phrase}'")

        # Check length
        if len(post) < 800:
            issues.append(f"Too short: {len(post)} chars")
        elif len(post) > 2800:
            issues.append(f"Too long: {len(post)} chars")

        # Check hashtags exist
        hashtags = re.findall(r'#\w+', post)
        if len(hashtags) < 3:
            issues.append(f"Need more hashtags: {len(hashtags)}")

        # Check has link
        if 'http' not in post:
            issues.append("Missing link")

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
    df = pd.DataFrame([data])
    exists = os.path.exists(CSV_FILE)
    df.to_csv(CSV_FILE, mode='a', header=not exists, index=False)


def run_agent(max_posts: int = 10, max_age_days: int = 7):
    print("=" * 60)
    print("🚀 LinkedIn Content Agent v2")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Filter: Last {max_age_days} days")
    print("=" * 60)

    seen_links = load_seen_links()

    # Initialize with date filter
    scout = ContentScout(max_age_days=max_age_days)
    extractor = ContentExtractor()
    writer = SmartWriter(MODEL_NAME)

    # Fetch all content (filtered by date)
    items = scout.fetch_all(seen_links, limit_per_source=5)

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
    # Usage: python agent_v2.py [max_posts] [max_age_days]
    # Example: python agent_v2.py 10 7  (10 posts, last 7 days)
    max_posts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_age_days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    run_agent(max_posts=max_posts, max_age_days=max_age_days)
