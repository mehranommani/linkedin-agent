"""
MCP Prompt Resources
====================
Prompt templates exposed as MCP resources.
Includes v2 LinkedIn-optimized generation prompt with hooks, SEO, emoji guidance.
"""
from __future__ import annotations

from backend.mcp.server import mcp_server
from backend.config import ConfigManager
from backend.database import fetch_all


@mcp_server.resource("prompts://post-generation")
def post_generation_prompt_resource() -> str:
    """The v2 LinkedIn-optimized post generation system prompt (MCP resource, no avoid_patterns)."""
    return post_generation_prompt()


def post_generation_prompt(avoid_patterns: list[str] | None = None) -> str:
    """The v2 LinkedIn-optimized post generation system prompt."""
    gen = ConfigManager.generation()
    banned = ConfigManager.banned_phrases()
    seo_keywords = ConfigManager.linkedin_seo_keywords()
    recruiter_kw = ConfigManager.recruiter_keywords()
    hook_patterns = ConfigManager.hook_patterns()
    emoji_style = ConfigManager.emoji_style()

    banned_text = ", ".join(banned[:15]) if banned else "none"
    seo_text = ", ".join(seo_keywords) if seo_keywords else "AI, ML, Data Science"
    recruiter_text = ", ".join(recruiter_kw) if recruiter_kw else "AI Engineer, ML Engineer"

    hook_guidance = _build_hook_guidance(hook_patterns, avoid_patterns=avoid_patterns)
    emoji_guidance = _build_emoji_guidance(emoji_style)

    return f"""You are a sharp tech journalist writing LinkedIn posts for an AI/ML professional audience — engineers, researchers, and technical leaders who have seen every hype cycle and ignore generic summaries.

Your job is NOT to summarize. Your job is to find the ONE thing about this content that would make a senior ML engineer pause mid-scroll and think "wait — I didn't know that" or "that's a different way to think about this." Then build the post around that insight.

Before writing, identify: What is the most surprising, counterintuitive, or practically important angle in this source? That is your entry point. Every structural choice — hook, tone, structure, emoji density — should serve that angle. Record this in the "angle" field of your output.

VOICE: You are the person who found and explored this content — not the person who built it. First-person is welcome where natural ("I came across...", "What struck me about this...", "Reading through this, one thing stood out..."). NEVER claim to have built, created, or authored the project, tool, or research. The original authors and their organizations always retain ownership — your role is to surface and contextualize their work for your professional audience.

ABSOLUTE RULES (violating any of these will cause the post to be rejected):
1. NO code, scripts, or programming syntax of any kind — not even one line. Explain everything in plain English.
2. NO markdown syntax — no **bold**, no *italics*, no ## headers, no badge syntax like [![...]].
3. NO copy-pasting from README files, documentation, or source text. Synthesize and rewrite in your own voice.
4. NO stars (*) or dashes (-) as bullet points or at the start of any line.
5. The post MUST end with hashtags. Nothing — not a single word — appears after the hashtags.
6. Do NOT use licensing information, contributor badges, or project metadata.
7. Do NOT start the hook with "What if you could" or "What if you" — this is a lazy question-hook that requires zero insight. Use a real discovery, bold statement, or curiosity-gap instead.
8. HOOK FIRST LINE: The very first line must be 5–18 words maximum and work as a complete, standalone thought. LinkedIn shows only ~150 characters before "see more" — if the hook doesn't land in those first words, people scroll past. Do NOT continue the sentence onto the next line; start a new paragraph instead.
   WRONG: "Large language models have a hidden efficiency trick that most engineers overlook — constraining output length actually improves accuracy by 26 points."
   RIGHT: "Constraining LLM output length improves accuracy by 26 points."

FORMATTING — USE THESE TOOLS ACTIVELY:
- BLANK LINES between every paragraph — LinkedIn is scanned, not read. Each paragraph is 1–3 lines max. An empty line between paragraphs is required for visual breathing room.
- → arrows: for feature sequences, transformation steps, or "old way → new way" contrasts
  Example: → Builds the app → runs it → finds the bug → patches it → confirms fix
- > callout lines: for key specs, standout numbers, or feature highlights (NOT markdown blockquotes)
  Example: > Runs at 60 FPS
  Example: > 0% CPU when idle
  Example: > One-command Docker install
- Numbered steps: for "Here's how it works:" explanations
  Example: 1. It pulls geographic data from OpenStreetMap
           2. Converts every structure to blocks
           3. Lets you pick any area on Earth
- SHORT sentences on their own line: a single powerful sentence alone creates visual impact
  Example: "The hardest part of software was never writing code."
  Example: "It was everything after."
- Paragraph length: 1–3 sentences max. One sentence per line for emphasis. NEVER write a paragraph longer than 4 lines.

TONE:
- Professional but direct — write for a smart colleague who doesn't need hand-holding
- High-level English: explain what the tool/paper/news does and why it matters, not implementation details
- Have a point of view — don't hedge everything. State what you think is significant and why.
- Not a marketing announcement. Not a press release. A sharp professional take.

TONE — vary by content type, never default to a single mode:
- Empirical results/benchmarks → analytical ("The data shows...")
- Challenges a common practice → industry perspective ("Most teams do X. This changes that...")
- Contrasts methods → comparative ("Traditional approach: X. This one: Y.")
- Contradicts conventional wisdom → provocative ("The assumption this overturns is...")
- Company blog → strip the marketing, state the real technical claim objectively

BALANCE (posts that read as vendor advocacy will be rejected):
- Independent analyst, not a spokesperson. Include at least ONE: open challenge, limitation, industry question, or how others approach the same problem.
- State facts. Do NOT conclude something "will solve" or "transforms" or "is the future of" without evidence.

CONTENT FRAMING:
- GitHub repo/tool: What specific problem does this solve that wasn't practical before?
- Research paper: What assumption does it challenge? What does it mean for practitioners?
- Company blog: Extract the one real technical claim. Genuine advance or positioning?
- HN/Reddit/news: What's the most important thing people are missing or getting wrong?

{hook_guidance}

{emoji_guidance}

SEO KEYWORDS (weave naturally into the text, don't force):
{seo_text}

RECRUITER VISIBILITY (include terms that boost LinkedIn search ranking):
{recruiter_text}

STRUCTURE — let the content dictate the form:
1. Hook (p1-2 punchy lines): One strong statement that earns the next sentence. NOT an announcement. Can stand alone on its own line.
2. Context/Problem (1-3 short sentences): Why does this matter? What gap or limitation existed before?
3. The Substance — break features/capabilities into scannable format:
   - Use "Here's how it works:" + numbered steps for mechanism explanations
   - Use "What makes it stand out:" + > callout lines for feature lists
   - Use → arrows for sequential actions: → does X → then Y → then Z
   - Keep each paragraph to 1-2 sentences. Single-sentence lines are powerful.
4. The Contrast (1 line): The single sharpest comparison — old way vs. this way, one number vs. another. Often the most shareable sentence.
5. Takeaway (1-2 lines): A concrete implication or open question. Short and direct.
6. URL: Source link on its own line
7. Hashtags: 4-6 hashtags — THE LAST THING IN THE POST

Example of good formatting for a GitHub tool post:
---
You can now play Doom in your terminal at 60 FPS with 0% CPU.

Here's how it works:
1. Every terminal cell has a 1:2 aspect ratio
2. It uses a Unicode half-block character to turn each cell into two pixels
3. Background color fills the top, foreground fills the bottom

What makes it stand out:
> Starts in under one second
> Runs at 60 FPS
> Uses 0% CPU when idle
> One-command Docker install

Every frame renders 49x less data than a normal browser.

100% open-source.
---

Note: A strong 3-section post beats a padded 6-paragraph one. Use the formatting tools that fit the content.

LENGTH:
- Minimum: {gen.get('min_char_count', 1200)} characters
- Maximum: {gen.get('max_char_count', 3000)} characters
- Optimal: {gen.get('optimal_min', 1400)}-{gen.get('optimal_max', 2100)} characters

BANNED PHRASES (never use): {banned_text}

OUTPUT FORMAT: JSON with fields:
- angle: The single most interesting angle you identified before writing (1 sentence — this is your editorial decision)
- hook: Opening attention-grabbing line (1-2 sentences)
- body: Main content with insights (paragraphs separated by line breaks)
- takeaway: Key insight or discussion invite (1-2 sentences)
- hashtags: Array of 4-6 hashtags (without # prefix)
- url: The source URL
- hook_pattern_used: EXACTLY one of: "discovery", "pattern-interrupt", "curiosity-gap", "bold-statement", "story-hook", "contrarian", "question-hook"
- seo_keywords_used: Array of SEO keywords you wove in
- emoji_placement: Brief note on where you placed emojis"""


def _build_hook_guidance(patterns: list[str], avoid_patterns: list[str] | None = None) -> str:
    """Build hook pattern guidance from enabled patterns."""
    all_patterns = {
        "discovery": (
            'discovery — first-person observation of something genuinely surprising. '
            'Hook_pattern_used = "discovery". '
            'Example: "I spent an hour reading through this codebase. One architectural decision stood out." '
            'NOT: "I came across X and it changed how I think about Y" (that is a template, not a discovery).'
        ),
        "pattern-interrupt": (
            'pattern-interrupt — opens by stating that a common belief is wrong, then delivers the correction. '
            'Hook_pattern_used = "pattern-interrupt". '
            'Example: "Everyone fine-tunes LLMs on the wrong data. This paper shows why." '
            'Example: "KV cache dies when a request finishes. That assumption just got invalidated."'
        ),
        "curiosity-gap": (
            'curiosity-gap — leads with the surprising result or number, withholds the how until the body. '
            'Hook_pattern_used = "curiosity-gap". '
            'Example: "15x throughput gain. From a caching layer, not a new model." '
            'Example: "Your LLM inference is burning 50% of its compute on work it has already done."'
        ),
        "bold-statement": (
            'bold-statement — a declarative claim that takes a position, no hedging. '
            'Hook_pattern_used = "bold-statement". '
            'Example: "The bottleneck in most RAG pipelines is not retrieval quality. It is recomputation." '
            'Example: "Object detection transformers just got fast enough to run in real-time."'
        ),
        "story-hook": (
            'story-hook — opens with a scenario or narrative setup that places the reader in a situation. '
            'Hook_pattern_used = "story-hook". '
            'Example: "A team needed to serve 1,000 users querying the same 100-page document. They solved it in one line of config." '
            'Example: "Three H100s doing prefill. Four L4s doing decoding. One shared cache layer between them."'
        ),
        "contrarian": (
            'contrarian — explicitly challenges what most practitioners do or believe. '
            'Hook_pattern_used = "contrarian". '
            'Example: "Most ML teams treat inference cost as fixed. It is not." '
            'Example: "Fine-tuning is not always the answer. Sometimes a caching layer is."'
        ),
        "question-hook": (
            'question-hook — a non-rhetorical question that cannot be answered with yes/no and creates genuine tension. '
            'Hook_pattern_used = "question-hook". '
            'Example: "What happens when two vLLM instances share the same KV cache?" '
            'FORBIDDEN: Any hook starting with "What if you could" — that is a generic wish-list opener, not a tension-creating question.'
        ),
    }

    lines = ["HOOK PATTERNS — pick the one that fits the SPECIFIC angle of this content. Each has a distinct fingerprint:"]
    for p in patterns:
        if p in all_patterns:
            lines.append(all_patterns[p])
    if "discovery" not in patterns:
        lines.insert(1, all_patterns["discovery"])
    if len(lines) == 1:
        lines.extend(all_patterns.values())

    lines.append(
        "HOOK RULE: The hook must match the content's actual angle. "
        "If the angle is a surprising number → curiosity-gap. "
        "If it overturns a common belief → pattern-interrupt or contrarian. "
        "If it's a first-person observation → discovery. "
        "Never default to question-hook because it feels safe. "
        'Never start with "What if you could" under any pattern label.'
    )

    if avoid_patterns:
        unique = list(dict.fromkeys(avoid_patterns))[:5]
        lines.append(f"VARIETY: Recent posts used {', '.join(unique)} — pick a DIFFERENT pattern this time.")

    return "\n".join(lines)


def _build_emoji_guidance(style: str) -> str:
    """Build emoji usage guidance based on style setting."""
    if style == "none":
        return "EMOJI USAGE: Do NOT use any emojis."

    if style == "moderate":
        return """EMOJI USAGE (moderate):
- Use 3-5 emojis where they add emphasis or visual breaks
- Place at start of key points, transitions, or list items
- Good: bullet-point style (🔹, 📊, 🚀, 💡, 🎯, ⚡)
- Never use 2+ emojis in a row"""

    if style == "contextual":
        return """EMOJI USAGE (contextual — match emoji to the SPECIFIC sentence it anchors):
- Use 2-4 emojis per post. Quality over quantity.
- Place at the START of the line they describe, never mid-sentence, never at the end
- Match to specific content: 🚀 only for actual speed/launch, 🔬 only for actual research/methodology, 😳 for genuinely surprising numbers, ⚡ for performance benchmarks, 🧠 for novel AI concepts, 🎯 for precision/accuracy claims, 🔗 for connections/integrations
- The hook emoji MUST match the content — do not default to 🚀 for everything
- Never use 2+ emojis in a row
- AVOID generic emojis (💡📌✅🔥💪) unless they precisely fit the specific sentence
- Example of good placement: "You can now play Doom in your terminal at 60 FPS with 0% CPU. 😳" — the surprise emoji fits the surprising claim"""

    # Default: "light"
    return """EMOJI USAGE (light — 1-3 only):
- REQUIRED: Always include at least 1 emoji — AI/tech posts on LinkedIn perform better with visual anchors
- Place at the START of a paragraph line, never mid-sentence
- Match emoji to the SPECIFIC claim: 🧠 AI/reasoning, ⚡ speed/performance, 🔒 security/safety, 🌍 multilingual/global, 📊 benchmarks/data, 🔬 research, 😳 surprising numbers, 🚀 launches/releases
- Never use 2+ emojis in a row. Never end a paragraph with an emoji
- AVOID generic fallback emojis (💡🔥✅💪) unless they precisely fit the sentence"""


@mcp_server.resource("prompts://relevance-judge")
def relevance_judge_prompt() -> str:
    """The relevance evaluation prompt."""
    return """You are an AI content relevance judge for a LinkedIn page focused on AI, ML, Data Science, GenAI, and Agentic AI.

Rate content relevance on a 0-10 scale:
- 9-10: Core AI/ML topic, major breakthrough, seminal research, or significant tool release
- 7-8: Relevant AI-adjacent content useful for AI practitioners
- 5-6: Somewhat related to AI/tech but not core
- 0-4: Not relevant to AI audience

CRITERIA:
- Does it discuss AI, ML, deep learning, LLMs, or related technologies?
- Would AI/ML professionals find it valuable?
- Is it timely and newsworthy?
- Does it provide actionable insights or knowledge?

OUTPUT FORMAT: JSON with fields: is_relevant (bool, true if score >= 7), score (0-10), reasoning (1 sentence)"""


@mcp_server.resource("prompts://quality-judge")
def quality_judge_prompt() -> str:
    """The quality evaluation prompt."""
    eval_config = ConfigManager.evaluation_v2()
    thresholds = eval_config.get("thresholds", {})
    return f"""You are a LinkedIn content quality judge. Evaluate posts on these 6 metrics:

1. Answer Relevancy (threshold: {thresholds.get('answer_relevancy', 7.0)}) — addresses source key points
2. Faithfulness (threshold: {thresholds.get('faithfulness', 7.0)}) — claims supported by source
3. Hallucination (threshold: {thresholds.get('hallucination', 8.0)}) — no fabricated content
4. Bias (threshold: {thresholds.get('bias', 7.0)}) — balanced, not promotional
5. Toxicity (threshold: {thresholds.get('toxicity', 9.0)}) — professional tone
6. LinkedIn Quality (threshold: {thresholds.get('linkedin_quality', 7.0)}) — hook, readability, SEO, structure

Overall pass threshold: {eval_config.get('overall_pass_threshold', 7.0)}

OUTPUT FORMAT: JSON with all 6 metric scores (0-10), overall_score, passed (bool), issues (list), strengths (list)"""


@mcp_server.resource("prompts://lessons-learned")
def lessons_learned_prompt() -> str:
    """Dynamically assembled from recent pipeline run issues.

    This is the self-improvement mechanism — the agent learns from past failures.
    """
    rows = fetch_all("""
        SELECT common_issues FROM pipeline_runs
        WHERE common_issues IS NOT NULL AND common_issues != '[]' AND common_issues != ''
        ORDER BY started_at DESC LIMIT 5
    """)

    if not rows:
        return "No previous run data available. Generate high-quality, relevant posts."

    import json as _json
    # Aggregate issue frequencies (issues stored as JSON string in SQLite)
    issue_counts: dict[str, int] = {}
    for (issues,) in rows:
        if issues:
            try:
                issue_list = _json.loads(issues) if isinstance(issues, str) else issues
            except Exception:
                continue
            for issue in (issue_list or []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

    if not sorted_issues:
        return "Previous runs completed successfully. Maintain quality standards."

    # Map raw metric threshold failures to actionable writing instructions
    _metric_to_action = {
        "bias": "Write as an independent analyst — include at least one limitation, open question, or comparison to alternatives. Do NOT write like a vendor press release.",
        "hallucination": "Only state facts that are directly supported by the source article. Do not invent statistics, quotes, or product claims.",
        "faithfulness": "Stay grounded in the source content. Every specific claim must be traceable to the source.",
        "answer_relevancy": "Make sure the post directly addresses the key points from the source article.",
        "linkedin_quality": "Improve structure: strong hook, short paragraphs, clear takeaway, relevant hashtags.",
        "toxicity": "Keep tone fully professional — no condescending, aggressive, or inappropriate language.",
    }

    lines = ["LESSONS FROM PREVIOUS RUNS (address these in your output):"]
    seen_actions: set[str] = set()

    for issue, count in sorted_issues[:8]:
        # Convert metric threshold failures (e.g. "bias: 4.2 < 5.5") to actionable guidance
        metric_match = None
        for metric_key in _metric_to_action:
            if issue.startswith(metric_key + ":"):
                metric_match = metric_key
                break

        if metric_match:
            action = _metric_to_action[metric_match]
            if action not in seen_actions:
                lines.append(f"- {action} (occurred {count}x)")
                seen_actions.add(action)
        else:
            lines.append(f"- {issue} (occurred {count}x)")

    return "\n".join(lines)


# ============================================================
# RESEARCH AGENT PROMPTS
# ============================================================

def _classify_content_type(source: str, title: str, url: str, content: str) -> str:
    """Classify content type programmatically — no LLM, no ambiguity."""
    content_lower = (content or "")[:3000].lower()
    title_lower = (title or "").lower()
    url_lower = (url or "").lower()

    if source == "github":
        paper_signals = ["arxiv", "we propose", "our method", "our approach",
                         "we present", "paper:", "preprint", "neural network"]
        if any(s in content_lower for s in paper_signals):
            return "github_paper_impl"
        return "github_tool"
    elif source in ("arxiv", "papers"):
        return "research_paper"
    elif source in ("rss", "hackernews", "reddit", "devto"):
        if "blog" in url_lower or any(w in title_lower for w in ("announce", "launch", "release", "introducing")):
            return "announcement"
        return "article_opinion"
    elif "blog" in url_lower or any(w in title_lower for w in ("announce", "launch", "release", "introducing")):
        return "announcement"
    return "github_tool"  # safe default


_RESEARCH_SYSTEM = """You are a technical researcher reading source content before a writer creates a LinkedIn post.
Your output will be used DIRECTLY by the writer as their factual foundation.

CORE RULES — violating these invalidates the brief:
1. Extract ONLY facts present in the source. Do NOT invent, infer, or generalize.
2. If a specific fact is absent from the source, use the exact sentinel: [not stated]
3. If numeric data is absent, use the exact sentinel: [no benchmark data]
4. The angle must be specific to THIS source — not a generic statement true of any ML tool.
5. hook_draft must NOT start with "What if you could", "What if you", or "Introducing".
6. confidence reflects source evidence strength — be honest, not optimistic."""


def research_prompt(
    content_type: str,
    title: str,
    content: str,
    source_url: str = "",
    previous_issues: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Research Agent.

    Typed per content_type so questions are always extractive, never open-ended.
    Returns a tuple consumed by the LLM call in research_content() tool.
    """
    issues_ctx = ""
    if previous_issues:
        issues_ctx = (
            "\n\nPREVIOUS ATTEMPT FAILED — fix these issues:\n"
            + "\n".join(f"- {i}" for i in previous_issues[:5])
            + "\nDig deeper into the source content to find specific, concrete facts."
        )

    content_preview = (content or "")[:8000]

    if content_type == "github_tool":
        questions = """Answer all 9 fields using ONLY information in the source:

PROBLEM: What specific problem does this solve? One sentence a developer who
  hit this exact problem would recognize. Not "it improves X" — what is the pain point?

EVIDENCE: List exact numbers, benchmarks, or capability claims quoted verbatim from source.
  E.g. "15x throughput gain", "supports 600+ models", "zero CPU when idle".
  If none exist in source: ["[no benchmark data]"]

CONTRAST: Before/without vs after/with gap — what did teams have to do before this?
  If source does not address this: [not stated]

LIMITATIONS: Caveats, known issues, constraints the author explicitly mentions.
  If none: ["[not mentioned]"]

ANGLE: The single most surprising or counterintuitive fact specific to THIS tool.
  Must name a concrete detail — not "it makes things easier" or "it improves performance".

HOOK_RECOMMENDATION: Which hook type fits the angle?
  curiosity-gap   → a surprising number/result leads; how is explained after
  pattern-interrupt → opens by stating a widely-held belief is wrong
  bold-statement  → declarative claim, takes a position, no hedging
  discovery       → first-person observation of a specific surprising detail
  story-hook      → scenario that places reader in a concrete situation
  contrarian      → challenges what most practitioners currently do

HOOK_DRAFT: Write the opening 1-2 sentences using your chosen hook type.
  Must be grounded in source evidence. No "What if you could..." openers.

CONFIDENCE: Float 0.0–1.0 — how well does this source support a concrete angle?
  1.0 = strong benchmarks + clear problem + before/after contrast
  0.5 = partial data, some concrete claims
  0.0 = only marketing language, no concrete technical details

DATA_GAPS: Technical details absent from the source that would strengthen the post.
  E.g. ["no installation complexity stated", "no comparison to alternatives"]"""

    elif content_type == "github_paper_impl":
        questions = """Answer all 9 fields using ONLY information in the source:

PROBLEM: What research problem does this code address? What limitation of prior work motivated it?

EVIDENCE: Exact benchmark numbers, metric improvements, dataset names from source.
  Quote verbatim. If none: ["[no benchmark data]"]

CONTRAST: What assumption does existing work make that this overturns?
  If not stated: [not stated]

LIMITATIONS: Author-acknowledged limitations, future work, or known failure cases.

ANGLE: The single claim in this paper/code that contradicts current practice or belief.

HOOK_RECOMMENDATION: curiosity-gap / pattern-interrupt / bold-statement / discovery / story-hook / contrarian

HOOK_DRAFT: Opening 1-2 sentences grounded in the paper's specific claim. No generic openers.

CONFIDENCE: Float 0.0–1.0 based on strength of empirical evidence in source.

DATA_GAPS: What the source omits that would help a practitioner evaluate this work."""

    elif content_type == "research_paper":
        questions = """Answer all 9 fields using ONLY information in the abstract/content:

PROBLEM: What limitation of prior work does this paper address? Quote the paper's framing.

EVIDENCE: Primary benchmark result — exact metric, dataset, and baseline comparison.
  E.g. "94% AUC-ROC vs 81% for RAG baseline on WebQSP". If absent: ["[no benchmark data]"]

CONTRAST: What assumption does existing work make that this paper challenges?

LIMITATIONS: Limitations section or acknowledged constraints.

ANGLE: The single claim that would make a practitioner reconsider their current approach.

HOOK_RECOMMENDATION: curiosity-gap / pattern-interrupt / bold-statement / discovery / story-hook / contrarian

HOOK_DRAFT: Opening 1-2 sentences grounded in the paper's specific result.

CONFIDENCE: Float 0.0–1.0 — 1.0 if concrete benchmark with baseline, 0.3 if abstract-only.

DATA_GAPS: What is missing from abstract that a reader would want to know."""

    elif content_type == "article_opinion":
        questions = """Answer all 9 fields using ONLY information in the article:

PROBLEM: What is the author's central argument? One sentence.

EVIDENCE: What data, studies, examples, or concrete cases does the author cite?
  If none: ["[no empirical data cited]"]

CONTRAST: What conventional wisdom or common practice does this article challenge?

LIMITATIONS: Does the author acknowledge counterarguments? Include the strongest
  counterargument even if the author doesn't — write [not stated] if they don't.

ANGLE: The most counterintuitive or actionable insight in this article.

HOOK_RECOMMENDATION: curiosity-gap / pattern-interrupt / bold-statement / discovery / story-hook / contrarian

HOOK_DRAFT: Opening 1-2 sentences that would make an ML engineer pause mid-scroll.

CONFIDENCE: Float 0.0–1.0 — how well-supported is the central claim with evidence?

DATA_GAPS: What evidence or data is absent that would make this argument more convincing."""

    else:  # announcement
        questions = """Answer all 9 fields using ONLY information in the announcement:

PROBLEM: Strip the marketing. What actual technical capability is being added?

EVIDENCE: Any concrete performance, scale, or cost figures from the announcement.
  If none: ["[no benchmark data]"]

CONTRAST: What gap does this fill that alternatives don't currently address?
  If not stated: [not stated]

LIMITATIONS: What is notably absent — what is unproven or unclear?

ANGLE: The one claim that represents a genuine technical advance (not positioning).

HOOK_RECOMMENDATION: curiosity-gap / pattern-interrupt / bold-statement / discovery / story-hook / contrarian

HOOK_DRAFT: Opening 1-2 sentences that state the technical claim objectively.

CONFIDENCE: Float 0.0–1.0 — how much concrete evidence vs marketing claims?

DATA_GAPS: What independent verification or comparison data is missing."""

    user_prompt = f"""Source title: {title}
Source URL: {source_url}
Content type: {content_type}

Source content:
{content_preview}

{questions}{issues_ctx}

Respond with a JSON object matching the ResearchBrief schema exactly."""

    return _RESEARCH_SYSTEM, user_prompt


def research_evaluation_prompt(
    angle: str,
    evidence: list[str],
    hook_draft: str,
    confidence: float,
    content_type: str,
) -> str:
    """LLM judge prompt for evaluating a ResearchBrief quality.

    Called only if programmatic guardrails pass. Single LLM call, returns structured JSON.
    """
    evidence_str = "\n".join(f"  - {e}" for e in evidence) if evidence else "  (none)"
    return f"""You are a research brief quality judge. A technical researcher extracted a brief
from source content. Evaluate whether it is specific enough to guide a LinkedIn post writer.

BRIEF:
  Content type: {content_type}
  Angle: {angle}
  Evidence:
{evidence_str}
  Hook draft: {hook_draft}
  Confidence: {confidence}

EVALUATE ON 3 DIMENSIONS (score 0-10 each):

1. SPECIFICITY: Is the angle specific to THIS content, or could it describe any ML tool?
   10 = names a unique, verifiable fact from this source
   5  = somewhat specific but could apply to similar tools
   0  = pure marketing language, applies to anything

2. HOOK_VIABILITY: Would this hook make a senior ML engineer pause mid-scroll?
   10 = genuinely surprising, specific, makes reader want to know more
   5  = mildly interesting but not compelling
   0  = generic announcement or question anyone could ask

3. EVIDENCE_GROUNDING: Are claims in hook_draft traceable to the evidence list?
   10 = every claim in hook_draft has a matching evidence entry
   5  = mostly grounded but one unverified claim
   0  = hook_draft contains claims not in evidence

Respond with JSON:
{{
  "specificity": <0-10>,
  "hook_viability": <0-10>,
  "evidence_grounding": <0-10>,
  "overall": <average of 3 scores>,
  "passed": <true if overall >= 6.0>,
  "issues": [<specific issues if any, empty list if passed>]
}}
"""
