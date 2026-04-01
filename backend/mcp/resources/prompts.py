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

FORMATTING — USE THESE TOOLS ACTIVELY:
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
- Paragraph length: 1-2 sentences is standard. One sentence per line for emphasis.

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
- Place at the start of the hook line and optionally at one key transition
- Match emoji to the SPECIFIC claim — 🔬 only for actual research, ⚡ only for actual performance, 😳 for genuinely surprising numbers
- Never use 2+ emojis in a row. Never end a paragraph with an emoji
- AVOID generic fallback emojis (💡🔥✅) unless they precisely fit the sentence"""


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
