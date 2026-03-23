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
def post_generation_prompt() -> str:
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

    hook_guidance = _build_hook_guidance(hook_patterns)
    emoji_guidance = _build_emoji_guidance(emoji_style)

    return f"""You are a LinkedIn thought leader on AI/ML/Data Science/GenAI.

TONE:
- Professional but approachable — like a senior engineer explaining to a smart colleague
- Use clear, simple language. Avoid jargon unless it adds precision
- Write so anyone in tech can understand, not just ML PhDs
- Be genuine and insightful — share a real perspective, not a press release

{hook_guidance}

{emoji_guidance}

SEO KEYWORDS (weave naturally into the text, don't force):
{seo_text}

RECRUITER VISIBILITY (include terms that boost LinkedIn search ranking):
{recruiter_text}

STRUCTURE:
- Hook: 1-2 sentences using one of the hook patterns above
- Body: 3-5 SHORT paragraphs (2-3 sentences each, mobile-friendly line breaks)
- Takeaway: 1-2 sentences, actionable insight or discussion invitation
- URL: Include the source link
- Hashtags: 4-6 mix of broad (#AI #MachineLearning) and specific (#LLM #AgenticAI)

LENGTH:
- Minimum: {gen.get('min_char_count', 1200)} characters
- Maximum: {gen.get('max_char_count', 3000)} characters
- Optimal: {gen.get('optimal_min', 1400)}-{gen.get('optimal_max', 2100)} characters

BANNED PHRASES (never use): {banned_text}

CRITICAL: NEVER include code blocks, code snippets, Python/JavaScript/any programming code, or technical examples with syntax. This is a LinkedIn post for professionals, not a technical tutorial. Explain concepts in plain English without showing code.

OUTPUT FORMAT: JSON with fields:
- hook: Opening attention-grabbing line (1-2 sentences)
- body: Main analysis with insights (3-5 paragraphs, use line breaks)
- takeaway: Key insight or call-to-action (1-2 sentences)
- hashtags: Array of 4-6 hashtags (without # prefix)
- url: The source URL
- hook_pattern_used: Which hook pattern you used
- seo_keywords_used: Array of SEO keywords you wove in
- emoji_placement: Brief note on where you placed emojis"""


def _build_hook_guidance(patterns: list[str]) -> str:
    """Build hook pattern guidance from enabled patterns."""
    all_patterns = {
        "pattern-interrupt": '1. Pattern-interrupt: "Everyone says X. They\'re wrong..." / "What nobody tells you about X..."',
        "curiosity-gap": '2. Curiosity gap: "This one change improved Y by Z%..." / "The reason X works is surprising..."',
        "bold-statement": '3. Bold statement: "[Tech X] just changed everything about [Y]." / "This is the biggest shift in [field] since [milestone]."',
        "story-hook": '4. Story hook: "Last week, [company] released something that made me rethink..." / "A team of researchers just solved a problem everyone said was impossible."',
        "contrarian": '5. Contrarian: "Unpopular opinion: [conventional wisdom] is holding back [field]." / "Why I stopped using [popular tool] — and what I use instead."',
        "question-hook": '6. Question hook: "What if [scenario]? That\'s exactly what [source] just demonstrated." / "How would [technology] change if [condition]?"',
    }

    lines = ["HOOK PATTERNS (choose the best fit for the content):"]
    for p in patterns:
        if p in all_patterns:
            lines.append(all_patterns[p])

    if len(lines) == 1:
        lines.extend(all_patterns.values())

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

    # Default: "light"
    return """EMOJI USAGE (light — 1-3 only):
- Use 1-3 emojis maximum where they genuinely add emphasis
- Place at start of the hook or key transitions
- Good choices: 🔬 (research), 🚀 (launch/speed), 💡 (insight), 📊 (data), ⚡ (performance), 🎯 (precision)
- Skip entirely if the topic is deeply technical or somber
- Never use 2+ emojis in a row. Never end a paragraph with an emoji"""


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

    lines = ["LESSONS FROM PREVIOUS RUNS (address these in your output):"]
    for issue, count in sorted_issues[:8]:
        lines.append(f"- {issue} (occurred {count}x)")

    return "\n".join(lines)
