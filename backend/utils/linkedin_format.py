"""
LinkedIn Unicode Bold Formatter
================================
LinkedIn doesn't support Markdown bold (**text**), but it renders Unicode
Mathematical Bold characters as visually bold text. This module converts
the hook line and key metrics to bold automatically before a post is saved.

Usage:
    from backend.utils.linkedin_format import apply_linkedin_bold
    post_text = apply_linkedin_bold(post_text)
"""
from __future__ import annotations

import re

# Unicode Mathematical Bold mappings (U+1D400 block)
_UPPER = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭"
_LOWER = "𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇"
_DIGITS = "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"

_BOLD_MAP: dict[str, str] = {}
for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    _BOLD_MAP[ch] = _UPPER[i]
for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
    _BOLD_MAP[ch] = _LOWER[i]
for i, ch in enumerate("0123456789"):
    _BOLD_MAP[ch] = _DIGITS[i]


def to_bold(s: str) -> str:
    """Convert ASCII letters and digits in *s* to Unicode bold equivalents."""
    return "".join(_BOLD_MAP.get(c, c) for c in s)


# Matches metric phrases like: 40x, 3x faster, 25k stars, 92.5%, 1.2B params,
# 10x, 300ms, 0.025 RTF, 600+ languages
_METRIC_RE = re.compile(
    r"\b(\d[\d,\.]*"              # leading number (with optional commas/decimals)
    r"(?:[xX%kKmMbB+]|"          # unit suffix: x, %, k, m, b, +
    r"(?:\s*(?:ms|ns|s|GB|MB|TB|params|tokens|languages?|stars?|repos?))\b"
    r")?)"
)


def apply_linkedin_bold(post_text: str) -> str:
    """
    Apply Unicode bold to:
      1. The first (hook) line — entirely bolded
      2. Standalone metric phrases in the body (numbers with units)

    Leaves hashtags, URLs, and emoji untouched.
    Skips posts that already contain Unicode bold characters.
    """
    if not post_text or not post_text.strip():
        return post_text

    # Skip if already bolded (idempotent)
    if any(c in post_text for c in ("𝗔", "𝗮", "𝟬")):
        return post_text

    lines = post_text.split("\n")

    # Bold the hook line (first non-empty line)
    hook_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if hook_idx is not None:
        lines[hook_idx] = to_bold(lines[hook_idx])

    # Bold metric phrases in the remaining body lines
    # but NOT in lines that are pure hashtag lines or URL lines
    for i in range(hook_idx + 1 if hook_idx is not None else 1, len(lines)):
        ln = lines[i]
        # Skip hashtag lines, URL-only lines, and empty lines
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("http"):
            continue
        if all(word.startswith("#") or word.startswith("http") for word in stripped.split()):
            continue
        lines[i] = _METRIC_RE.sub(lambda m: to_bold(m.group()), ln)

    return "\n".join(lines)
