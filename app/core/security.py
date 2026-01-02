"""
Security utilities for sanitizing user-facing content.

Prevents injection attacks in HITL interfaces by stripping potentially
dangerous markup from explanations and user-facing text.
"""

import re


def sanitize_explanation(text: str) -> str:
    """
    Sanitize explanation text to prevent injection in HITL interface.

    Strips:
    - Markdown links: [text](url) -> text
    - All HTML tags: <tag>content</tag> -> content

    Args:
        text: Raw explanation text that may contain markup

    Returns:
        Sanitized text with markup removed
    """
    if not text:
        return ""

    # Strip Markdown links: [text](url) -> text
    # Matches [text](url) or [text](url "title") and extracts just the text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Strip all HTML tags: <tag>content</tag> -> content
    # This regex matches any HTML tag (opening, closing, self-closing)
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()
