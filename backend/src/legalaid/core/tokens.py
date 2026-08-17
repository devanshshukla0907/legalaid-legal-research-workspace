"""Token estimation.

Ollama returns authoritative `prompt_eval_count` / `eval_count` values for every generation,
and those are what the UI displays for model runs. This module provides an *estimate* for
cases where no model call has happened yet - chunk sizing, context budgeting, and the chunk
inspector's token column.

The estimator is deliberately labelled as an estimate everywhere it surfaces, so no figure in
the UI is presented as exact when it is not.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]")

# Llama-family tokenizers average roughly 1.3-1.4 tokens per whitespace word for English legal
# prose, which is denser than plain text because of citations, numbers and punctuation.
_TOKENS_PER_PIECE = 1.05
_CHARS_PER_TOKEN_FALLBACK = 3.7


def estimate_tokens(text: str) -> int:
    """Estimate token count for a piece of text."""
    if not text:
        return 0
    pieces = _WORD_RE.findall(text)
    if not pieces:
        return max(1, round(len(text) / _CHARS_PER_TOKEN_FALLBACK))
    by_pieces = len(pieces) * _TOKENS_PER_PIECE
    by_chars = len(text) / _CHARS_PER_TOKEN_FALLBACK
    return max(1, round((by_pieces + by_chars) / 2))


def estimate_tokens_many(texts: list[str]) -> int:
    return sum(estimate_tokens(text) for text in texts)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to approximately `max_tokens`, returning (text, was_truncated)."""
    if max_tokens <= 0:
        return "", bool(text)
    if estimate_tokens(text) <= max_tokens:
        return text, False
    approx_chars = int(max_tokens * _CHARS_PER_TOKEN_FALLBACK)
    if approx_chars >= len(text):
        return text, False
    cut = text[:approx_chars]
    boundary = max(cut.rfind("\n\n"), cut.rfind(". "), cut.rfind("\n"))
    if boundary > approx_chars * 0.6:
        cut = cut[: boundary + 1]
    return cut.rstrip(), True
