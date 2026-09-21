"""Pure-Python text chunking utilities — no PDF dependency.

The chunker works at the **character** level with a configurable sliding window.
Word-boundary snapping is applied so chunks never split in the middle of a word,
which keeps embedding quality high.

Typical call
------------
    from pdf_semantic_search.pdf.chunker import chunk_text

    chunks = chunk_text("A long article …", chunk_size=512, overlap=64)
"""
from __future__ import annotations

import re


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split *text* into overlapping, word-boundary-aligned chunks.

    Algorithm
    ---------
    1. Normalise whitespace (collapse runs of spaces/newlines to a single space).
    2. Slide a window of *chunk_size* characters across the text, advancing by
       ``chunk_size - overlap`` characters each step.
    3. If the window end falls inside a word, walk *backwards* up to
       ``SNAP_WINDOW`` characters to find the nearest space, preventing
       mid-word cuts.  If no space is found within the snap window the hard
       boundary is kept (handles languages without spaces gracefully).
    4. Strip leading/trailing whitespace from each chunk and discard empties.

    Args:
        text:       Raw text to split.  Empty strings return ``[]``.
        chunk_size: Target maximum length of each chunk in characters.
                    Must be > 0.
        overlap:    Number of characters of context shared between consecutive
                    chunks.  Must be ≥ 0 and < *chunk_size*.

    Returns:
        Ordered list of non-empty text chunks.  The final chunk may be shorter
        than *chunk_size*.

    Raises:
        ValueError: If *chunk_size* ≤ 0 or *overlap* ≥ *chunk_size*.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must be in [0, chunk_size), got overlap={overlap}, chunk_size={chunk_size}."
        )

    # 1. Normalise whitespace
    text = _normalise(text)

    if not text:
        return []

    # 2. Special case: text fits in a single chunk
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # 3. Snap to nearest word boundary within SNAP_WINDOW chars
            end = _snap_to_word_boundary(text, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step

        # Guard: if snap pushed end backwards past start + step we could
        # loop forever on very long words.  Force progress in that case.
        if start >= end:
            start = end

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# How far back (chars) we're willing to walk to find a word boundary
_SNAP_WINDOW = 80


def _normalise(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def _snap_to_word_boundary(text: str, pos: int) -> int:
    """Return the position of the nearest space at or before *pos*.

    Searches backwards up to ``_SNAP_WINDOW`` characters.  Returns *pos*
    unchanged if no space is found (hard boundary kept).
    """
    search_start = max(0, pos - _SNAP_WINDOW)
    # rfind searches right-to-left — finds the nearest space
    space_pos = text.rfind(" ", search_start, pos)
    return space_pos if space_pos != -1 else pos
