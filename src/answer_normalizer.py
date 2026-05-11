"""Lightweight answer normalizer for RAG-VQA.

Goal: turn verbose / over-articled answers into VQA-style canonical short form
WITHOUT inventing new information. The normalizer only collapses an answer to
a known candidate when the candidate appears as a whole word in the original.

Returns (normalized_answer, reason) so the demo can show why a change happened.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

ARTICLES = {"a", "an", "the"}
VERBOSE_THRESHOLD = 3   # >= this many words → eligible for candidate collapse
LEADING_FILLERS = {
    "it", "is", "this", "that", "they", "are", "there",
    "it's", "thats", "that's", "its",
}


def _strip_punct(s: str) -> str:
    """Remove punctuation but keep word chars, spaces, internal hyphens."""
    return re.sub(r"[^\w\s\-]", " ", s).strip()


def _strip_leading_articles(words: List[str]) -> List[str]:
    while words and words[0] in ARTICLES:
        words = words[1:]
    return words


def _strip_leading_fillers(words: List[str]) -> List[str]:
    """e.g., 'it is a dog' → 'dog'  (drops 'it is a')."""
    while words and (words[0] in LEADING_FILLERS or words[0] in ARTICLES):
        words = words[1:]
    return words


def normalize(answer: str, candidates: Optional[List[str]] = None) -> Tuple[str, str]:
    """Return (normalized_answer, reason).

    Steps:
      1. lowercase + strip punctuation
      2. drop leading 'it is/that is/the/a/an'
      3. if answer has >=3 words AND a candidate appears as a whole word in it
         → collapse to that candidate
      4. otherwise return cleaned form
    """
    if answer is None:
        return "", "empty"
    raw = answer.strip()
    if not raw:
        return "", "empty"

    a_lower = _strip_punct(raw.lower())
    words   = a_lower.split()
    if not words:
        return raw, "no_change_punct_only"

    cleaned_words = _strip_leading_fillers(words)
    cleaned       = " ".join(cleaned_words) if cleaned_words else a_lower

    # Candidate-aware verbose collapse: if 3+ words and a candidate appears
    # as a whole word in the answer, prefer the candidate.
    if candidates and len(words) >= VERBOSE_THRESHOLD:
        # Normalize candidate strings the same way the input was normalized
        cand_norms = []
        for c in candidates:
            if not c:
                continue
            cn = _strip_punct(c.strip().lower())
            if cn:
                cand_norms.append(cn)

        # Search for a whole-word candidate match in the cleaned answer
        for cn in cand_norms:
            pat = r"\b" + re.escape(cn) + r"\b"
            if re.search(pat, a_lower):
                return cn, f"verbose ({len(words)}w) → matched candidate '{cn}'"

    # Single-word or short cleaned form: pass through (with article strip)
    if cleaned == a_lower and cleaned == raw.lower():
        return cleaned, "no_change"

    if cleaned == a_lower:
        return cleaned, "stripped_punctuation"

    return cleaned, "stripped_leading_articles_or_fillers"


def normalize_with_audit(answer: str, candidates: Optional[List[str]] = None) -> dict:
    """Verbose form: returns the normalized answer plus structured audit info."""
    norm, reason = normalize(answer, candidates)
    return {
        "raw":         answer,
        "normalized":  norm,
        "reason":      reason,
        "changed":     norm.strip().lower() != (answer or "").strip().lower(),
        "candidates":  list(candidates or []),
    }


# Quick self-test when run as module
if __name__ == "__main__":
    cases = [
        ("Watching the skateboarder",  ["picnic table", "down", "watching"]),
        ("It is a dog",                ["dog", "cat", "person"]),
        ("a young boy holding a baseball bat", ["boy", "bat", "baseball"]),
        ("blueblue",                   ["blue"]),
        ("yes",                        ["no", "yes"]),
        ("Looking at the crowd",       ["down", "up", "watching"]),
        ("",                           ["a"]),
        ("This is a bedroom",          ["bedroom", "kitchen"]),
    ]
    for a, cs in cases:
        n, r = normalize(a, cs)
        print(f"{a!r:<45}  candidates={cs}\n   -> {n!r:<25} ({r})")
