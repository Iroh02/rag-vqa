"""Prompt templates for RAG-VQA inference.

Provides:
  - compute_prompt_helpers(question, retrieved) -> dict of all derived fields
  - build_rag_prompt(template_name, question, retrieved, caption=None) -> str
  - TEMPLATES: list of all available template names

Templates 0-8 from the prompt-ablation specification.
"""
from __future__ import annotations

import re
from typing import List, Optional

TEMPLATES = [
    "current_prompt",          # 0 — control / existing
    "short_evidence",          # 1
    "candidate_constrained",   # 2
    "question_match_priority", # 3
    "minimal_fewshot",         # 4
    "image_first",             # 5
    "evidence_noisy_warning",  # 6
    "caption_aware",           # 7 — needs caption
    "direct_evidence_override",# 8
]

_STOPWORDS = {
    "is","the","a","an","this","that","what","where","when","how","why","who",
    "in","on","of","to","at","for","with","do","does","are","there","it","be",
    "was","were","has","have","had","can","could","should","would","will",
}


def _toks(s: str) -> set:
    return {t for t in re.findall(r"[a-z]+", (s or "").lower()) if t not in _STOPWORDS}


def _jaccard(s1: str, s2: str) -> float:
    a, b = _toks(s1), _toks(s2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _retrieved_q(r: dict) -> str:
    return r.get("question") or r.get("retrieved_question") or ""


def _retrieved_a(r: dict) -> str:
    return r.get("best_answer") or r.get("answer") or ""


def compute_prompt_helpers(question: str, retrieved: List[dict]) -> dict:
    """Compute all derived fields used by templates."""
    rqs = [_retrieved_q(r) for r in retrieved[:3]]
    ras = [_retrieved_a(r) for r in retrieved[:3]]

    bullets_lines = []
    for q, a in zip(rqs, ras):
        if q:
            bullets_lines.append(f"- Q: {q}\n  A: {a}")
    retrieved_qas_bullets = "\n".join(bullets_lines) if bullets_lines else "- (none)"

    numbered_lines = []
    for i, (q, a) in enumerate(zip(rqs, ras), 1):
        if q:
            numbered_lines.append(f"{i}. Q: {q}\n   A: {a}")
    retrieved_qas_numbered = "\n".join(numbered_lines) if numbered_lines else "1. (none)"

    plain_lines = []
    for q, a in zip(rqs, ras):
        if q:
            plain_lines.append(f"Q: {q} A: {a}")
    retrieved_qas_plain = "\n".join(plain_lines) if plain_lines else ""

    # Candidate answers — deduped, ordered by retrieval rank
    seen = set()
    candidates = []
    for a in ras:
        a_norm = (a or "").strip()
        if a_norm and a_norm.lower() not in seen:
            seen.add(a_norm.lower())
            candidates.append(a_norm)
    candidate_answers_str = ", ".join(candidates) if candidates else "none"

    # Best question-matched
    best_score = -1.0
    best_q = ""
    best_a = ""
    for q, a in zip(rqs, ras):
        if not q:
            continue
        s = _jaccard(question, q)
        if s > best_score:
            best_score = s
            best_q = q
            best_a = a

    if not best_q and rqs:
        best_q = rqs[0]
        best_a = ras[0]
        best_score = 0.0

    return {
        "retrieved_qas_bullets":   retrieved_qas_bullets,
        "retrieved_qas_numbered":  retrieved_qas_numbered,
        "retrieved_qas_plain":     retrieved_qas_plain,
        "candidate_answers":       candidates,
        "candidate_answers_str":   candidate_answers_str,
        "best_question_matched_question": best_q,
        "best_question_matched_answer":   best_a,
        "best_question_match_score":      round(best_score, 4),
        "retrieved_questions": rqs,
        "retrieved_answers":   ras,
    }


def build_rag_prompt(
    template_name: str,
    question: str,
    retrieved: List[dict],
    caption: Optional[str] = None,
) -> str:
    """Build a prompt string for the given template.

    The CURRENT prompt (template 0) matches the format used in the headline
    59.33% number — no caption, naive few-shot.
    """
    h = compute_prompt_helpers(question, retrieved)
    bullets   = h["retrieved_qas_bullets"]
    numbered  = h["retrieved_qas_numbered"]
    plain     = h["retrieved_qas_plain"]
    cand_str  = h["candidate_answers_str"]
    best_q    = h["best_question_matched_question"] or "(none)"
    best_a    = h["best_question_matched_answer"]   or "none"

    if template_name == "current_prompt":
        # Matches src/model.py generate_answer_with_context (no caption — headline config)
        lines = []
        for q, a in zip(h["retrieved_questions"], h["retrieved_answers"]):
            if q:
                lines.append(f"Q: {q} A: {a}")
        lines.append(f"Q: {question} A:")
        return "\n".join(lines)

    if template_name == "short_evidence":
        return (
            f"Question: {question}\n\n"
            f"Helpful evidence from similar images:\n{bullets}\n\n"
            f"Best matching evidence answer: {best_a}\n\n"
            f"Answer the question using the image. If the best matching evidence is relevant, use it.\n"
            f"Answer:"
        )

    if template_name == "candidate_constrained":
        return (
            f"Question: {question}\n\n"
            f"Similar-image evidence:\n{bullets}\n\n"
            f"Candidate answers:\n{cand_str}\n\n"
            f"Choose the best answer from the candidate answers if one matches the image.\n"
            f"If none match, answer normally.\n\n"
            f"Answer:"
        )

    if template_name == "question_match_priority":
        return (
            f"Question: {question}\n\n"
            f"Retrieved evidence from visually similar images:\n{numbered}\n\n"
            f"Best question-matched evidence:\n"
            f"Q: {best_q}\n"
            f"A: {best_a}\n\n"
            f"Instruction:\n"
            f"If the best question-matched evidence is relevant to the current image, "
            f"strongly prefer that answer.\n"
            f"Otherwise answer from the image.\n\n"
            f"Answer:"
        )

    if template_name == "minimal_fewshot":
        return f"{plain}\nQ: {question}\nA:" if plain else f"Q: {question}\nA:"

    if template_name == "image_first":
        return (
            f"Look at the image and answer the question.\n\n"
            f"Question: {question}\n\n"
            f"Relevant examples from similar images:\n{bullets}\n\n"
            f"Use the examples only if they help.\n"
            f"Give a short answer.\n\n"
            f"Answer:"
        )

    if template_name == "evidence_noisy_warning":
        return (
            f"Question: {question}\n\n"
            f"The following examples come from visually similar images. "
            f"They may be helpful, but they may also be noisy:\n{bullets}\n\n"
            f"Candidate answers:\n{cand_str}\n\n"
            f"Use the current image as the main source of truth.\n"
            f"Use retrieved evidence only when it clearly matches the question.\n\n"
            f"Short answer:"
        )

    if template_name == "caption_aware":
        cap = caption or "(no caption)"
        return (
            f"Image summary:\n{cap}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{bullets}\n\n"
            f"Best matching evidence answer:\n{best_a}\n\n"
            f"Use the image summary, the image, and the retrieved evidence to answer briefly.\n"
            f"Answer:"
        )

    if template_name == "direct_evidence_override":
        return (
            f"Question: {question}\n\n"
            f"Retrieved Q&A evidence:\n{bullets}\n\n"
            f"Most relevant retrieved answer:\n{best_a}\n\n"
            f"If the most relevant retrieved answer matches the image, output it directly.\n"
            f"Otherwise answer from the image.\n\n"
            f"Answer:"
        )

    raise ValueError(f"Unknown template: {template_name}")
