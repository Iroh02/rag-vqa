"""
FastAPI backend for Open-Book VQA demo.

Endpoints:
    GET  /health          — model load status
    POST /infer           — run full RAG-VQA pipeline on uploaded image

Usage:
    python api.py                        # default port 8000
    python api.py --port 8000
    python api.py --no_rerank            # disable cross-encoder
    python api.py --no_caption           # disable BLIP-2 caption
"""

import argparse
import base64
import io
import time
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Open-Book VQA API")

# Allow the HTML demo (any localhost origin) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── lazy globals ──────────────────────────────────────────────────────────────
_model = _processor = _retriever = None
_models_loading = False
_cfg = {"rerank": True, "caption": True, "default_template": "current_prompt"}


def _load_models():
    global _model, _processor, _retriever, _models_loading
    if _model is not None:
        return
    _models_loading = True
    from src.model import load_model
    from src.rag_retriever import RAGRetriever

    print("Loading BLIP-2 (4-bit)...")
    _model, _processor = load_model(quantize=True)

    print("Loading CLIP + FAISS...")
    _retriever = RAGRetriever(device="cpu")
    _retriever.load_clip()
    _retriever.load_index()

    if _cfg["rerank"]:
        _retriever._load_cross_encoder()

    _models_loading = False
    print("All models ready.")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from src.prompt_templates import TEMPLATES
    return {
        "status":           "ready" if _model is not None else "not_loaded",
        "loading":          _models_loading,
        "rerank":           _cfg["rerank"],
        "caption":          _cfg["caption"],
        "prompt_templates": TEMPLATES,
        "default_template": _cfg.get("default_template", "current_prompt"),
    }


def _question_relevance(query_q: str, retrieved_q: str) -> float:
    """Jaccard similarity over content words (ignoring stopwords)."""
    stop = {"is","the","a","an","this","that","what","where","when","how","why","who",
            "in","on","of","to","at","for","with","do","does","are","there","it","be"}
    def toks(s):
        import re
        return {t for t in re.findall(r"[a-z]+", s.lower()) if t not in stop}
    A, B = toks(query_q), toks(retrieved_q)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(3),
    alpha: float = Form(1.0),
    tau: float = Form(0.0),
    rerank: bool = Form(False),
    use_caption: bool = Form(True),
    caption_weight: float = Form(0.0),
    use_answer_prior: bool = Form(False),
    filter_hints: bool = Form(False),
    filter_threshold: float = Form(0.15),
    prompt_template: str = Form("current_prompt"),
    type_gate: bool = Form(False),
):
    """Run the full pipeline:
        1. baseline answer (no retrieval)
        2. (optional) generate BLIP-2 caption — used for both retrieval *and* prompt
        3. caption-augmented retrieval over CLIP+FAISS
        4. τ-gate: if top score < τ, fall back to baseline
        5. RAG answer with the chosen prompt_template
    """
    from src.model import generate_answer, generate_answer_with_context, generate_caption, generate_with_template
    from src.prompt_templates import build_rag_prompt, compute_prompt_helpers

    t0 = time.time()
    timing = {}

    if _model is None:
        _load_models()

    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 1. Baseline (no retrieval)
    t = time.time()
    baseline = generate_answer(_model, _processor, pil_image, question)
    timing["baseline"] = round(time.time() - t, 2)

    # 2. Generate caption FIRST so it can feed retrieval
    caption = None
    if use_caption and _cfg["caption"]:
        t = time.time()
        caption = generate_caption(_model, _processor, pil_image)
        timing["caption"] = round(time.time() - t, 2)

    # 3. Caption-augmented retrieval
    t = time.time()
    do_rerank = rerank and _cfg["rerank"]
    cap_w = caption_weight if caption else 0.0
    retrieved = _retriever.retrieve(
        pil_image, question, top_k=top_k, alpha=alpha,
        rerank=do_rerank, caption=caption, caption_weight=cap_w,
    )
    timing["retrieval"] = round(time.time() - t, 2)

    # 4. τ-gate
    top_img_score = retrieved[0].get("img_score", 0.0) if retrieved else 0.0
    rag_used = top_img_score >= tau

    # 4b. Optional question-relevance filter on retrieved hints (demo improvement,
    # not in the reported benchmark). Drops hints whose question shares too few
    # content words with the user's question. If everything is filtered out, we
    # use the empty list — the prompt becomes caption + question only.
    retrieved_for_prompt = retrieved
    filtered_count = 0
    if filter_hints and retrieved:
        scored = [(r, _question_relevance(question, r.get("question", ""))) for r in retrieved]
        retrieved_for_prompt = [r for r, s in scored if s >= filter_threshold]
        filtered_count = len(retrieved) - len(retrieved_for_prompt)

    # Candidate answer prior (computed from the retrieved-for-prompt set)
    candidate_answers = []
    most_common_answer = ""
    prior_line = None
    if retrieved_for_prompt:
        from collections import Counter
        candidate_answers = [
            (r.get("best_answer") or r.get("answer", "")) for r in retrieved_for_prompt
        ]
        counts = Counter(a for a in candidate_answers if a)
        most_common_answer = counts.most_common(1)[0][0] if counts else ""
        if use_answer_prior and candidate_answers:
            prior_line = (
                f"Candidate answers from similar examples: {', '.join(candidate_answers)}. "
                f"Choose the best if one matches the image; otherwise answer normally."
            )

    # 5. RAG answer — uses the chosen prompt template
    rag_ans = None
    rag_prompt_used = ""
    if rag_used:
        t = time.time()
        if retrieved_for_prompt:
            cap_for_prompt = caption if (use_caption and prompt_template == "caption_aware") else None
            try:
                rag_ans, rag_prompt_used = generate_with_template(
                    _model, _processor, pil_image, question, retrieved_for_prompt,
                    template_name=prompt_template, caption=cap_for_prompt,
                    max_new_tokens=10,
                )
            except ValueError:
                # Unknown template name — fall back to current prompt
                rag_ans, rag_prompt_used = generate_with_template(
                    _model, _processor, pil_image, question, retrieved_for_prompt,
                    template_name="current_prompt", caption=None, max_new_tokens=10,
                )
            # Optionally inject the answer-prior preamble (legacy use_answer_prior path)
            if use_answer_prior and prior_line:
                # Re-generate with prior prepended via the legacy context path
                rag_ans = generate_answer_with_context(
                    _model, _processor, pil_image, question, retrieved_for_prompt,
                    caption=(prior_line if not caption else f"{caption}\n{prior_line}"),
                )
                rag_prompt_used = f"{prior_line}\n{rag_prompt_used}"
        else:
            # All hints filtered — caption + question only, or pure baseline
            if caption:
                rag_ans = generate_answer_with_context(
                    _model, _processor, pil_image, question, [], caption=caption,
                )
                rag_prompt_used = f"Image: {caption}\nQ: {question} A:"
            else:
                rag_ans = generate_answer(_model, _processor, pil_image, question)
                rag_prompt_used = f"Question: {question} Answer:"
        timing["rag_generation"] = round(time.time() - t, 2)

    # Prompt-for-display: the actual prompt sent to BLIP-2 (from the template),
    # plus a header noting the filter status.
    if rag_used:
        header = []
        if filter_hints and retrieved_for_prompt:
            header.append(f"# Filter ON — {filtered_count} of {len(retrieved)} retrieved hints dropped")
        elif filter_hints and not retrieved_for_prompt:
            header.append(f"# Filter ON — all {len(retrieved)} hints filtered; using caption + question only")
        prompt_text = ("\n".join(header) + ("\n" if header else "")) + (rag_prompt_used or "")
    else:
        prompt_text = f"[Gated — top score {top_img_score:.3f} < τ={tau:.2f}]\nQ: {question} A:"

    # Helpers (best question-matched, candidate list) for the UI
    helpers = compute_prompt_helpers(question, retrieved_for_prompt or retrieved)

    # Reasoning chain (timeline of inference steps)
    reasoning = [
        {"step": "baseline",  "label": "🎯 Baseline answer (no retrieval)",
         "value": baseline, "elapsed": timing.get("baseline")},
    ]
    if caption:
        reasoning.append({"step": "caption", "label": "🔍 BLIP-2 sees",
                          "value": caption, "elapsed": timing.get("caption")})
    reasoning.append({"step": "retrieval",
                      "label": f"📚 Retrieved {len(retrieved)} similar Q&A pairs (top score {top_img_score:.3f})",
                      "value": "; ".join(f"{r.get('question','')} → {r.get('best_answer') or r.get('answer','')}" for r in retrieved),
                      "elapsed": timing.get("retrieval")})
    if rag_used:
        reasoning.append({"step": "answer", "label": "✨ Final answer (with hints + caption)",
                          "value": rag_ans, "elapsed": timing.get("rag_generation")})
    else:
        reasoning.append({"step": "gated", "label": f"⚠ τ-gate: top score {top_img_score:.3f} < τ={tau:.2f} — fell back to baseline",
                          "value": baseline, "elapsed": 0})

    thumb = pil_image.copy()
    thumb.thumbnail((400, 400))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    elapsed = round(time.time() - t0, 2)

    # Type-conditional gate (validated +1.00 pts on the 100-sample eval).
    YN_PREFIXES = ("is ", "are ", "do ", "does ", "can ", "has ", "have ", "had ",
                   "was ", "were ", "will ", "could ", "should ", "would ")
    is_yn = question.lower().lstrip().startswith(YN_PREFIXES)
    type_gate_applied = bool(type_gate and is_yn)
    if type_gate_applied:
        post_router = baseline
        type_gate_reason = "yes/no detected → baseline (RAG can hurt yes/no by ~3 pts on average)"
    else:
        post_router = rag_ans if rag_used else baseline
        type_gate_reason = ("type_gate off" if not type_gate
                            else "open-ended → RAG kept")

    # Final-system layer: evidence-aware suggestion + normalization
    from src.answer_normalizer import normalize as _normalize_answer

    # Evidence-aware suggestion (rule-based)
    eaw_suggestion = post_router
    eaw_reason = "no_override"
    qn = question.strip().lower()
    if retrieved:
        for r in retrieved[:3]:
            rq = (r.get("question") or "").strip().lower()
            if rq == qn and r.get("img_score", 0.0) >= 0.90:
                eaw_suggestion = r.get("best_answer") or r.get("answer", "")
                eaw_reason = "exact_q_match_img>=0.90"
                break
        if eaw_reason == "no_override" and most_common_answer:
            cnt_local = Counter(a for a in candidate_answers if a)
            if cnt_local and cnt_local.most_common(1)[0][1] >= 2:
                eaw_suggestion = most_common_answer
                eaw_reason = "majority>=2_in_retrieved"

    # Normalize the post-router answer using retrieved candidates
    normalized_ans, normalize_reason = _normalize_answer(post_router, candidate_answers)

    # The single answer the demo should show as "final"
    final_ans = normalized_ans or post_router

    return JSONResponse({
        "baseline":           baseline,
        "rag":                rag_ans if rag_used else baseline,
        "raw_rag":            rag_ans if rag_used else baseline,
        "post_router":        post_router,
        "evidence_suggestion": eaw_suggestion,
        "evidence_reason":    eaw_reason,
        "normalized":         normalized_ans,
        "normalize_reason":   normalize_reason,
        "final_answer":       final_ans,
        "is_yes_no":          is_yn,
        "type_gate_applied":  type_gate_applied,
        "type_gate_reason":   type_gate_reason,
        "rag_used":           rag_used,
        "top_img_score":      round(top_img_score, 4),
        "caption":            caption or "",
        "candidate_answers":  candidate_answers,
        "most_common_answer": most_common_answer,
        "answer_prior_used":  use_answer_prior,
        "best_question_matched_question": helpers["best_question_matched_question"],
        "best_question_matched_answer":   helpers["best_question_matched_answer"],
        "best_question_match_score":      helpers["best_question_match_score"],
        "prompt_template":   prompt_template,
        "reasoning":          reasoning,
        "timing":             timing,
        "retrieved": [
            {
                "question":  r.get("question", ""),
                "answer":    r.get("best_answer") or r.get("answer", ""),
                "img_score": round(r.get("img_score", 0), 4),
                "q_score":   round(r.get("q_score",   0), 4),
                "cap_score": round(r.get("cap_score", 0), 4),
                "ce_score":  round(r.get("ce_score",  0), 4),
                "score":     round(r.get("score",     0), 4),
                "q_relevance": round(_question_relevance(question, r.get("question", "")), 3),
                "kept":       r in retrieved_for_prompt,
            }
            for r in retrieved
        ],
        "prompt":            prompt_text,
        "elapsed_sec":       elapsed,
        "image_b64":         img_b64,
        "filtered_count":    filtered_count,
        "filter_threshold":  filter_threshold if filter_hints else None,
        "config": {
            "top_k":            top_k,
            "alpha":            alpha,
            "tau":              tau,
            "caption_weight":   cap_w,
            "rerank":           do_rerank,
            "caption":          bool(caption),
            "use_answer_prior": use_answer_prior,
            "filter_hints":     filter_hints,
            "prompt_template":  prompt_template,
            "type_gate":        type_gate,
        },
    })


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int,  default=8000)
    parser.add_argument("--no_rerank",  action="store_true")
    parser.add_argument("--no_caption", action="store_true")
    parser.add_argument("--preload",    action="store_true", help="Load models at startup")
    args = parser.parse_args()

    _cfg["rerank"]  = not args.no_rerank
    _cfg["caption"] = not args.no_caption

    if args.preload:
        _load_models()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
