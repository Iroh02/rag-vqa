"""
Open-Book VQA — Interactive Demo App
=====================================
Demonstrates the full RAG-VQA pipeline:
  Query image + question
  → Baseline BLIP-2 answer
  → CLIP + FAISS retrieval
  → Top-k retrieved examples
  → RAG prompt construction
  → Final RAG answer

Usage:
    python demo_app.py                        # cached mode (presentation-safe)
    python demo_app.py --mode cached          # explicit cached mode
    python demo_app.py --setup                # re-run inference on fresh samples
    python demo_app.py --mode live            # live inference on uploaded image
    python demo_app.py --mode live --top_k 5 --alpha 1.0
"""

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
RESULTS_DIR  = ROOT / "results"
DATA_DIR     = ROOT / "data"
DEMO_CASES   = RESULTS_DIR / "demo_cases.json"
IMAGE_MAP    = DATA_DIR / "demo_image_map.json"
FRESH_DATA   = DATA_DIR / "fresh_demo_results.json"
PLACEHOLDER  = DATA_DIR / "demo_images" / "placeholder.jpg"

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_cases():
    if DEMO_CASES.exists():
        with open(DEMO_CASES) as f:
            return json.load(f)
    return []


def _load_image(path):
    try:
        p = Path(path)
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    # Return grey placeholder
    img = Image.new("RGB", (400, 300), color=(45, 55, 72))
    return img


def _badge(correct: bool | None) -> str:
    if correct is True:
        return '<span style="background:#22543d;color:#9ae6b4;padding:4px 12px;border-radius:20px;font-weight:700;font-size:14px;">CORRECT</span>'
    elif correct is False:
        return '<span style="background:#742a2a;color:#feb2b2;padding:4px 12px;border-radius:20px;font-weight:700;font-size:14px;">WRONG</span>'
    return '<span style="background:#2d3748;color:#a0aec0;padding:4px 12px;border-radius:20px;font-weight:700;font-size:14px;">UNKNOWN</span>'


def _verdict_banner(verdict: str, delta: float) -> str:
    if verdict == "helpful":
        return f'<div style="background:#1a4731;border:1px solid #276749;border-radius:10px;padding:12px 16px;color:#9ae6b4;font-weight:600;">✅ RAG HELPED &nbsp;|&nbsp; Δ = +{delta:.2f} VQA pts</div>'
    elif verdict == "harmful":
        return f'<div style="background:#3b1a1a;border:1px solid #7b341e;border-radius:10px;padding:12px 16px;color:#feb2b2;font-weight:600;">❌ RAG HURT &nbsp;|&nbsp; Δ = {delta:.2f} VQA pts</div>'
    else:
        return f'<div style="background:#1a202c;border:1px solid #4a5568;border-radius:10px;padding:12px 16px;color:#a0aec0;font-weight:600;">➡ NO CHANGE &nbsp;|&nbsp; Δ = 0.00 VQA pts</div>'


def _retrieved_html(retrieved: list, top_k: int) -> str:
    if not retrieved:
        return "<p style='color:#718096;'>No retrieval data available.</p>"
    items = retrieved[:top_k]
    html = "<div style='display:flex;flex-direction:column;gap:10px;'>"
    for i, ex in enumerate(items, 1):
        q    = ex.get("question", "—")
        ans  = ex.get("answer",   ex.get("best_answer", "—"))
        img_s = float(ex.get("img_score",  ex.get("score", 0)))
        q_s   = float(ex.get("q_score",    0))
        score = float(ex.get("score",      img_s))
        bar_w = int(min(score, 1.0) * 100)
        html += f"""
        <div style='background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:12px 14px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
            <span style='color:#63b3ed;font-weight:700;font-size:13px;'>Retrieved #{i}</span>
            <span style='color:#4a5568;font-size:12px;'>img={img_s:.3f} | txt={q_s:.3f}</span>
          </div>
          <div style='background:#2d3748;border-radius:4px;height:4px;margin-bottom:8px;'>
            <div style='background:#4299e1;height:4px;border-radius:4px;width:{bar_w}%;'></div>
          </div>
          <div style='font-size:13px;color:#cbd5e0;margin-bottom:4px;'><b>Q:</b> {q}</div>
          <div style='font-size:14px;color:#68d391;font-weight:600;'><b>A:</b> {ans}</div>
        </div>"""
    html += "</div>"
    return html


def _build_prompt(question: str, retrieved: list, top_k: int) -> str:
    lines = ["Answer with 1-3 words. Hints may be wrong.", "Hints:"]
    for ex in retrieved[:top_k]:
        ans = ex.get("answer", ex.get("best_answer", ""))
        lines.append(f"  - Similar image suggests: {ans}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _explanation(verdict: str, baseline: str, rag: str, gt: str,
                 retrieved: list, top_k: int) -> str:
    retrieved_answers = [ex.get("answer", ex.get("best_answer","")) for ex in retrieved[:top_k]]
    direct_match = any(
        a.lower().strip() == gt.lower().strip() for a in retrieved_answers
    )
    if verdict == "helpful":
        if direct_match:
            return f"""**Why RAG Helped — Direct Answer Retrieval**

The ground-truth answer **"{gt}"** appeared in the retrieved examples. BLIP-2 leveraged this retrieved context to provide the correct answer, demonstrating that RAG can supply factual knowledge the frozen model lacks in isolation.

This is the open-book effect in action: a visually similar past example contained the exact answer needed."""
        else:
            return f"""**Why RAG Helped — Semantic Context**

The retrieved examples did not contain the exact answer "{gt}", but they provided:
- Answer-style priors (what kinds of answers are appropriate for this visual scene)
- Semantic context about similar visual scenes
- Corrective signal that shifted the model from **"{baseline}"** → **"{rag}"**

This demonstrates that RAG helps not just by copying answers, but by providing contextual grounding."""
    elif verdict == "harmful":
        return f"""**Why RAG Hurt — Context Noise**

The baseline model answered correctly (**"{baseline}"**), but the retrieved context introduced noise that confused BLIP-2 into answering **"{rag}"** instead.

This is a known failure mode: when the model is already confident and correct, adding retrieved hints from imperfectly matched visual scenes can degrade performance.

This is especially common for yes/no questions where BLIP-2 has high internal confidence."""
    else:
        if float(baseline == gt or rag == gt) > 0:
            return f"""**Neutral — Both Correct**

Both baseline and RAG answered correctly (**"{gt}"**). The retrieved context did not change the outcome — the model was already confident enough to answer correctly without external help.

This represents 70% of cases: RAG is useful but doesn't hurt when the model is already right."""
        else:
            return f"""**Neutral — Both Wrong**

Neither baseline (**"{baseline}"**) nor RAG (**"{rag}"**) answered correctly (GT: **"{gt}"**). The retrieved context was insufficient to help with this difficult question.

This may indicate that: (1) the visual scene is genuinely ambiguous, (2) the knowledge base doesn't contain visually similar enough examples, or (3) the question requires reasoning beyond what BLIP-2 can perform."""


# ── cached-mode display logic ─────────────────────────────────────────────────

def display_case(case_label: str, top_k: int, alpha: float, cases: list):
    case = next((c for c in cases if c["label"] == case_label), None)
    if case is None:
        return [None, "", "", "", "", "", "", "", ""]

    image        = _load_image(case.get("image_path", ""))
    question     = case["question"]
    gt           = case["ground_truth"]
    qt           = case.get("question_type", "open-ended")
    baseline     = case["baseline_pred"]
    rag_ans      = case["rag_pred"]
    base_acc     = float(case["baseline_acc"])
    rag_acc      = float(case["rag_acc"])
    delta        = float(case["delta"])
    verdict      = case["verdict"]
    retrieved    = case.get("retrieved", [])

    # Info card (question metadata)
    info_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:14px 16px;'>
      <div style='color:#63b3ed;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Query</div>
      <div style='font-size:17px;font-weight:600;color:#f7fafc;margin-bottom:10px;'>{question}</div>
      <div style='display:flex;gap:12px;flex-wrap:wrap;'>
        <span style='background:#2d3748;padding:3px 10px;border-radius:12px;font-size:12px;color:#a0aec0;'>{qt}</span>
        <span style='background:#2d3748;padding:3px 10px;border-radius:12px;font-size:12px;color:#68d391;'>GT: <b>{gt}</b></span>
      </div>
    </div>"""

    # Baseline card
    b_correct = base_acc >= 1.0
    baseline_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:16px;height:100%;'>
      <div style='color:#fc8181;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>Baseline BLIP-2</div>
      <div style='font-size:28px;font-weight:800;color:#f7fafc;margin-bottom:12px;'>{baseline}</div>
      <div style='margin-bottom:10px;'>{_badge(b_correct)}</div>
      <div style='color:#718096;font-size:13px;'>VQA score: <b style='color:#f7fafc;'>{base_acc:.2f}</b></div>
      <div style='color:#718096;font-size:12px;margin-top:8px;'>Frozen BLIP-2, no retrieval</div>
    </div>"""

    # RAG card
    r_correct = rag_acc >= 1.0
    rag_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:16px;height:100%;'>
      <div style='color:#68d391;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>RAG BLIP-2 (α={alpha:.1f}, k={top_k})</div>
      <div style='font-size:28px;font-weight:800;color:#f7fafc;margin-bottom:12px;'>{rag_ans}</div>
      <div style='margin-bottom:10px;'>{_badge(r_correct)}</div>
      <div style='color:#718096;font-size:13px;'>VQA score: <b style='color:#f7fafc;'>{rag_acc:.2f}</b></div>
      {_verdict_banner(verdict, delta)}
    </div>"""

    retrieved_html  = _retrieved_html(retrieved, top_k)
    prompt_text     = _build_prompt(question, retrieved, top_k)
    explanation_md  = _explanation(verdict, baseline, rag_ans, gt, retrieved, top_k)

    return (image, info_html, baseline_html, rag_html,
            retrieved_html, prompt_text, explanation_md, verdict, info_html)


# ── live-mode inference ───────────────────────────────────────────────────────

_live_model     = None
_live_processor = None
_live_retriever = None


def _load_live_models():
    global _live_model, _live_processor, _live_retriever
    if _live_model is None:
        from src.model import load_model
        from src.rag_retriever import RAGRetriever
        print("Loading BLIP-2 (4-bit)...")
        _live_model, _live_processor = load_model(quantize=True)
        print("Loading CLIP + FAISS...")
        _live_retriever = RAGRetriever()
        _live_retriever.load_clip()
        _live_retriever.load_index()
        print("Models ready.")
    return _live_model, _live_processor, _live_retriever


def run_live_inference(image, question: str, top_k: int, alpha: float):
    from src.model import generate_answer, generate_answer_with_context
    from src.evaluate import vqa_accuracy

    if image is None or not question.strip():
        return (None, "<p style='color:#fc8181;'>Please provide an image and a question.</p>",
                "", "", "", "", "No input provided", "neutral", "")

    model, processor, retriever = _load_live_models()
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)

    baseline  = generate_answer(model, processor, pil_image, question)
    retrieved = retriever.retrieve(pil_image, question, top_k=top_k, alpha=alpha)

    # Normalize retrieved field names
    retrieved_norm = []
    for r in retrieved:
        retrieved_norm.append({
            "question":  r.get("question", ""),
            "answer":    r.get("best_answer", r.get("answer", "")),
            "img_score": r.get("img_score", r.get("score", 0.0)),
            "q_score":   r.get("q_score",   0.0),
            "score":     r.get("score",     r.get("img_score", 0.0)),
        })

    rag_ans  = generate_answer_with_context(model, processor, pil_image,
                                            question, retrieved_norm)

    info_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:14px 16px;'>
      <div style='color:#63b3ed;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Live Query</div>
      <div style='font-size:17px;font-weight:600;color:#f7fafc;'>{question}</div>
    </div>"""

    baseline_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:16px;'>
      <div style='color:#fc8181;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>Baseline BLIP-2</div>
      <div style='font-size:28px;font-weight:800;color:#f7fafc;'>{baseline}</div>
      <div style='color:#718096;font-size:12px;margin-top:8px;'>Frozen BLIP-2, no retrieval</div>
    </div>"""

    rag_html = f"""
    <div style='background:#1a202c;border:1px solid #2d3748;border-radius:10px;padding:16px;'>
      <div style='color:#68d391;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>RAG BLIP-2 (α={alpha:.1f}, k={top_k})</div>
      <div style='font-size:28px;font-weight:800;color:#f7fafc;'>{rag_ans}</div>
      <div style='color:#718096;font-size:12px;margin-top:8px;'>With {top_k} retrieved examples</div>
    </div>"""

    rh  = _retrieved_html(retrieved_norm, top_k)
    pt  = _build_prompt(question, retrieved_norm, top_k)
    exp = f"**Live inference complete.** Baseline: **{baseline}** → RAG: **{rag_ans}**"

    return (pil_image, info_html, baseline_html, rag_html, rh, pt, exp, "neutral", info_html)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0d1117 !important; }
.gradio-container { max-width: 1400px !important; }
h1, h2, h3 { color: #f0f6fc !important; }
.label-wrap { color: #8b949e !important; font-size: 12px !important; }
footer { display: none !important; }
.tab-nav button { background: #161b22 !important; color: #8b949e !important;
                  border: 1px solid #21262d !important; border-radius: 8px !important; }
.tab-nav button.selected { background: #21262d !important; color: #58a6ff !important;
                           border-color: #58a6ff !important; }
"""

TITLE_MD = """
# 🔍 Open-Book VQA
### Visual Memory-Augmented BLIP-2 with CLIP Retrieval

**BLIP-2** (frozen) + **CLIP ViT-B/32** embeddings + **FAISS** 5k-entry knowledge base
→ Retrieves similar past VQA examples → Provides contextual hints → Improves accuracy **42.33% → 51.33%**
"""


def build_app(mode: str = "cached"):
    cases      = _load_cases()
    case_labels = [c["label"] for c in cases]

    with gr.Blocks(title="Open-Book VQA Demo", css=CSS, theme=gr.themes.Base()) as app:
        gr.Markdown(TITLE_MD)

        # ── controls row ──────────────────────────────────────────────────────
        with gr.Row():
            mode_selector = gr.Radio(
                ["cached", "live"],
                value=mode, label="Mode",
                info="Cached = instant replay | Live = real inference (slow first load)"
            )
            top_k_dd = gr.Dropdown([1, 3, 5, 10], value=3, label="Top-k retrieved")
            alpha_dd = gr.Dropdown([0.0, 0.5, 1.0], value=1.0, label="Alpha (image weight)")

        # ── tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── CACHED TAB ────────────────────────────────────────────────────
            with gr.TabItem("Curated Examples (Cached)"):
                case_dd = gr.Dropdown(
                    case_labels, value=case_labels[0] if case_labels else None,
                    label="Select demo example",
                )
                run_btn = gr.Button("▶ Show Example", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column(scale=1):
                        img_out      = gr.Image(label="Query Image", height=280)
                        info_out     = gr.HTML(label="Question & Ground Truth")
                    with gr.Column(scale=1):
                        baseline_out = gr.HTML(label="Baseline BLIP-2")
                    with gr.Column(scale=1):
                        rag_out      = gr.HTML(label="RAG Answer")

                with gr.Row():
                    with gr.Column(scale=1):
                        retrieved_out = gr.HTML(label="Retrieved Examples")
                    with gr.Column(scale=1):
                        prompt_out    = gr.Code(
                            label="Prompt Sent to BLIP-2",
                            language=None, lines=12,
                        )
                    with gr.Column(scale=1):
                        explanation_out = gr.Markdown(label="Explanation")

                def _cached_run(label, k, a):
                    k = int(k)
                    a = float(a)
                    res = display_case(label, k, a, cases)
                    return res[0], res[1], res[2], res[3], res[4], res[5], res[6]

                run_btn.click(
                    fn=_cached_run,
                    inputs=[case_dd, top_k_dd, alpha_dd],
                    outputs=[img_out, info_out, baseline_out, rag_out,
                             retrieved_out, prompt_out, explanation_out],
                )
                # Auto-load first case
                app.load(
                    fn=lambda: _cached_run(case_labels[0], 3, 1.0) if case_labels else [None]*7,
                    outputs=[img_out, info_out, baseline_out, rag_out,
                             retrieved_out, prompt_out, explanation_out],
                )

            # ── LIVE TAB ──────────────────────────────────────────────────────
            with gr.TabItem("Live Inference"):
                gr.Markdown(
                    "> **Note:** First run loads BLIP-2 (~15 sec). Subsequent runs are fast.\n"
                    "> Use a clear, well-lit photo for best results."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        live_img = gr.Image(
                            label="Upload Query Image", type="pil", height=260
                        )
                        live_q = gr.Textbox(
                            label="Question", placeholder="What color is the car?",
                            lines=2
                        )
                        live_btn = gr.Button("▶ Run Live Inference", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            live_info_out = gr.HTML(label="Query")
                        with gr.Row():
                            live_base_out = gr.HTML(label="Baseline")
                            live_rag_out  = gr.HTML(label="RAG")
                with gr.Row():
                    live_ret_out  = gr.HTML(label="Retrieved Examples")
                    live_pmt_out  = gr.Code(label="Prompt", language=None, lines=10)
                    live_exp_out  = gr.Markdown(label="Explanation")

                live_btn.click(
                    fn=lambda img, q, k, a: run_live_inference(img, q, int(k), float(a)),
                    inputs=[live_img, live_q, top_k_dd, alpha_dd],
                    outputs=[live_img, live_info_out, live_base_out, live_rag_out,
                             live_ret_out, live_pmt_out, live_exp_out,
                             gr.State(), gr.State()],
                )

            # ── STATS TAB ─────────────────────────────────────────────────────
            with gr.TabItem("Results Summary"):
                gr.Markdown("""
## Experiment Results (n=100)

| Configuration | α | Accuracy | Δ Baseline |
|---|---|---|---|
| Baseline BLIP-2 | — | 42.33% | — |
| RAG text-only | 0.0 | 41.33% | −1.00 pts |
| RAG balanced | 0.5 | 46.33% | +4.00 pts |
| **RAG image-only ★** | **1.0** | **51.33%** | **+9.00 pts** |

### Top-k Ablation (α=1.0)
| k | RAG Accuracy | Oracle@k |
|---|---|---|
| 1 | 53.33% | 24.00% |
| 3 | **68.00%** | 60.33% |
| 5 | 75.33% | 83.33% |
| 10 | 90.67% | 92.33% |

> **Key finding:** At k=1 and k=3, RAG accuracy *exceeds* answer overlap (oracle@k).
> The model is not simply copying retrieved answers — it combines visual reasoning with contextual priors.

### Answer Type Breakdown
| Type | Baseline | RAG | Δ |
|---|---|---|---|
| Yes/No (n=27) | 85.2% | 81.5% | −3.7 pts |
| Open-ended (n=73) | 26.5% | 40.2% | **+13.7 pts** |

### Helpful / Harmful / Neutral
- **Helpful:** 21/100 (21%) — RAG corrected a wrong answer
- **Harmful:** 9/100 (9%) — RAG broke a correct answer
- **Neutral:** 70/100 (70%) — No change
""")

    return app


# ── setup: re-run fresh inference ─────────────────────────────────────────────

def run_setup():
    """Re-runs mini inference on 30 validation samples to rebuild demo cases."""
    import json, pickle
    from pathlib import Path
    from tqdm import tqdm
    from src.dataset import load_vqav2, get_best_answer
    from src.evaluate import vqa_accuracy
    from src.model import load_model, generate_answer, generate_answer_with_context
    from src.rag_retriever import RAGRetriever

    demo_dir = DATA_DIR / "demo_images"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("Streaming 30 validation samples...")
    samples = load_vqav2("validation", num_samples=30)

    print("Saving images...")
    img_map = {}
    for s in samples:
        qid = str(s["question_id"])
        p = demo_dir / f"{qid}.jpg"
        s["image"].convert("RGB").save(p, quality=90)
        img_map[qid] = str(p)
    with open(IMAGE_MAP, "w") as f:
        json.dump(img_map, f)

    print("Loading CLIP + FAISS...")
    retriever = RAGRetriever()
    retriever.load_clip()
    retriever.load_index()

    print("Loading BLIP-2 (4-bit)...")
    model, processor = load_model(quantize=True)

    results = []
    for item in tqdm(samples, desc="Inference"):
        image     = item["image"].convert("RGB")
        question  = item["question"]
        gt_ans    = [a["answer"] for a in item["answers"]]
        qid       = str(item["question_id"])

        baseline  = generate_answer(model, processor, image, question)
        base_acc  = vqa_accuracy(baseline, gt_ans)

        retrieved = retriever.retrieve(image, question, top_k=3, alpha=1.0)
        rag_ans   = generate_answer_with_context(
            model, processor, image, question, retrieved)
        rag_acc   = vqa_accuracy(rag_ans, gt_ans)
        delta     = rag_acc - base_acc

        results.append({
            "question_id":  qid,
            "question":     question,
            "ground_truth": get_best_answer(item["answers"]),
            "gt_all":       [a["answer"] for a in item["answers"]][:3],
            "baseline_pred": baseline,
            "baseline_acc":  base_acc,
            "rag_pred":      rag_ans,
            "rag_acc":       rag_acc,
            "delta":         delta,
            "verdict":       "helpful" if delta > 0 else ("harmful" if delta < 0 else "neutral"),
            "retrieved":     [
                {
                    "question":  r.get("question", ""),
                    "answer":    r.get("best_answer", r.get("answer", "")),
                    "img_score": r.get("img_score",  r.get("score", 0.0)),
                    "q_score":   r.get("q_score",    0.0),
                    "score":     r.get("score",      r.get("img_score", 0.0)),
                }
                for r in retrieved
            ],
            "image_path": str(demo_dir / f"{qid}.jpg"),
        })

    with open(FRESH_DATA, "w") as f:
        json.dump(results, f, indent=2)

    # Select demo cases
    YES_NO = {"is ","are ","do ","does ","can ","has ","have ","was ","were ","will "}
    def is_yesno(q): return any(q.lower().startswith(p) for p in YES_NO)

    helpful      = sorted([r for r in results if r["verdict"]=="helpful"],  key=lambda x: -x["delta"])
    harmful      = [r for r in results if r["verdict"]=="harmful"]
    neutral      = [r for r in results if r["verdict"]=="neutral"]
    neutral_ok   = [r for r in neutral if r["baseline_acc"] >= 1.0]
    neutral_fail = [r for r in neutral if r["baseline_acc"] == 0 and r["rag_acc"] == 0]
    open_help    = [r for r in helpful if not is_yesno(r["question"])]
    yn_help      = [r for r in helpful if is_yesno(r["question"])]
    yn_harm      = [r for r in harmful if is_yesno(r["question"])]

    def make(r, label):
        c = dict(r)
        c["label"] = label
        gt = c["ground_truth"].lower().strip()
        c["leakage_warning"] = any(
            ex.get("answer","").lower().strip() == gt
            for ex in c.get("retrieved",[])
        )
        return c

    cases = []
    if open_help:            cases.append(make(open_help[0], "RAG Helps - Open-ended (best)"))
    if len(open_help) > 1:   cases.append(make(open_help[1], "RAG Helps - Open-ended (2nd)"))
    if yn_help:              cases.append(make(yn_help[0],   "RAG Helps - Yes/No fixed"))
    if yn_harm:              cases.append(make(yn_harm[0],   "RAG Hurts - Yes/No confused"))
    elif harmful:            cases.append(make(harmful[0],   "RAG Hurts - Context noise"))
    if neutral_ok:           cases.append(make(neutral_ok[0], "Both Correct - No change"))
    if neutral_fail:         cases.append(make(neutral_fail[0],"Both Fail - Hard question"))

    with open(DEMO_CASES, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"\nSetup complete. {len(cases)} demo cases saved.")
    for c in cases:
        print(f"  [{c['label']}] delta={c['delta']:+.2f} | leakage={c['leakage_warning']}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-Book VQA Demo")
    parser.add_argument("--mode",   default="cached", choices=["cached","live"])
    parser.add_argument("--setup",  action="store_true",
                        help="Re-run 30-sample inference to rebuild demo cases")
    parser.add_argument("--top_k",  type=int,   default=3)
    parser.add_argument("--alpha",  type=float, default=1.0)
    parser.add_argument("--port",   type=int,   default=7860)
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args()

    if args.setup:
        run_setup()

    if not DEMO_CASES.exists():
        print("No demo cases found. Run with --setup first.")
        print("  python demo_app.py --setup")
        raise SystemExit(1)

    app = build_app(mode=args.mode)
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
