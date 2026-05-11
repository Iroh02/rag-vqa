# Open-Book VQA — RAG-BLIP-2

Improves a frozen BLIP-2 (3B) model at inference time by adding a CLIP + FAISS visual memory. No retraining. +9 percentage points absolute on VQAv2.

**42.33% → 51.33%** baseline → RAG (alpha=1.0, k=3)

---

## How It Works

1. Query image is encoded by CLIP ViT-B/32 (512-d embedding)
2. FAISS IndexFlatIP searches a 5,000-entry knowledge base of VQA training examples
3. Top-k most visually similar Q&A pairs are prepended as hints to the BLIP-2 prompt
4. BLIP-2 generates the answer with context

Score fusion: `score = α · img_similarity + (1-α) · text_similarity`

Best result: **alpha=1.0** (image-only retrieval)

---

## Setup

```bash
pip install -r requirements.txt
```

Download VQAv2 data to `data/vqav2/` or let the streaming loader fetch it automatically.

---

## Run Experiments

```bash
# baseline evaluation
python run.py --mode eval --baseline

# RAG evaluation (best config)
python run.py --mode eval --alpha 1.0 --top_k 3

# alpha sweep
python run.py --mode sweep

# top-k ablation
python run.py --mode topk_ablation

# oracle@k analysis
python run.py --mode oracle
```

Results saved to `results/eval_results.json` and `results/experiments/`.

---

## Interactive Demo

### Standalone HTML demo (recommended)

```bash
# terminal 1 — static file server for the HTML demo
python -m http.server 8081

# terminal 2 — inference API (needed for live inference tab)
python api.py
```

Open `http://localhost:8081/demo.html`. Three tabs:
- **Curated Examples** — 6 pre-computed cases, instant
- **Live Inference** — upload an image, ask a question, full RAG pipeline runs locally
- **Results Summary** — all experiment numbers

### Live tab options

- **Top-k** (1–5): how many retrieved examples to use
- **α** (0–1): retrieval weight (1.0 = image-only, the headline config)
- **τ** (0–1): confidence gate — skips RAG when top-1 image-similarity score < τ; falls back to baseline. Use τ=0.7 to suppress noisy retrieval on out-of-distribution queries.
- **Cross-encoder re-rank**: re-ranks the top-30 CLIP candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generate image caption**: prepends a BLIP-2 caption to the prompt for richer context

### Rebuilding the knowledge base

```bash
python -m src.rag_build_kb --num_samples 5000 --diverse
```

The `--diverse` flag streams VQAv2 validation past offset=5100 (image-id-disjoint from the 100-sample eval) and keeps one entry per unique image_id. Yields a 5.9× increase in visual coverage at the same KB size. Takes ~15 min on CPU.

---

## Presentation

```
presentation/rag_vqa_open_book.html   # 17-slide deck (open in browser)
presentation/rag_vqa_open_book.pdf    # PDF export
presentation/speaker_script.md        # 30s / 2min / 5min scripts + Q&A
presentation/demo_script.md           # demo walkthrough script
```

Navigate slides with arrow keys. Press **N** for speaker notes.

---

## Key Results

| Configuration | Accuracy |
|---------------|----------|
| Baseline BLIP-2 | 42.33% |
| Text-only retrieval (α=0.0) | 41.33% |
| Balanced (α=0.5) | 46.33% |
| Image-only retrieval (α=1.0) | **51.33%** |
| Image-only, k=3 | 68.00% |
| Image-only, k=5 | 75.33% |
| Image-only, k=10 | 90.67% |

| Question type | Baseline | RAG (α=1.0) | Delta |
|---------------|----------|-------------|-------|
| Yes/No | 85.2% | 81.5% | −3.7 |
| Open-ended | 26.5% | 40.2% | +13.7 |

Helpful / Harmful / Neutral: 21 / 9 / 70 (2.3:1 ratio)

---

## Project Structure

```
src/
  model.py          # BLIP-2 + RAG pipeline
  finetune.py       # QLoRA fine-tuning (experimental)
  analyze_offline.py
  experiments.py
run.py              # CLI entry point
demo_app.py         # Gradio demo app
results/
  eval_results.json
  predictions.json
  demo_cases.json
  experiments/      # CSV ablation results
  plots/            # PNG figures
data/
  demo_images/      # 30 demo images
```
