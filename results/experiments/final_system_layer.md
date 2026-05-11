# Final system layer — RAG-VQA improvements

100-sample eval (offset=5000) · 20k diverse KB · BLIP-2 frozen, 4-bit · `current_prompt`.

Same eval set, no new training, no KB rebuild. Strict VQA accuracy.

| Configuration | Overall | Yes/No | Open | Δ vs RAG |
|---|---:|---:|---:|---:|
| Baseline (no retrieval) | 42.33% | 82.86% | 20.51% | -17.00 |
| ★ RAG α=1.0 k=3 (headline) ★ | 59.33% | 80.00% | 48.21% | +0.00 |
| D. Question-aware rerank (α=0.75, c50) | 50.33% | 80.00% | 34.36% | -9.00 |
| A. Type-router on RAG | 60.33% | 82.86% | 48.21% | +1.00 |
| A. Type-router on QAR | 51.33% | 82.86% | 34.36% | -8.00 |
| B. Evidence-aware selector on RAG | 56.00% | 80.00% | 43.08% | -3.33 |
| B. Evidence-aware selector on QAR | 44.33% | 71.43% | 29.74% | -15.00 |
| C. Normalizer on baseline | 44.00% | 82.86% | 23.08% | -15.33 |
| C. Normalizer on RAG | 59.33% | 80.00% | 48.21% | +0.00 |
| C. Normalizer on QAR | 50.33% | 80.00% | 34.36% | -9.00 |
| ★★ A→B→C on QAR (full stack) ★★ | 48.33% | 82.86% | 29.74% | -11.00 |