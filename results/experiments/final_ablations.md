# Final ablations — RAG-VQA

Eval set: 100 validation samples (offset=5000) · 20k diverse KB · BLIP-2 frozen, 4-bit · `current_prompt`.

| Configuration | Accuracy | Δ vs k=3 | Helped | Hurt |
|---|---:|---:|---:|---:|
| Baseline (frozen, no retrieval) | 42.33% | -17.00 | — | — |
| RAG · α=1.0 · k=1 | 53.67% | -5.67 | 6 | 13 |
| ★ RAG · α=1.0 · k=3 (headline reproduce) ★ | 59.33% | +0.00 | — | — |
| RAG · α=1.0 · k=5 | 59.33% | +0.00 | 0 | 0 |
| RAG · α=1.0 · k=10 | 59.33% | +0.00 | 0 | 0 |
| RAG · α=0.5 · k=3 | 46.67% | -12.67 | 8 | 20 |
| RAG · α=0.0 · k=3 | 43.67% | -15.67 | 7 | 23 |
| RAG · α=1.0 · k=3 + cross-encoder rerank | 57.33% | -2.00 | 5 | 8 |
| RAG + Evidence-Aware Selector | 56.00% | -3.33 | 0 | 3 |
| Type-conditional gate (yes/no → baseline) | 60.33% | +1.00 | 4 | 3 |
| Self-consistency vote (α ∈ {0.0, 0.5, 1.0}) | 49.67% | -9.67 | 7 | 16 |

## Type breakdown (yes/no vs open-ended)

| Configuration | Yes/No (n=27) | Open-ended (n=73) |
|---|---:|---:|
| Baseline | 82.86% | 20.51% |
| RAG headline (k=3, α=1.0) | 80.00% | 48.21% |
| Type-conditional gate | 82.86% | 48.21% |

**Winner: Type-conditional gate (yes/no → baseline) at 60.33%** (Δ +1.00 vs the k=3 headline).
