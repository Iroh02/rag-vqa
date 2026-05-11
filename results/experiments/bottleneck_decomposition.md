# Clean Oracle@k + Bottleneck Decomposition

100-sample eval (validation offset=5000) · 20k diverse KB (image-id-disjoint) · α=1.0.

## Clean oracle@k on the 20k KB

| k | clean 20k oracle@k | leaky 5k val oracle@k (legacy) |
|---:|---:|---:|
| 1 | **6.00%** | 24.00% |
| 3 | **24.00%** | 60.33% |
| 5 | **36.00%** | 83.33% |
| 10 | **46.00%** | 92.33% |

The legacy `oracle@10 = 92.33%` was image-id-leakage-driven on the 5k validation-built KB. The clean number on the image-id-disjoint 20k KB is much lower — that's the **real** ceiling of any selector that picks among the top-k retrieved answers.

## Bottleneck decomposition (rag_k3, n=100)

| Category | Count | % of all | % of wrong |
|---|---:|---:|---:|
| retrieval_then_lm | 18 | 18.0% | — |
| generator_smart | 37 | 37.0% | — |
| generator_failure | 6 | 6.0% | 13.3% |
| retrieval_failure | 39 | 39.0% | 86.7% |

**Definitions:**
- `retrieval_then_lm` — GT in retrieved AND model picked it (working as designed)
- `generator_smart`  — GT NOT in retrieved AND model still got it right (image-grounded baseline strength)
- `generator_failure` — GT IS in retrieved BUT model picked wrong (the **'evidence ignored'** failure mode)
- `retrieval_failure` — GT not in retrieved (the **retrieval ceiling** — needs more KB / better retriever)

**Of the 45 wrong cases:**
- 6 (13%) are generator-bound — fixing requires a fine-tuned generator or better evidence-aware selection.
- 39 (87%) are retrieval-bound — fixing requires bigger/better KB or a trainable retriever.

**Headline bottleneck: retrieval-bound.**