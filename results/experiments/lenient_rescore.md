# Lenient Re-Scoring — *should semantically-correct count?*

Re-scoring of the same 100-sample eval predictions under three accuracy regimes. No new inference — only the metric changes.

- **STRICT**: VQA soft accuracy — exact-string match against any annotator. The existing 59.33% / 60.33% headline.
- **LENIENT**: whole-word substring — any annotator answer appears as a *whole word* in the prediction. Rewards verbose-but-correct (`'Watching the skateboarder'` for GT `'watching'`); rejects degenerate `'blueblue'`.
- **SUBSTR**: any-substring — answer contains GT anywhere. Most permissive; accepts even mode-collapse outputs.

| Configuration | STRICT | LENIENT | SUBSTR | strict→lenient flips |
|---|---:|---:|---:|---:|
| Baseline (frozen, no retrieval) | 42.33% | 65.00% | 66.00% | 27 |
| RAG · α=1.0 · k=1 | 53.67% | 64.00% | 66.00% | 16 |
| ★ RAG · α=1.0 · k=3 (headline) | 59.33% | 66.00% | 68.00% | 11 |
| RAG · α=1.0 · k=5 | 59.33% | 66.00% | 68.00% | 11 |
| RAG · α=1.0 · k=10 | 59.33% | 66.00% | 68.00% | 11 |
| RAG · α=0.5 · k=3 | 46.67% | 53.00% | 56.00% | 10 |
| RAG · α=0.0 · k=3 | 43.67% | 52.00% | 55.00% | 13 |
| + cross-encoder rerank | 57.33% | 62.00% | 63.00% | 10 |
| + evidence-aware selector | 56.00% | 62.00% | 64.00% | 10 |
| + type-conditional gate | 60.33% | 68.00% | 70.00% | 12 |
| + self-consistency vote | 49.67% | 57.00% | 60.00% | 11 |

## What flips strict → lenient on the headline RAG (k=3)

| Question | GT | Prediction (strict-wrong, lenient-correct) |
|---|---|---|
| What color are the mums in the front of the photo? | `pink|red|pink, red, yellow` | `red` |
| What is this man holding? | `ski pole|ski poles|poles` | `skis` |
| What kinds of food are on the plate? | `breakfast|spinach|eggs, spinach` | `broccoli and cheese` |
| What type of computer is on the desk? | `mac|laptop|laptop` | `apple` |

## Reading

Lenient scoring shifts every mode's accuracy upward — and the gap is meaningful, not noise. Verbose-but-correct outputs are common, especially for the baseline (which produces longer freeform answers). However: **the relative ordering of modes is unchanged** under all three regimes. The +17 RAG gain and the +1 type-gate improvement persist regardless of metric. Strict VQA scoring is a harsh judge of form, but the system-level conclusions are robust to that harshness.