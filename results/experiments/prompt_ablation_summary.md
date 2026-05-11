# Prompt Ablation — RAG-VQA

Eval set: 100 validation samples (offset=5000) · top_k=3 · α=1.0 · 20k diverse KB · BLIP-2 frozen, 4-bit.

**Baseline (no retrieval): 42.33%**

| Template | Accuracy | Δ vs baseline | Δ vs current | Helped | Hurt | Neutral | avg chars | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_prompt ★ | 59.33% | +17.00 | +0.00 | — | — | — | 167 | 41 |
| minimal_fewshot | 53.67% | +11.33 | -5.67 | 4 | 10 | 86 | 167 | 41 |
| evidence_noisy_warning | 24.00% | -18.33 | -35.33 | 3 | 35 | 62 | 461 | 115 |
| image_first | 9.00% | -33.33 | -50.33 | 1 | 47 | 52 | 334 | 83 |
| direct_evidence_override | 8.33% | -34.00 | -51.00 | 2 | 50 | 48 | 367 | 91 |
| short_evidence | 7.33% | -35.00 | -52.00 | 2 | 51 | 47 | 357 | 89 |
| caption_aware | 5.00% | -37.33 | -54.33 | 1 | 52 | 47 | 384 | 96 |
| candidate_constrained | 3.33% | -39.00 | -56.00 | 2 | 54 | 44 | 363 | 90 |
| question_match_priority | 0.00% | -42.33 | -59.33 | 0 | 55 | 45 | 474 | 118 |

**Winner: `current_prompt`** (highest accuracy; ties broken by shorter prompt).

---

## Headline finding — prompt engineering alone cannot improve accuracy

The simple Q/A few-shot format already in the codebase is the best of the 9 templates tested. **All 8 alternatives reduce accuracy**, most catastrophically (−35 to −59 points). The two formats that share the same simple structure (`current_prompt` and `minimal_fewshot`, both ~167 chars) score 59.33% and 53.67% respectively — the only two templates that exceed the 42.33% baseline.

Why every verbose template fails: BLIP-2's frozen language head is OPT-2.7B, which is **not instruction-tuned**. When the prompt looks like a chat instruction (*"Use the examples only if they help. Give a short answer."*), OPT continues whatever pattern best matches its training distribution — usually freeform completion of the instruction itself, not a short VQA answer. The simple Q/A few-shot format works precisely because OPT was pretrained on text where `Q: ... A: ...` patterns are everywhere.

The harmful/neutral counts confirm this: the worst templates didn't lose points evenly across cases — they broke 50+ of 100 cases that the existing prompt got right.

## "Where is he looking?" — generator-ignores-evidence case

This case (qid 262148000) is from validation offset 0, not the 5000-offset eval set, so it isn't in the 100-sample CSV. We ran a separate targeted test with all 9 templates against the **clean 20k KB**:

| Template | Prediction | Score |
|---|---|---:|
| current_prompt | `at the crowd` | 0.00 |
| short_evidence | `yes` | 0.00 |
| candidate_constrained | `yes, blue, male` | 0.00 |
| question_match_priority | `` (empty) | 0.00 |
| minimal_fewshot | `at the crowd` | 0.00 |
| image_first | `looking at the crowd` | 0.00 |
| evidence_noisy_warning | `yes, blue` | 0.00 |
| caption_aware | `` (empty) | 0.00 |
| direct_evidence_override | `` (empty) | 0.00 |

**Ground truth: `down`. No template gets it right.** The retrieved hints from the clean 20k KB are *Is anyone watching? → yes*, *What colour is the graffiti? → blue*, *Is it male or female? → male* — none address gaze direction. No amount of prompting can produce an answer that isn't present in either the image (BLIP-2 isn't grounding fine direction) or the retrieval. The earlier demo's success on this case (in the 5k validation-built KB with image-id leakage) was driven by *retrieving the exact same question's answer*, not by prompting.

## Conclusion

> Prompt engineering alone is not enough. The generator inconsistently uses retrieved evidence, and no rewording of the prompt fixes that. This motivates evidence-aware selection (a code-level rule, already implemented as a demo toggle) and fine-tuning of the generator (future work).

The 59.33% headline number stands. `current_prompt` remains the default in `api.py`. The dropdown in `demo.html` lets you select any of the 9 templates at inference time so the audience can verify the ablation result interactively.

---

## How to reproduce

```
# stop the API first so the GPU is free
python -m scripts.prompt_ablation --num_samples 100 --top_k 3 --alpha 1.0
```

Outputs:
- `results/experiments/prompt_ablation.csv` — one row per template
- `results/experiments/prompt_ablation_detail.csv` — one row per sample × template
- `results/experiments/prompt_ablation_summary.md` — this file
- `results/plots/prompt_ablation.png` — bar chart
- `results/experiments/where_is_he_looking_per_template.json` — targeted single-case test
