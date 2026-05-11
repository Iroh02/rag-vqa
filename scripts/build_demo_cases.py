"""Build results/demo_cases.json with the 7-category schema.

All numbers come from existing precomputed data — no new inference.
Source: data/precomputed_demo_data.json (clean 20k retrievals)
       + data/fresh_demo_results.json (legacy 5k retrievals — used for category E
         to demonstrate the 'generator ignores evidence' failure mode)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE  = json.loads((ROOT / "data" / "precomputed_demo_data.json").read_text())
FRESH = json.loads((ROOT / "data" / "fresh_demo_results.json").read_text())
PRE_BY_QID  = {d["question_id"]: d for d in PRE}
FRESH_BY_QID = {str(d["question_id"]): d for d in FRESH}


def fmt_retrieved(retrieved, src_label):
    """Format retrieved examples to the documented schema."""
    out = []
    for i, r in enumerate(retrieved[:3]):
        out.append({
            "rank":         i + 1,
            "question":     r.get("question", ""),
            "answer":       r.get("answer") or r.get("best_answer") or "",
            "image_score":  round(r.get("img_score", 0.0), 4),
            "text_score":   round(r.get("q_score",   0.0), 4),
            "final_score":  round(r.get("score",     r.get("img_score", 0.0)), 4),
            "image_path":   "",   # we don't store retrieved-image paths in the KB
            "_source":      src_label,
        })
    return out


def evidence_trace(question, caption, retrieved, candidate_priors, final_answer):
    return [
        f"Visual summary: {caption}" if caption else "Visual summary: (no caption generated)",
        f"User question: {question}",
        "Retrieved similar examples: " + " · ".join(
            f"[{r['question']} → {r['answer']}]" for r in retrieved[:3]
        ),
        "Candidate answer priors: " + ", ".join(candidate_priors) if candidate_priors else "Candidate answer priors: (none)",
        f"Final answer: {final_answer}",
    ]


cases = []

# ── A. RAG Success — Open-ended ──────────────────────────────────────────────
qid = "262148001"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "A_open_ended",
    "category":        "RAG Success — Open-ended",
    "title":           "Open-ended format flip — *people watching the skateboarder*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline answers in a long-form sentence — <b>semantically right</b> ('watching the "
                       "skateboarder' clearly contains 'watching') but the VQA metric requires short canonical "
                       "answers, so strict matching gives it zero. The 20k KB retrieves visually-similar "
                       "skateboarding scenes whose Q&A pairs nudge BLIP-2 toward the canonical short answer "
                       "'watching'. The win here is <b>format compression, not semantic correction</b> — both "
                       "5k and 20k get this right. We mark the baseline as VERBOSE rather than WRONG to be "
                       "honest about what the gain is."),
    "demo_script": ("Honest framing first. The baseline says \"Watching the skateboarder\" — this is "
                    "<b>semantically correct</b>. It contains 'watching'. The VQA metric, though, requires "
                    "short canonical short-form answers, so strict scoring gives it zero. RAG over the 20k "
                    "diverse KB compresses to the canonical 'watching' — full credit. So the win on this case "
                    "is <em>format compression</em>, not semantic rescue. Open-ended questions like this are "
                    "exactly where the format-compression effect is largest — the eval shows +13.7 points on "
                    "open-ended."),
    "presenter_notes": {
        "what_to_say":   "Open with this — be upfront that 'Watching the skateboarder' is semantically correct. The win is format compression, not semantic rescue.",
        "what_it_proves": "RAG compresses BLIP-2's verbose answers to the canonical VQA short form. This is a real win on the metric, but frame it honestly.",
        "caveat":         "Baseline isn't conceptually wrong — it's verbose. The UI marks it VERBOSE ✓ rather than WRONG to reflect this. Same image_id is NOT in the 20k KB.",
    },
    "warning": None,
})

# ── B. RAG Success — Visual Memory ───────────────────────────────────────────
qid = "262162007"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "B_visual_memory",
    "category":        "RAG Success — Visual Memory",
    "title":           "Retrieved Q&A visibly supports the answer — *are the lights on?*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline confidently says 'yes'. The 20k KB retrieves two on/off questions "
                       "from semantically related scenes — *Is the TV on? → no* and *Is the lamp "
                       "turned on? → no*. The retrieved answers visibly support the correct answer, "
                       "and BLIP-2 flips its confident wrong yes into a correct no."),
    "demo_script": ("Look at the retrieved examples. *Is the TV on? — no.* *Is the lamp turned on? "
                    "— no.* Two on/off questions from similar scenes both answered no. The model "
                    "sees this pattern in the prompt and correctly answers 'no' — instead of the "
                    "baseline's confident 'yes'. This is what 'open-book' literally means."),
    "presenter_notes": {
        "what_to_say":   "Point at the retrieved Q&A list. Trace the priors visibly.",
        "what_it_proves": "RAG doesn't need the EXACT same question — it needs semantically-aligned hints.",
        "caveat":         "Retrieved priors here are 'no, 6, no' — only 'no' appears twice, so it's a real majority signal.",
    },
    "warning": None,
})

# ── C. KB Scaling Win ────────────────────────────────────────────────────────
qid = "262162000"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "C_kb_scaling",
    "category":        "KB Scaling Win",
    "title":           "5k KB couldn't fix this; 20k KB can — *is that a folding chair?*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline says 'yes' (wrong). The 5k diverse RAG also says 'yes' — its KB "
                       "has too few visually-similar scenes to flip the prior. The 20k diverse KB "
                       "retrieves furniture/lighting questions whose answers cluster on 'no', "
                       "and the model correctly answers 'no'. KB scale matters when the answer "
                       "depends on tail-distribution scenes."),
    "demo_script": ("Same question, three modes side-by-side. Baseline: yes. 5k RAG: yes — the "
                    "KB couldn't surface enough scene context. 20k RAG: no — correct. The +5 "
                    "point aggregate gain from 5k → 20k is concretely visible here."),
    "presenter_notes": {
        "what_to_say":   "Use this to motivate KB scaling. The 5k is wrong; the 20k is right; same model.",
        "what_it_proves": "KB scale and image-diversity drive most of the gain in this system — not model changes.",
        "caveat":         "5k diverse KB used here was already image-id-disjoint. The gain is from MORE diversity, not just disjointness.",
    },
    "warning": None,
})

# ── D. Caption Interpretability ──────────────────────────────────────────────
qid = "240301002"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "D_caption_interp",
    "category":        "Caption Interpretability",
    "title":           "Caption makes the pipeline legible — *why is the cow laying down?*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline gives a strange hallucination: 'Because the cow is dead'. The 5k "
                       "KB had image-id leakage on this case, so it correctly retrieved the literal "
                       "answer 'tired' — that win was real but came partly from leakage. The 20k "
                       "KB is image-disjoint and answers 'to sleep' — close but not the canonical "
                       "VQA short form. **The caption ('a group of cows in a barn')** plays no "
                       "accuracy role here, but it makes the reasoning trace legible: viewers can "
                       "see exactly what BLIP-2 'sees' before answering."),
    "demo_script": ("This is the caption interpretability case. The caption — 'a group of cows "
                    "in a barn' — doesn't change the answer. But it makes the pipeline legible. "
                    "Without the caption, the reasoning trace just shows retrieved Q&A pairs. With "
                    "it, you can read what the model sees in plain English. **Captions help "
                    "interpretability, not accuracy** — that's an honest finding from the ablation: "
                    "57.00% with caption-in-prompt vs 59.33% without. We keep captions in the demo "
                    "for transparency, not for the leaderboard."),
    "presenter_notes": {
        "what_to_say":   "Stress: captions don't improve the number. They make the system legible.",
        "what_it_proves": "Even an honest negative result (captions hurt accuracy) earns its keep when it adds interpretability.",
        "caveat":         "The 5k 'tired' answer in the legacy table was helped by image-id leakage — flag this when demoing.",
    },
    "warning": "5k diverse RAG result for this case used the legacy validation-built KB which had image_id overlap with eval. The 20k diverse KB is fully image-id-disjoint.",
})

# ── E. Weakness — Generator Ignores Evidence (best illustrated with 5k retrieval) ──
# Use the legacy 5k retrieval for this case — it had image-id leakage which put the
# exact GT into the retrieved hints, and BLIP-2 *still* answered wrong. That's the
# textbook 'evidence ignored' pattern. We disclose the leakage as a CAVEAT.
qid = "262148000"
d = PRE_BY_QID[qid]
fresh = FRESH_BY_QID[qid]
legacy_retrieved = fresh.get("retrieved", [])
cases.append({
    "id":              "E_evidence_ignored",
    "category":        "Weakness — Generator Ignores Evidence",
    "title":           "Retrieval contained the exact answer — generator ignored it",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_5k"],   # raw RAG with 5k retrieval — wrong despite GT in hints
    "final_display_answer": d["rag_5k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(legacy_retrieved, "5k validation-built (legacy)"),
    "candidate_answers":  [r.get("answer", "") for r in legacy_retrieved[:3]],
    "evidence_trace": evidence_trace(
        d["question"], d["caption"], legacy_retrieved,
        [r.get("answer", "") for r in legacy_retrieved[:3]], d["rag_5k"],
    ),
    "why_it_matters": ("The legacy 5k validation-built KB had image-id overlap with the eval. For "
                       "this case, the exact same question was retrieved with the answer 'down' — "
                       "literally in the prompt — yet BLIP-2 generated 'up'. Even when the answer "
                       "is in the prompt, the generator does not always attend to it. This is the "
                       "**generator-reliability** problem: retrieval did its job; generation did "
                       "not. It motivates trainable retrievers/re-rankers and a fine-tuned generator "
                       "as future work."),
    "demo_script": ("Look at the retrieved hints. The model literally has *Where is he looking? — "
                    "down* in its prompt. Its answer? 'up'. The wrong direction. This is not a "
                    "retrieval failure — retrieval did exactly its job. This is the generator "
                    "ignoring the evidence. It's the honest reason inference-time RAG has a ceiling: "
                    "the frozen language model doesn't perfectly attend to retrieved context."),
    "presenter_notes": {
        "what_to_say":   "Frame this as generator weakness, not retrieval failure. Use the word \"evidence trace\".",
        "what_it_proves": "Retrieval is not the bottleneck on this case — context use is.",
        "caveat":         "The 5k retrieval ALSO contained image-id leakage — full disclosure. The 20k clean KB doesn't have 'down' in retrieval, and the model still gets it wrong (but for a different reason: retrieval is weaker).",
    },
    "warning": "Legacy 5k retrieval shown — image_id present in KB. This is intentional: the leakage put the exact GT into the prompt, which is precisely what makes this an 'evidence ignored' demonstration.",
})

# ── F. Weakness — Retrieval Noise (yes/no flip) ──────────────────────────────
qid = "240301001"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "F_retrieval_noise",
    "category":        "Weakness — Retrieval Noise",
    "title":           "Retrieval flips a confident correct yes/no — *is it daylight?*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline correctly says 'yes'. 5k RAG also says 'yes'. But the 20k KB "
                       "retrieves cow/farm questions whose answers cluster around 'no' (*Are the "
                       "cows grazing? — no*). The retrieved noise causes BLIP-2 to flip its correct "
                       "yes into a wrong no. This is the harmful-RAG failure mode — present in 9 "
                       "of 100 eval cases and skewed toward yes/no questions, where the model is "
                       "already strong and any extra context can mislead."),
    "demo_script": ("This is the honest weakness. Baseline correctly says yes — it's clearly "
                    "daytime in the image. RAG retrieves cow-related questions answered 'no', "
                    "and BLIP-2 picks up that 'no' bias and flips a correct answer into a wrong "
                    "one. **9 of 100 cases in the eval look like this**, and they are mostly yes/no "
                    "questions. The mitigation is the τ-gate, which we'll demo at the end."),
    "presenter_notes": {
        "what_to_say":   "Show this AFTER the wins, never before. Frame as 'controlled 100-sample evaluation found 9 such cases'.",
        "what_it_proves": "RAG hurts when the model is already confident and retrieval adds noise.",
        "caveat":         "20k diverse KB is image-disjoint — the 'no' bias here is genuine retrieval-induced noise, not leakage.",
    },
    "warning": None,
})

# ── G. Neutral — Both Correct ────────────────────────────────────────────────
qid = "393226002"
d = PRE_BY_QID[qid]
cases.append({
    "id":              "G_both_correct",
    "category":        "Neutral / Both Correct",
    "title":           "RAG adds no harm when baseline is already right — *what does the truck sell?*",
    "question":        d["question"],
    "ground_truth":    d["ground_truth"],
    "gt_all":          d["gt_all"],
    "image_path":      d["image_path"],
    "baseline_answer": d["baseline"],
    "rag_5k_answer":   d["rag_5k"],
    "rag_20k_answer":  d["rag_20k"],
    "caption_answer":  d["rag_20k_caption"],
    "raw_rag_answer":  d["rag_20k"],
    "final_display_answer": d["rag_20k"],
    "caption":         d["caption"],
    "retrieved_examples": fmt_retrieved(d["retrieved_20k"], "20k diverse"),
    "candidate_answers":  d["candidate_answers"],
    "evidence_trace": evidence_trace(d["question"], d["caption"], d["retrieved_20k"],
                                      d["candidate_answers"], d["rag_20k"]),
    "why_it_matters": ("Baseline already gets this right ('ice cream'). All RAG variants also "
                       "answer 'ice cream'. **70% of the 100-sample eval looks like this** — RAG "
                       "is a no-op when the model is already correct. Useful as a calm closer "
                       "showing the system doesn't degrade easy cases."),
    "demo_script": ("Brief calm closer. Easy question. Baseline already correct. All RAG variants "
                    "also correct. 70% of the eval looks like this. The system doesn't degrade "
                    "easy cases."),
    "presenter_notes": {
        "what_to_say":   "Use only if you have time after the weakness cases. Calm closer.",
        "what_it_proves": "RAG is non-destructive on easy cases.",
        "caveat":         "Caption mentions 'ice cream truck' — this case has redundant retrieval and caption signals.",
    },
    "warning": None,
})

# ── audit pass: flag any leakage warnings ────────────────────────────────────
for c in cases:
    audit = []
    for r in c["retrieved_examples"]:
        if r["question"].strip().lower() == c["question"].strip().lower():
            audit.append(f"Same question text appears in retrieved (rank {r['rank']})")
        if r["image_score"] >= 0.95:
            audit.append(f"Retrieval image-score {r['image_score']} ≥ 0.95 — possible same-image overlap (rank {r['rank']})")
    if audit:
        existing = c.get("warning") or ""
        merged = (existing + " | " if existing else "") + " | ".join(audit)
        c["warning"] = merged

(ROOT / "results" / "demo_cases.json").write_text(json.dumps(cases, indent=2))
print(f"Wrote {len(cases)} curated cases:")
for c in cases:
    flag = " ⚠" if c.get("warning") else ""
    print(f"  [{c['category']:42s}] {c['title']}{flag}")
