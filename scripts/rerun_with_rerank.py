"""Re-run all 7 main demo cases with question-aware re-ranking and hint
filtering enabled. Cleans up the reasoning-trace presentation without changing
the eval-validated headline config.

For each main case:
  1. Call API with rerank=true, filter_hints=true (no caption)
  2. Call API with rerank=true, filter_hints=true, use_caption=true
  3. Update baseline / rag_20k / caption / retrieved / evidence_trace / modes
  4. Preserve prose (why_it_matters, demo_script, presenter_notes)
  5. Print before/after diff so any answer changes are visible
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8000"

MAIN_IDS = {
    "B_visual_memory", "A_open_ended", "C_kb_scaling",
    "L1_cat_color", "L8_cat_sleeping", "L9_cyclist_weather", "L10_pizza_eating",
}

# Validated headline config + rerank/filter on for cleaner reasoning trace.
# top_k=3, alpha=1.0, tau=0.0 (no gate so we capture raw RAG, then layer router elsewhere)
NO_CAP = {
    "top_k": "3", "alpha": "1.0", "tau": "0.0",
    "use_caption": "false", "caption_weight": "0.0",
    "rerank": "true",                    # ← question-aware re-ranking
    "use_answer_prior": "false",
    "filter_hints": "true",              # ← drop low-relevance hints
    "filter_threshold": "0.15",
    "type_gate": "false",
    "prompt_template": "current_prompt",
}
WITH_CAP = dict(NO_CAP, use_caption="true")


def call(img_path: Path, question: str, form: dict) -> dict:
    with open(img_path, "rb") as f:
        files = {"image": (img_path.name, f.read(), "image/png")}
    data = dict(form, question=question)
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def vqa_correct(answer: str, gt_all: list[str]) -> float:
    if not answer or not gt_all:
        return 0.0
    a = answer.strip().lower()
    for g in gt_all:
        gl = (g or "").strip().lower()
        if not gl:
            continue
        if gl == a or gl in a or a in gl:
            return 1.0
    return 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = ROOT / "results" / "demo_cases.json"
    cases = json.loads(p.read_text(encoding="utf-8"))

    print(f"{'id':<22} {'OLD baseline':<32} {'OLD rag':<26} ->  {'NEW baseline':<32} {'NEW rag':<26}")
    print("-" * 160)

    updated = 0
    for c in cases:
        if c["id"] not in MAIN_IDS:
            continue
        img = ROOT / c["image_path"].replace("\\", "/")
        if not img.exists():
            print(f"SKIP {c['id']} — missing {img}")
            continue

        old_base = c.get("baseline_answer", "")
        old_rag  = c.get("rag_20k_answer", "")

        try:
            r1 = call(img, c["question"], NO_CAP)
            r2 = call(img, c["question"], WITH_CAP)
        except Exception as e:
            print(f"FAIL {c['id']}: {e}")
            continue

        baseline = r1["baseline"]
        rag_20k  = r1["rag"]
        cap_ans  = r2["rag"]
        caption  = r2.get("caption", c.get("caption", ""))
        retrieved = r1["retrieved"]
        candidates = [r["answer"] for r in retrieved]
        cnt = Counter(a for a in candidates if a)
        most_common = cnt.most_common(1)[0][0] if cnt else ""
        gt_all = c.get("gt_all", [c.get("ground_truth","")])
        base_acc = vqa_correct(baseline, gt_all)
        rag_acc  = vqa_correct(rag_20k,  gt_all)

        # Update only the data fields. Prose preserved.
        c["baseline_pred"]    = baseline
        c["baseline_acc"]     = base_acc
        c["rag_pred"]         = rag_20k
        c["rag_acc"]          = rag_acc
        c["delta"]            = rag_acc - base_acc
        c["caption"]          = caption
        c["candidate_answers"] = candidates
        c["most_common_answer"] = most_common
        c["retrieved"]        = retrieved
        c["baseline_answer"]  = baseline
        c["rag_20k_answer"]   = rag_20k
        c["caption_answer"]   = cap_ans
        c["raw_rag_answer"]   = rag_20k
        c["final_display_answer"] = rag_20k
        c["retrieved_examples"] = [
            {
                "rank": i + 1,
                "question":   r.get("question", ""),
                "answer":     r.get("answer", ""),
                "image_score": r.get("img_score", 0.0),
                "text_score":  r.get("q_score",  0.0),
                "final_score": r.get("score",    0.0),
                "image_path":  "",
                "_source":     "20k diverse · question-reranked",
            }
            for i, r in enumerate(retrieved)
        ]
        c["evidence_trace"] = [
            f"Visual summary: {caption}" if caption else "Visual summary: (no caption)",
            f"User question: {c['question']}",
            "Retrieved similar examples (question-reranked): " + " · ".join(
                f"[{r.get('question','')} → {r.get('answer','')}]" for r in retrieved[:3]
            ),
            "Candidate answer priors: " + ", ".join(candidates),
            f"Final answer: {rag_20k}",
        ]
        c["modes"] = [
            {"name": "Baseline BLIP-2 (frozen, no retrieval)",
             "answer": baseline, "correct": base_acc >= 1.0,
             "note": "Frozen BLIP-2 alone"},
            {"name": "20k diverse RAG (image + question rerank)",
             "answer": rag_20k, "correct": rag_acc >= 1.0,
             "note": f"Top retrieval score {retrieved[0].get('img_score', 0):.3f}" if retrieved else ""},
            {"name": "20k RAG + caption in prompt",
             "answer": cap_ans, "correct": vqa_correct(cap_ans, gt_all) >= 1.0,
             "note": f'Caption: "{caption}"'},
        ]
        updated += 1
        flag = ""
        if old_base != baseline: flag += " ⚠BASE-CHANGED"
        if old_rag  != rag_20k:  flag += " ⚠RAG-CHANGED"
        print(f"{c['id']:<22} {old_base!r:<32} {old_rag!r:<26} ->  {baseline!r:<32} {rag_20k!r:<26}{flag}")

    p.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"\nUpdated {updated} cases.")


if __name__ == "__main__":
    main()
