"""Re-run all 7 main demo cases with the headline config (rerank=false,
filter_hints=false, type_gate=false). Restores the pre-rerank state.
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

# Headline (validated) config — matches the 60.33% benchmark
HEADLINE_NO_CAP = {
    "top_k": "3", "alpha": "1.0", "tau": "0.0",
    "use_caption": "false", "caption_weight": "0.0",
    "rerank": "false", "use_answer_prior": "false",
    "filter_hints": "false", "type_gate": "false",
    "prompt_template": "current_prompt",
}
WITH_CAP = dict(HEADLINE_NO_CAP, use_caption="true")


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
        if not gl: continue
        if gl == a or gl in a or a in gl: return 1.0
    return 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = ROOT / "results" / "demo_cases.json"
    cases = json.loads(p.read_text(encoding="utf-8"))

    print(f"{'id':<22} {'baseline':<32} {'rag':<32}")
    print("-" * 92)
    updated = 0
    for c in cases:
        if c["id"] not in MAIN_IDS:
            c["is_main_demo"] = False
            continue
        c["is_main_demo"] = True
        img = ROOT / c["image_path"].replace("\\", "/")
        if not img.exists():
            print(f"SKIP {c['id']} — missing {img}")
            continue
        try:
            r1 = call(img, c["question"], HEADLINE_NO_CAP)
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

        c.update({
            "baseline_pred": baseline, "baseline_acc": base_acc,
            "rag_pred": rag_20k, "rag_acc": rag_acc,
            "delta": rag_acc - base_acc,
            "caption": caption,
            "candidate_answers": candidates, "most_common_answer": most_common,
            "retrieved": retrieved,
            "baseline_answer": baseline,
            "rag_20k_answer": rag_20k,
            "caption_answer": cap_ans,
            "raw_rag_answer": rag_20k,
            "final_display_answer": rag_20k,
        })
        c["retrieved_examples"] = [
            {"rank": i + 1,
             "question": r.get("question", ""), "answer": r.get("answer", ""),
             "image_score": r.get("img_score", 0.0),
             "text_score": r.get("q_score", 0.0),
             "final_score": r.get("score", 0.0),
             "image_path": "", "_source": "20k diverse"}
            for i, r in enumerate(retrieved)
        ]
        c["evidence_trace"] = [
            f"Visual summary: {caption}" if caption else "Visual summary: (no caption)",
            f"User question: {c['question']}",
            "Retrieved similar examples: " + " · ".join(
                f"[{r.get('question','')} → {r.get('answer','')}]" for r in retrieved[:3]
            ),
            "Candidate answer priors: " + ", ".join(candidates),
            f"Final answer: {rag_20k}",
        ]
        c["modes"] = [
            {"name": "Baseline BLIP-2 (frozen, no retrieval)",
             "answer": baseline, "correct": base_acc >= 1.0,
             "note": "Frozen BLIP-2 alone"},
            {"name": "20k diverse RAG (image-only retrieval)",
             "answer": rag_20k, "correct": rag_acc >= 1.0,
             "note": f"Top retrieval score {retrieved[0].get('img_score', 0):.3f}" if retrieved else ""},
            {"name": "20k RAG + caption in prompt",
             "answer": cap_ans, "correct": vqa_correct(cap_ans, gt_all) >= 1.0,
             "note": f'Caption: "{caption}"'},
        ]
        updated += 1
        print(f"{c['id']:<22} {baseline!r:<32} {rag_20k!r:<32}")

    p.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"\nRestored {updated} cases. Main lineup is 7 (F demoted).")


if __name__ == "__main__":
    main()
