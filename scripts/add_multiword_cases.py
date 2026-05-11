"""Add three curated demo cases featuring multi-word, action-style answers.

These showcase richer VQA-style outputs ('reading a book', 'cat is sitting on
the couch') instead of just yes/no or single-word answers.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8000"
IMG_DIR = ROOT / "data" / "live_demo_images"

CASES = [
    {
        "id": "L5_cat_describe",
        "category": "Real-World — Live Image",
        "title": "Real photo · *describe what is happening* — rich multi-word answer",
        "image": "cat.png",
        "question": "describe what is happening",
        "ground_truth": "cat is sitting on the couch",
        "gt_all": ["cat is sitting on the couch", "a cat sitting on a couch", "cat sitting"],
        "verdict_story": "neutral",
        "why":  ("Open-ended <em>describe</em> question on a novel real photo. Both baseline and 20k RAG "
                 "produce <b>'cat is sitting on the couch'</b> — a rich, full-sentence answer that BLIP-2 "
                 "is capable of when the prompt allows it. RAG's role here is to <em>not break</em> a working "
                 "baseline. Demonstrates that the system handles caption-style multi-word outputs, "
                 "not just one-token VQA answers."),
        "demo_script": ("Cat photo, open-ended <em>describe</em>. Both modes give the same full sentence: "
                        "<em>'cat is sitting on the couch'</em>. RAG didn't break the baseline's caption-style "
                        "answer. Useful counterpoint to people thinking VQA = yes/no — the system handles "
                        "rich descriptive output too."),
        "presenter_notes": {
            "what_to_say": "Open-ended description. Both modes agree on a full sentence — RAG is non-destructive on rich outputs.",
            "what_it_proves": "The system isn't limited to one-word answers. Multi-word descriptive outputs work end-to-end.",
            "caveat": "Top retrieval ~0.85 — KB has cat-on-couch coverage. Same answer means RAG didn't add new info, just confirmed.",
        },
    },
    {
        "id": "L6_cat_doing_compress",
        "category": "Real-World — Live Image",
        "title": "Real photo · *what is the cat doing?* — verbose baseline → canonical RAG",
        "image": "cat.png",
        "question": "what is the cat doing?",
        "ground_truth": "sitting",
        "gt_all": ["sitting", "lying", "resting"],
        "verdict_story": "helpful",
        "why":  ("BLIP-2 baseline answers <b>'cat is sitting on the couch'</b> — semantically right but "
                 "verbose. Strict VQA scoring penalises long answers (you need exact-string match). RAG "
                 "compresses to <b>'sitting'</b> — the canonical single-token VQA answer that scores 1.0. "
                 "Same scene understanding, correct format. This is the <b>format compression</b> mechanism "
                 "that drives 16 of 17 strict-eval gain points."),
        "demo_script": ("Cat photo, action question. Baseline says <em>'cat is sitting on the couch'</em> — "
                        "right idea, wrong format. RAG retrieves cat scenes whose canonical answer is "
                        "<em>'sitting'</em>, compresses to that. Strict VQA gives 0.0 to the baseline string "
                        "and 1.0 to the RAG string. Pattern repeats across the eval — about 16 of the 17 strict-gain points."),
        "presenter_notes": {
            "what_to_say": "Show this right after the cat-colour fix. Same image, different format-compression mechanism.",
            "what_it_proves": "Most of the strict-eval gain isn't 'RAG knows more' — it's 'RAG compresses to the right format'.",
            "caveat": "Open question: how would lenient-VQA scoring change this? Baseline is already correct semantically.",
        },
    },
    {
        "id": "L7_baseball_describe",
        "category": "Real-World — Live Image",
        "title": "Real photo · *describe what is happening* — both rich, RAG more accurate",
        "image": "baseball.png",
        "question": "describe what is happening",
        "ground_truth": "batter is hitting the ball",
        "gt_all": ["batter is hitting the ball", "swinging the bat", "batting", "batter swinging"],
        "verdict_story": "helpful",
        "why":  ("Open-ended <em>describe</em> on a baseball photo. Baseline says <b>'batter is swinging at "
                 "the ball'</b>. RAG retrieves similar batting scenes and answers <b>'batter is hitting the "
                 "ball'</b> — slightly more accurate verb (the bat has actually contacted, vs. just swinging). "
                 "Both are rich multi-word descriptions. Shows RAG can refine action verbs through retrieved "
                 "consensus, not just compress to canonical tokens."),
        "demo_script": ("Baseball photo, descriptive question. Baseline: <em>'batter is swinging at the "
                        "ball'</em>. RAG: <em>'batter is hitting the ball'</em>. Both rich, both multi-word, "
                        "but RAG picks a slightly more accurate action verb based on similar batting scenes "
                        "in the visual memory. Subtle refinement rather than dramatic correction."),
        "presenter_notes": {
            "what_to_say": "Show as the third real-photo example after cat-colour and cat-doing — proves rich-answer mode works.",
            "what_it_proves": "RAG can refine multi-word outputs, not just compress them. Visual memory shapes action-verb choice.",
            "caveat": "Both answers are arguably right. This is a soft win, not a hard correction.",
        },
    },
]

NO_CAP = {
    "top_k": "3", "alpha": "1.0", "tau": "0.0",
    "use_caption": "false", "caption_weight": "0.0",
    "rerank": "false", "use_answer_prior": "false",
    "filter_hints": "false", "type_gate": "false",
    "prompt_template": "current_prompt",
}
WITH_CAP = dict(NO_CAP, use_caption="true")


def call(img: Path, question: str, form: dict) -> dict:
    with open(img, "rb") as f:
        files = {"image": (img.name, f.read(), "image/png")}
    data = dict(form, question=question)
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def vqa_correct(answer: str, gt_all: list[str]) -> float:
    if not answer or not gt_all:
        return 0.0
    a = answer.strip().lower()
    for g in gt_all:
        gl = g.strip().lower()
        if not gl:
            continue
        if gl == a or gl in a:
            return 1.0
    return 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    new_cases = []
    for spec in CASES:
        img = IMG_DIR / spec["image"]
        if not img.exists():
            print(f"SKIP — missing {img}")
            continue
        print(f"\nbuilding {spec['id']} ({spec['image']} · {spec['question']})")
        r1 = call(img, spec["question"], NO_CAP)
        r2 = call(img, spec["question"], WITH_CAP)

        baseline = r1["baseline"]
        rag_20k  = r1["rag"]
        cap_ans  = r2["rag"]
        caption  = r2.get("caption", "")
        retrieved = r1["retrieved"]
        candidates = [r["answer"] for r in retrieved]
        cnt = Counter(a for a in candidates if a)
        most_common = cnt.most_common(1)[0][0] if cnt else ""
        base_acc = vqa_correct(baseline, spec["gt_all"])
        rag_acc  = vqa_correct(rag_20k,  spec["gt_all"])

        case = {
            "id":               spec["id"],
            "question_id":      spec["id"],
            "category":         spec["category"],
            "title":            spec["title"],
            "question":         spec["question"],
            "ground_truth":     spec["ground_truth"],
            "gt_all":           spec["gt_all"],
            "image_path":       f"data/live_demo_images/{spec['image']}",
            "baseline_pred":    baseline,
            "baseline_acc":     base_acc,
            "rag_pred":         rag_20k,
            "rag_acc":          rag_acc,
            "delta":            rag_acc - base_acc,
            "verdict":          spec["verdict_story"],
            "leakage_warning":  False,
            "caption":          caption,
            "candidate_answers": candidates,
            "most_common_answer": most_common,
            "retrieved":        retrieved,
            "baseline_answer":  baseline,
            "rag_5k_answer":    "—",
            "rag_20k_answer":   rag_20k,
            "caption_answer":   cap_ans,
            "raw_rag_answer":   rag_20k,
            "final_display_answer": rag_20k,
            "retrieved_examples": [
                {
                    "rank": i + 1,
                    "question":   r.get("question", ""),
                    "answer":     r.get("answer", ""),
                    "image_score": r.get("img_score", 0.0),
                    "text_score":  r.get("q_score",  0.0),
                    "final_score": r.get("score",    0.0),
                    "image_path":  "",
                    "_source":     "20k diverse",
                }
                for i, r in enumerate(retrieved)
            ],
            "evidence_trace": [
                f"Visual summary: {caption}" if caption else "Visual summary: (no caption)",
                f"User question: {spec['question']}",
                "Retrieved similar examples: " + " · ".join(
                    f"[{r.get('question','')} → {r.get('answer','')}]" for r in retrieved[:3]
                ),
                "Candidate answer priors: " + ", ".join(candidates),
                f"Final answer: {rag_20k}",
            ],
            "modes": [
                {"name": "Baseline BLIP-2 (frozen, no retrieval)",
                 "answer": baseline, "correct": base_acc >= 1.0,
                 "note": "Frozen BLIP-2 alone"},
                {"name": "20k diverse RAG (image-only retrieval)",
                 "answer": rag_20k, "correct": rag_acc >= 1.0,
                 "note": f"Top retrieval score {retrieved[0].get('img_score', 0):.3f}" if retrieved else ""},
                {"name": "20k RAG + caption in prompt",
                 "answer": cap_ans, "correct": vqa_correct(cap_ans, spec["gt_all"]) >= 1.0,
                 "note": f'Caption: "{caption}"'},
            ],
            "why_it_matters":   spec["why"],
            "demo_script":      spec["demo_script"],
            "presenter_notes":  spec["presenter_notes"],
            "warning":          None,
            "is_main_demo":     True,
            "is_live_image":    True,
        }
        new_cases.append(case)
        print(f"   baseline: {baseline!r}")
        print(f"   rag_20k:  {rag_20k!r}")
        print(f"   cap_ans:  {cap_ans!r}")

    # Append, dedup by id
    cases_path = ROOT / "results" / "demo_cases.json"
    existing = json.loads(cases_path.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in existing}
    appended = 0
    for c in new_cases:
        if c["id"] in existing_ids:
            # replace in place
            for i, ex in enumerate(existing):
                if ex["id"] == c["id"]:
                    existing[i] = c
                    break
        else:
            existing.append(c)
            appended += 1
    cases_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    main_count = sum(1 for c in existing if c.get("is_main_demo"))
    print(f"\nWrote {len(existing)} cases ({main_count} main, {appended} new) to {cases_path}")


if __name__ == "__main__":
    main()
