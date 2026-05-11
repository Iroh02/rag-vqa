"""Rebuild the curated demo around the strongest RAG wins from wide_probe.

Adds 3 new high-impact cases (cat-sleeping, cyclist-weather, pizza-eating)
where RAG genuinely fixes a baseline hallucination, and demotes the weak
multi-word cases (L5, L6, L7) to advanced.
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
        "id": "L8_cat_sleeping",
        "category": "Real-World — Live Image",
        "title": "Real photo · *is the cat sleeping?* — RAG fixes baseline state hallucination",
        "image": "cat.png",
        "question": "is the cat sleeping?",
        "ground_truth": "no",
        "gt_all": ["no"],
        "verdict_story": "helpful",
        "why":  ("BLIP-2 baseline answers <b>'yes'</b> — the cat is not sleeping in this photo, baseline "
                 "hallucinates state. RAG retrieves similar cats-on-couches scenes from the 20k visual "
                 "memory and answers <b>'no'</b> — correct. <b>Subtle scene-state correction</b>: not "
                 "object identity but <em>activity inference</em>, which baseline gets wrong on novel images."),
        "demo_script": ("Cat photo, state question: <em>is the cat sleeping?</em>. Baseline hallucinates "
                        "<em>yes</em> — wrong. RAG looks at similar cat-on-couch photos in the visual memory, "
                        "most of which show awake cats, answers <em>no</em>. This is RAG correcting a state-inference error."),
        "presenter_notes": {
            "what_to_say": "Strong second-card after cat-colour. Same image, different question, same lesson — RAG fixes hallucinations.",
            "what_it_proves": "RAG fixes more than just object identity. State and activity inference improve too.",
            "caveat": "Top retrieval 0.82 — KB has good cat coverage. Without that, the system can't do this kind of correction.",
        },
    },
    {
        "id": "L9_cyclist_weather",
        "category": "Real-World — Live Image",
        "title": "Real photo · *what is the weather like?* — multi-word RAG fixes 'raining' hallucination",
        "image": "cyclist.png",
        "question": "what is the weather like?",
        "ground_truth": "sunny",
        "gt_all": ["sunny", "sunny and warm", "sunny and clear", "clear", "warm", "fair"],
        "verdict_story": "helpful",
        "why":  ("BLIP-2 baseline answers <b>'raining'</b> — clearly wrong, the photo is sunny. RAG "
                 "retrieves outdoor cycling scenes (which are predominantly sunny) and produces a "
                 "<b>multi-word answer: 'sunny and warm'</b>. Strong correction <em>and</em> a richer, "
                 "more descriptive output than baseline ever produces — exactly the kind of "
                 "narrative VQA answer that lives beyond yes/no."),
        "demo_script": ("Cyclist photo, weather question. Baseline hallucinates <em>raining</em> — wrong, "
                        "the photo is clearly sunny. RAG over the 20k visual memory says <em>sunny and warm</em> — "
                        "correct, multi-word, descriptive. This is the kind of answer that proves the "
                        "system handles open-ended VQA, not just yes/no."),
        "presenter_notes": {
            "what_to_say": "Show this as the multi-word RAG win. 'Sunny and warm' beats 'raining' on every dimension — accuracy and descriptiveness.",
            "what_it_proves": "RAG produces rich multi-word answers when the question demands it. Not limited to single-token VQA outputs.",
            "caveat": "Top retrieval 0.81 — outdoor cycling scenes dominate the KB and they're mostly sunny.",
        },
    },
    {
        "id": "L10_pizza_eating",
        "category": "Real-World — Live Image",
        "title": "Real photo · *is anyone eating the pizza?* — RAG fixes scene-context hallucination",
        "image": "pizza.png",
        "question": "is anyone eating the pizza?",
        "ground_truth": "no",
        "gt_all": ["no"],
        "verdict_story": "helpful",
        "why":  ("Photo shows a pizza on a table — <b>no people in frame</b>. BLIP-2 baseline hallucinates "
                 "<b>'yes'</b> based on common pizza-context priors (pizzas are usually being eaten). "
                 "RAG retrieves similar food-photography scenes and answers <b>'no'</b> — grounded "
                 "in what's actually in the image, not what's typical. <b>Scene-context correction</b>."),
        "demo_script": ("Pizza photo, scene-context question. Baseline says <em>yes</em> — but no one is "
                        "in the image. The model is hallucinating from prior. RAG retrieves food-photography "
                        "scenes (also no people) and grounds back to <em>no</em>. RAG corrects a context-prior bias."),
        "presenter_notes": {
            "what_to_say": "Show this as the third real-photo fix. Different mechanism from cat-colour — this is bias correction, not visual perception.",
            "what_it_proves": "RAG can correct LM-prior hallucinations by anchoring on visual evidence.",
            "caveat": "Top retrieval 0.88 — food photos are well-covered. KB scale matters here.",
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
        gl = (g or "").strip().lower()
        if not gl:
            continue
        if gl == a or gl in a or a in gl:
            return 1.0
    return 0.0


def build_case(spec: dict) -> dict | None:
    img = IMG_DIR / spec["image"]
    if not img.exists():
        print(f"SKIP — missing {img}")
        return None
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
    print(f"   baseline: {baseline!r}")
    print(f"   rag_20k:  {rag_20k!r}")
    print(f"   cap_ans:  {cap_ans!r}")
    return {
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    new_cases = [c for c in (build_case(s) for s in CASES) if c]

    cases_path = ROOT / "results" / "demo_cases.json"
    existing = json.loads(cases_path.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in existing}

    # 1) Add or replace the new cases
    appended = 0
    for c in new_cases:
        if c["id"] in existing_ids:
            for i, ex in enumerate(existing):
                if ex["id"] == c["id"]:
                    existing[i] = c
                    break
        else:
            existing.append(c)
            appended += 1

    # 2) Set the main demo lineup
    MAIN = {
        "B_visual_memory",      # eval — yes ✗ → no ✓
        "C_kb_scaling",         # eval — yes ✗ → no ✓
        "A_open_ended",         # eval — verbose → canonical
        "F_retrieval_noise",    # eval — honest weakness
        "L1_cat_color",         # real — green ✗ → gray ✓
        "L8_cat_sleeping",      # real — yes ✗ → no ✓
        "L9_cyclist_weather",   # real — multi-word weather fix
        "L10_pizza_eating",     # real — yes ✗ → no ✓
    }
    for c in existing:
        c["is_main_demo"] = (c["id"] in MAIN)

    cases_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    main_count = sum(1 for c in existing if c.get("is_main_demo"))
    print(f"\nWrote {len(existing)} cases ({main_count} main, {appended} new) to {cases_path}")
    print("\nMain lineup:")
    for c in existing:
        if c.get("is_main_demo"):
            print(f"  {c['id']:<22}  {c['baseline_answer']!r:35} → {c['rag_20k_answer']!r}")


if __name__ == "__main__":
    main()
