"""Precompute 20k RAG answers + captions for the 30 fresh demo cases.

Calls the running local API (http://localhost:8000) — much faster than
re-loading BLIP-2. ~30 cases × 2 modes (no-caption, with-caption) = ~60 calls.

Output: data/precomputed_demo_data.json

This is demo data prep, not a new experiment — does not change reported numbers.
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API  = "http://localhost:8000"

def call_api(image_path, question, use_caption):
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f.read(), "image/jpeg")}
    data = {
        "question":         question,
        "top_k":             "3",
        "alpha":             "1.0",
        "tau":               "0.0",
        "use_caption":       "true" if use_caption else "false",
        "caption_weight":    "0.0",
        "rerank":            "false",
        "use_answer_prior":  "false",
    }
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    fresh = json.loads((ROOT / "data" / "fresh_demo_results.json").read_text())
    out = []
    sys.stdout.reconfigure(encoding="utf-8")

    for i, f in enumerate(fresh):
        img_path = ROOT / f["image_path"]
        if not img_path.exists():
            print(f"[skip] {img_path} not found")
            continue

        print(f"[{i+1:2d}/{len(fresh)}] qid={f['question_id']}  Q: {f['question'][:60]}", flush=True)

        # Pass 1: no caption (pure 20k RAG)
        try:
            t0 = time.time()
            r_no_cap = call_api(img_path, f["question"], use_caption=False)
            print(f"   20k     -> {r_no_cap['rag']!r}   ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"   20k FAIL: {e}", flush=True)
            continue

        # Pass 2: with caption
        try:
            t0 = time.time()
            r_cap = call_api(img_path, f["question"], use_caption=True)
            print(f"   20k+cap -> {r_cap['rag']!r}   caption={r_cap['caption']!r}   ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"   20k+cap FAIL: {e}", flush=True)
            continue

        # Audit retrieval for leakage signs
        retrieved = r_no_cap.get("retrieved", [])
        exact_q_match = any(
            r.get("question", "").strip().lower() == f["question"].strip().lower()
            for r in retrieved
        )
        max_score = max((r.get("img_score", 0) for r in retrieved), default=0)

        out.append({
            "question_id":     f["question_id"],
            "question":        f["question"],
            "ground_truth":    f["ground_truth"],
            "gt_all":          f["gt_all"],
            "image_path":      f["image_path"],
            "baseline":        r_no_cap["baseline"],
            "rag_5k":          f["rag_pred"],
            "rag_20k":         r_no_cap["rag"],
            "rag_20k_caption": r_cap["rag"],
            "caption":         r_cap["caption"],
            "retrieved_20k":   retrieved,
            "candidate_answers":  r_no_cap["candidate_answers"],
            "most_common_answer": r_no_cap["most_common_answer"],
            "audit": {
                "exact_question_in_retrieved": exact_q_match,
                "max_retrieval_score":          round(max_score, 4),
                "near_duplicate_score_warning": max_score >= 0.95,
            },
        })

    out_path = ROOT / "data" / "precomputed_demo_data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} entries to {out_path}")


if __name__ == "__main__":
    main()
