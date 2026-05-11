"""Probe the 4 live images with action/state questions to find multi-word
baseline-vs-RAG diffs worth featuring in the curated demo."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8000"
IMG_DIR = ROOT / "data" / "live_demo_images"

PROBES = [
    ("cat.png", [
        "what is the cat doing?",
        "describe what is happening",
        "what is the cat sitting on?",
        "what room is this?",
        "is the cat looking at the camera?",
    ]),
    ("pizza.png", [
        "what toppings are on the pizza?",
        "where is the pizza?",
        "what kind of pizza is this?",
        "is the pizza fresh?",
        "describe what is happening",
    ]),
    ("cyclist.png", [
        "what is the person wearing?",
        "where is this taking place?",
        "what is the weather like?",
        "what is in the background?",
        "describe what is happening",
    ]),
    ("baseball.png", [
        "what is the player about to do?",
        "what is the player holding?",
        "what color is the uniform?",
        "where is this taking place?",
        "describe what is happening",
    ]),
]

# raw RAG (no router, no τ-gate, captures full diff vs baseline)
NO_CAP = {
    "top_k": "3", "alpha": "1.0", "tau": "0.0",
    "use_caption": "false", "caption_weight": "0.0",
    "rerank": "false", "use_answer_prior": "false",
    "filter_hints": "false", "type_gate": "false",
    "prompt_template": "current_prompt",
}

def call(img: Path, q: str) -> dict:
    with open(img, "rb") as f:
        files = {"image": (img.name, f.read(), "image/png")}
    data = dict(NO_CAP, question=q)
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = []
    for img_name, questions in PROBES:
        img = IMG_DIR / img_name
        if not img.exists():
            continue
        print(f"\n=== {img_name} ===")
        for q in questions:
            try:
                d = call(img, q)
            except Exception as e:
                print(f"  FAIL {q!r}: {e}")
                continue
            base = (d.get("baseline") or "").strip()
            rag  = (d.get("rag") or "").strip()
            same = base.lower() == rag.lower()
            multi = max(len(base.split()), len(rag.split())) >= 2
            mark = ""
            if multi and not same: mark = "  ★★★ MULTI+DIFF"
            elif multi:            mark = "  ★ multi"
            elif not same:         mark = "  · diff"
            print(f"  Q: {q}")
            print(f"     base: {base!r}   rag: {rag!r}{mark}")
            results.append({
                "image": img_name, "question": q,
                "baseline": base, "rag": rag,
                "same": same, "multi": multi,
                "retrieved": [r.get("answer") for r in d.get("retrieved", [])],
                "top_img_score": d.get("retrieved",[{}])[0].get("img_score", 0.0) if d.get("retrieved") else 0.0,
            })

    out = ROOT / "results" / "experiments" / "multiword_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("\n=== MULTI+DIFF candidates ===")
    for r in results:
        if r["multi"] and not r["same"]:
            print(f"  [{r['image']}] {r['question']!r}")
            print(f"      base={r['baseline']!r}  rag={r['rag']!r}")

if __name__ == "__main__":
    main()
