"""Wide probe: 40+ varied questions across the 4 live images.
Find cases where RAG genuinely produces the more impressive answer.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8000"
IMG_DIR = ROOT / "data" / "live_demo_images"

PROBES = {
    "cat.png": [
        # color/object
        "what color is the cat?",
        "what color is the couch?",
        "what is the cat lying on?",
        "is the cat wearing a collar?",
        # action / state
        "what is the cat doing?",
        "is the cat sleeping?",
        "is the cat awake?",
        "is the cat indoors?",
        # scene
        "what room is this?",
        "is this indoors or outdoors?",
        # counting
        "how many cats are in the picture?",
        # appearance
        "is the cat tabby?",
        "does the cat have stripes?",
    ],
    "pizza.png": [
        # identification
        "what kind of food is this?",
        "what type of pizza is this?",
        "what toppings does the pizza have?",
        # location
        "where is the pizza?",
        "is the pizza in a box?",
        "is the pizza on a plate?",
        # state
        "is the pizza whole or sliced?",
        "is this pizza fresh?",
        "is the pizza hot?",
        # counting
        "how many slices are there?",
        "how many pizzas are visible?",
        # context
        "is anyone eating the pizza?",
    ],
    "cyclist.png": [
        # action
        "what is the person doing?",
        "is the person riding a bike?",
        "is the person standing still?",
        # gear
        "is the person wearing a helmet?",
        "what is the person wearing?",
        "what color is the bike?",
        # scene
        "where is this taking place?",
        "is this on a road?",
        "what is the weather like?",
        "is it sunny?",
        "is it raining?",
        # counting
        "how many people are visible?",
    ],
    "baseball.png": [
        # action
        "is the player batting?",
        "what is the player doing?",
        "what is the player about to do?",
        # equipment
        "what is the player holding?",
        "is the player wearing a helmet?",
        "what color is the bat?",
        # scene
        "what sport is this?",
        "is this a baseball game?",
        "is this on a field?",
        # counting
        "how many players are visible?",
    ],
}

NO_GATE = {
    "top_k": "3", "alpha": "1.0", "tau": "0.0",
    "use_caption": "false", "caption_weight": "0.0",
    "rerank": "false", "use_answer_prior": "false",
    "filter_hints": "false", "type_gate": "false",
    "prompt_template": "current_prompt",
}

def call(img: Path, q: str) -> dict:
    with open(img, "rb") as f:
        files = {"image": (img.name, f.read(), "image/png")}
    data = dict(NO_GATE, question=q)
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = []
    print("Wide probe — 40+ questions\n" + "="*100)
    for img_name, questions in PROBES.items():
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
            base_words = len(base.split())
            rag_words  = len(rag.split())
            top_score  = d.get("retrieved",[{}])[0].get("img_score", 0.0) if d.get("retrieved") else 0.0
            print(f"  Q: {q}")
            print(f"     base={base!r:35}  rag={rag!r:30}  topScore={top_score:.3f}")
            results.append({
                "image": img_name, "question": q,
                "baseline": base, "rag": rag, "same": same,
                "base_words": base_words, "rag_words": rag_words,
                "top_img_score": top_score,
                "retrieved_answers": [r.get("answer") for r in d.get("retrieved", [])][:3],
            })
    out = ROOT / "results" / "experiments" / "wide_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}\n")

    # Rank-impressive: cases where rag differs and rag is multi-word, plus retrieval is high-confidence
    impressive = [r for r in results
                  if not r["same"]
                  and r["rag_words"] >= 2
                  and r["top_img_score"] >= 0.75]
    print("="*100)
    print(f"\nIMPRESSIVE (RAG multi-word + DIFF + high retrieval, n={len(impressive)}):")
    for r in impressive:
        print(f"  [{r['image']}] {r['question']!r}")
        print(f"     base={r['baseline']!r}")
        print(f"     rag={r['rag']!r}   (top={r['top_img_score']:.3f})")

    # Yes/no fixes (clear visual win — yes vs no diff)
    yn_fixes = [r for r in results
                if r["base_words"] == 1 and r["rag_words"] == 1
                and {r["baseline"].lower(), r["rag"].lower()} == {"yes", "no"}]
    print(f"\nYES/NO FLIPS (n={len(yn_fixes)}):")
    for r in yn_fixes:
        print(f"  [{r['image']}] {r['question']!r}  →  base={r['baseline']!r}  rag={r['rag']!r}")

if __name__ == "__main__":
    main()
