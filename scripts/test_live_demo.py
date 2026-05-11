"""Smoke-test the 4 live-demo images against the running API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8000"
IMG_DIR = ROOT / "data" / "live_demo_images"

# (image, list of test questions)
TESTS = [
    ("cat.png", [
        "what color is the cat?",
        "is the cat sleeping?",
        "what is the cat sitting on?",
    ]),
    ("pizza.png", [
        "what kind of food is this?",
        "is this a pizza?",
        "are people eating?",
    ]),
    ("cyclist.png", [
        "what is the person doing?",
        "is the person wearing a helmet?",
        "what is the bike's color?",
    ]),
    ("baseball.png", [
        "what sport is this?",
        "is the player batting?",
        "what is the catcher wearing?",
    ]),
]

# Best-validated config (matches benchmark)
DEFAULT_FORM = {
    "top_k":            "3",
    "alpha":            "1.0",
    "tau":              "0.70",   # gates OOD live uploads
    "use_caption":      "false",
    "caption_weight":   "0.0",
    "rerank":           "false",
    "use_answer_prior": "false",
    "filter_hints":     "false",
    "type_gate":        "true",
    "prompt_template":  "current_prompt",
}


def call_api(img_path: Path, question: str) -> dict | None:
    with open(img_path, "rb") as f:
        files = {"image": (img_path.name, f.read(), "image/png")}
    data = dict(DEFAULT_FORM, question=question)
    r = requests.post(f"{API}/infer", files=files, data=data, timeout=120)
    if not r.ok:
        return None
    return r.json()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Testing 4 live demo images with default best config\n")
    print("τ=0.70  ·  α=1.0  ·  k=3  ·  caption=off  ·  type-gate=on  ·  prompt=current\n")
    print("=" * 96)

    out = []
    for img_name, questions in TESTS:
        img = IMG_DIR / img_name
        if not img.exists():
            print(f"SKIP {img_name} — not found at {img}")
            continue

        print(f"\n📷 {img_name}")
        print("-" * 96)
        for q in questions:
            t0 = time.time()
            d = call_api(img, q)
            elapsed = time.time() - t0
            if d is None:
                print(f"  [API FAIL] {q}")
                continue
            base = d["baseline"]
            rag = d.get("final_answer") or d["rag"]
            score = d.get("top_img_score", 0.0)
            gated = d.get("type_gate_applied", False)
            rag_used = d.get("rag_used", True)

            tag = ""
            if gated:    tag = " [yn → BASELINE]"
            elif not rag_used: tag = " [τ-gated → BASELINE]"

            print(f"  Q: {q}")
            print(f"     baseline:  {base!r}")
            print(f"     final:     {rag!r}{tag}   (top_score={score:.3f}, {elapsed:.1f}s)")
            out.append({
                "image": img_name, "question": q,
                "baseline": base, "final": rag,
                "top_img_score": score, "type_gate_applied": gated,
                "rag_used": rag_used, "elapsed_sec": round(elapsed, 2),
            })

    print("\n" + "=" * 96)
    out_path = ROOT / "results" / "experiments" / "live_demo_smoketest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
