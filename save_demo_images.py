"""Save demo images from VQAv2 for the Netlify site gallery.

Usage: python save_demo_images.py
"""

import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("site/assets/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Define demo examples ---
# Each: (demo_id, query_qid, retrieved_questions_to_match)
# We'll find query images in eval split (offset=5000) and retrieved images in KB (offset=0..5000)

DEMOS = [
    {
        "id": 1,
        "qid": 264683001,  # airplane
        "retrieved_qs": [
            "How many engines does the plane have?",
            "How many planes?",
            "Is the plane at an airport?",
        ],
    },
    {
        "id": 2,
        "qid": 133622006,  # food plate — eggs
        "retrieved_qs": [
            "What is this?",
            "Is this a meal for a child?",
            "Is this a sandwich with a lot of bread?",
        ],
    },
    {
        "id": 3,
        "qid": 395801000,  # night city scene
        "retrieved_qs": [
            "Are street lights on?",
            "Is this a night or daytime shot?",
            "What are the long lines of light in the picture?",
        ],
    },
    {
        "id": 4,
        "qid": 133631001,  # elephants
        "retrieved_qs": [
            "Are these elephants inside or outside?",
            "Does the smaller elephant have tusks?",
            "Are all of the elephants the same size?",
        ],
    },
    {
        "id": 5,
        "qid": 2532003,  # skiing — RAG ignored
        "retrieved_qs": [
            "Are they doing the same sport?",
            "Has this area been windy lately?",
            "Is this the morning?",
        ],
    },
    {
        "id": 6,
        "qid": 133636001,  # parking meter
        "retrieved_qs": [
            "Could you see someone clearly in the reflection of this object?",
            "Is the water turned on or off?",
            "Where is the electrical outlet?",
        ],
    },
]


def main():
    print("Loading VQAv2 validation split...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)

    # We need:
    # - KB images: val[0:5000] — for retrieved images
    # - Eval images: val[5000:5100] — for query images
    # Total needed: up to index ~5100

    # Collect query qids we need
    query_qids = {d["qid"] for d in DEMOS}

    # Collect retrieved question texts we need (for KB image lookup)
    retrieved_q_texts = set()
    for d in DEMOS:
        for q in d["retrieved_qs"]:
            retrieved_q_texts.add(q)

    # Build lookup: question_text -> image (from KB, first 5000)
    # Also save query images from eval range
    kb_q_to_image = {}
    saved_queries = set()
    saved_retrieved = set()

    print("Scanning dataset for matching images...")
    for idx, sample in enumerate(ds):
        if idx >= 5200:
            break

        # KB range: look for retrieved question matches
        if idx < 5000:
            q_text = sample["question"]
            if q_text in retrieved_q_texts and q_text not in kb_q_to_image:
                kb_q_to_image[q_text] = sample["image"].convert("RGB")
                print(f"  KB match [{idx}]: {q_text[:50]}")

        # Eval range: look for query images
        if 5000 <= idx < 5200:
            qid = sample.get("question_id")
            if qid in query_qids and qid not in saved_queries:
                # Find which demo this belongs to
                for d in DEMOS:
                    if d["qid"] == qid:
                        path = OUT_DIR / f"q{d['id']}.jpg"
                        sample["image"].convert("RGB").save(path, "JPEG", quality=85)
                        saved_queries.add(qid)
                        print(f"  Saved query image: {path}")
                        break

        # Early exit if we have everything
        if len(saved_queries) == len(DEMOS) and len(kb_q_to_image) >= len(retrieved_q_texts):
            print("Found all needed images!")
            break

    # Save retrieved images
    for d in DEMOS:
        for r_idx, q_text in enumerate(d["retrieved_qs"], 1):
            path = OUT_DIR / f"q{d['id']}_r{r_idx}.jpg"
            if q_text in kb_q_to_image:
                kb_q_to_image[q_text].save(path, "JPEG", quality=85)
                print(f"  Saved retrieved: {path}")
            else:
                print(f"  WARNING: No KB image found for: {q_text[:50]}")

    # Summary
    print(f"\nDone! Saved {len(saved_queries)} query images and retrieved images to {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.jpg")):
        print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()