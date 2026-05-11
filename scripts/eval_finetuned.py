"""Evaluate the existing 5-epoch QLoRA checkpoint on the 100-sample eval.

Same eval set, same KB, same top_k/alpha as the 59.33% headline. The only
change is loading the LoRA adapter on top of frozen BLIP-2.

Tests:
  1. fine-tuned model + RAG hints (using `generate_answer_with_hints` —
     the prompt format the model was trained on)
  2. (sanity) fine-tuned model + the standard `current_prompt` Q/A few-shot

Run:
  python -m scripts.eval_finetuned
  python -m scripts.eval_finetuned --epoch 3   # try a different checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.model import (
    load_finetuned_model, generate_answer, generate_answer_with_context,
    generate_answer_with_hints,
)
from src.rag_retriever import RAGRetriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch",        type=int, default=5,
                   help="Which checkpoint epoch to load (1–5)")
    p.add_argument("--num_samples",  type=int, default=100)
    p.add_argument("--top_k",        type=int, default=3)
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--offset",       type=int, default=5000)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ckpt = ROOT / "results" / "checkpoints" / f"rag_checkpoint_epoch_{args.epoch}"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return 1
    print(f"Loading checkpoint: {ckpt}")

    # Retriever
    print("\nLoading retriever (CLIP + 20k FAISS)...")
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()
    print(f"  KB size: {retriever.image_index.ntotal}")

    # Fine-tuned model
    print("\nLoading fine-tuned BLIP-2 (4-bit + LoRA)...")
    model, processor = load_finetuned_model(str(ckpt), quantize=True)

    # Eval samples
    print(f"\nLoading {args.num_samples} eval samples (offset={args.offset})...")
    dataset = load_vqav2(split="validation", num_samples=args.num_samples, offset=args.offset)

    # Run all 3 conditions on the same retrieval
    scores = {"baseline_ft": [], "rag_hints_ft": [], "rag_current_ft": []}
    rows = []
    t0 = time.time()
    for i, item in enumerate(dataset):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]
        qid      = str(item.get("question_id", i))

        retrieved = retriever.retrieve(image, question, top_k=args.top_k, alpha=args.alpha)

        # Condition 1: fine-tuned baseline (no retrieval)
        b = generate_answer(model, processor, image, question)
        # Condition 2: fine-tuned + RAG hints (the format it was trained on)
        try:
            r_h = generate_answer_with_hints(model, processor, image, question, retrieved)
        except Exception as e:
            r_h = ""
        # Condition 3: fine-tuned + current_prompt (Q/A few-shot — for like-for-like comparison vs the 59.33%)
        r_c = generate_answer_with_context(model, processor, image, question, retrieved, caption=None)

        sb = vqa_accuracy(b,   gt_all)
        sh = vqa_accuracy(r_h, gt_all)
        sc = vqa_accuracy(r_c, gt_all)
        scores["baseline_ft"].append(sb)
        scores["rag_hints_ft"].append(sh)
        scores["rag_current_ft"].append(sc)

        rows.append({
            "question_id": qid, "question": question,
            "gt": "|".join(gt_all[:3]),
            "baseline_ft": b, "baseline_ft_score": round(sb, 2),
            "rag_hints_ft": r_h, "rag_hints_ft_score": round(sh, 2),
            "rag_current_ft": r_c, "rag_current_ft_score": round(sc, 2),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dataset) - i - 1)
            print(f"  {i+1}/{len(dataset)}  ({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    # Aggregate
    print("\n" + "=" * 70)
    print(f"FINE-TUNED EVAL — epoch {args.epoch}, n={len(dataset)}, top_k={args.top_k}, α={args.alpha}")
    print("=" * 70)
    for name, sc in scores.items():
        acc = sum(sc) / len(sc) * 100
        print(f"  {name:30s}  {acc:6.2f}%")

    # Compare to known headlines
    print()
    print("Frozen BLIP-2 reference numbers (same eval):")
    print(f"  baseline (frozen)             42.33%")
    print(f"  20k RAG current_prompt        59.33%   ← headline")

    # Save outputs
    exp = ROOT / "results" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    out = {
        "epoch": args.epoch,
        "num_samples": len(dataset),
        "top_k": args.top_k,
        "alpha": args.alpha,
        "accuracies": {k: sum(v)/len(v) for k, v in scores.items()},
    }
    (exp / f"finetuned_eval_epoch{args.epoch}.json").write_text(json.dumps(out, indent=2))
    import csv
    with open(exp / f"finetuned_eval_epoch{args.epoch}_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved to {exp}/finetuned_eval_epoch{args.epoch}.json + .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
