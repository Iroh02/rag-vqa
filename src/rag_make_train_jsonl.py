"""Generate RAG-augmented training JSONL from VQAv2.

Runs CLIP retrieval for each sample and builds Hints-style prompts.
Output JSONL can be used by finetune.py for QLoRA training.

Usage:
    # Generate training data (validation[5100:8100], disjoint from KB and eval)
    python -m src.rag_make_train_jsonl --offset 5100 --num_samples 3000

    # Generate eval data (validation[5000:5100], same as our standard eval)
    python -m src.rag_make_train_jsonl --offset 5000 --num_samples 100 \
        --output data/rag_eval.jsonl --no_hints_ratio 0.0
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_DIR, RAG_TOP_K
from src.dataset import get_best_answer, load_vqav2
from src.model import build_hints_prompt
from src.rag_retriever import RAGRetriever


def make_train_jsonl(offset=5100, num_samples=3000, top_k=RAG_TOP_K,
                     output_path=None, no_hints_ratio=0.15):
    """Generate JSONL with RAG-style hints prompts for fine-tuning.

    Args:
        offset: VQAv2 validation offset (skip KB + eval samples)
        num_samples: Number of samples to generate
        top_k: Number of retrieved examples for hints
        output_path: Output JSONL path
        no_hints_ratio: Fraction of samples without hints (teaches baseline behavior)
    """
    if output_path is None:
        output_path = DATA_DIR / "rag_train.jsonl"
    output_path = Path(output_path)

    # Load retriever (CLIP on CPU + FAISS indices)
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()

    # Load VQAv2 samples
    print(f"Loading VQAv2 validation (offset={offset}, n={num_samples})...")
    samples = load_vqav2(split="validation", num_samples=num_samples, offset=offset)

    # Generate JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    no_hints_count = 0

    with open(output_path, "w") as f:
        # Write metadata line
        meta = {
            "_meta": True,
            "offset": offset,
            "num_samples": len(samples),
            "split": "validation",
            "top_k": top_k,
            "no_hints_ratio": no_hints_ratio,
        }
        f.write(json.dumps(meta) + "\n")

        for i in tqdm(range(len(samples)), desc="Building prompts"):
            item = samples[i]
            image = item["image"].convert("RGB")
            question = item["question"]
            best_answer = get_best_answer(item["answers"])
            all_answers = [a["answer"] for a in item["answers"]]
            qid = item.get("question_id", i)

            # Randomly skip hints for some samples (teaches baseline behavior)
            skip_hints = random.random() < no_hints_ratio

            if skip_hints:
                prompt = f"Question: {question} Answer:"
                no_hints_count += 1
                retrieved_info = []
            else:
                # Retrieve using image-only (alpha=1.0)
                retrieved = retriever.retrieve(
                    image, question, top_k=top_k, alpha=1.0,
                )
                prompt = build_hints_prompt(question, retrieved)
                retrieved_info = [
                    {"answer": r["best_answer"], "score": round(r["score"], 4)}
                    for r in retrieved
                ]

            entry = {
                "index": i,
                "question_id": qid,
                "prompt": prompt,
                "answer": best_answer,
                "all_answers": all_answers,
                "has_hints": not skip_hints,
                "retrieved": retrieved_info,
            }
            f.write(json.dumps(entry) + "\n")

    retriever.unload_clip()

    print(f"Wrote {len(samples)} entries to {output_path}")
    print(f"  With hints: {len(samples) - no_hints_count}, "
          f"without: {no_hints_count} ({no_hints_ratio*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG training JSONL")
    parser.add_argument("--offset", type=int, default=5100)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--top_k", type=int, default=RAG_TOP_K)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_hints_ratio", type=float, default=0.15)
    args = parser.parse_args()

    make_train_jsonl(
        offset=args.offset,
        num_samples=args.num_samples,
        top_k=args.top_k,
        output_path=args.output,
        no_hints_ratio=args.no_hints_ratio,
    )
