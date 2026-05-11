"""Caption ablation: baseline / +retrieval / +caption / +cap-aug retrieval.

Runs all 4 configs on the same 100 eval samples for an apples-to-apples comparison.
"""
import json
import time
from pathlib import Path
from tqdm import tqdm

from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.model import load_model, generate_answer, generate_answer_with_context, generate_caption
from src.rag_retriever import RAGRetriever


def run_ablation(num_samples=100, top_k=3, alpha=1.0, caption_weight=0.3):
    print(f"Loading retriever...")
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()

    print(f"Loading BLIP-2 (4-bit)...")
    model, processor = load_model(quantize=True)

    kb_size = retriever.image_index.ntotal
    print(f"\nLoading {num_samples} eval samples (offset={5000})...")
    dataset = load_vqav2(split="validation", num_samples=num_samples, offset=5000)

    # Pre-compute captions and retrievals once
    cache = []
    print("\nPre-computing captions and retrievals (one pass)...")
    for item in tqdm(dataset, desc="Cache"):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]

        baseline = generate_answer(model, processor, image, question)
        caption  = generate_caption(model, processor, image)

        retr_no_cap = retriever.retrieve(image, question, top_k=top_k, alpha=alpha,
                                          caption=None, caption_weight=0.0)
        retr_cap    = retriever.retrieve(image, question, top_k=top_k, alpha=alpha,
                                          caption=caption, caption_weight=caption_weight)

        cache.append({
            "image":    image,
            "question": question,
            "gt_all":   gt_all,
            "baseline": baseline,
            "caption":  caption,
            "retr_no_cap": retr_no_cap,
            "retr_cap":    retr_cap,
        })

    # Run the 4 conditions
    results = {
        "baseline":              [],
        "rag_no_caption":        [],
        "rag_caption_in_prompt": [],
        "rag_caption_full":      [],
    }

    print("\nRunning 4 conditions...")
    for c in tqdm(cache, desc="Conditions"):
        # 1. Baseline
        results["baseline"].append(vqa_accuracy(c["baseline"], c["gt_all"]))

        # 2. RAG, no caption (no caption in prompt, no cap-aug retrieval)
        ans = generate_answer_with_context(
            model, processor, c["image"], c["question"], c["retr_no_cap"], caption=None
        )
        results["rag_no_caption"].append(vqa_accuracy(ans, c["gt_all"]))

        # 3. RAG + caption in prompt only (retrieval not caption-aware)
        ans = generate_answer_with_context(
            model, processor, c["image"], c["question"], c["retr_no_cap"], caption=c["caption"]
        )
        results["rag_caption_in_prompt"].append(vqa_accuracy(ans, c["gt_all"]))

        # 4. RAG + caption in prompt + caption-augmented retrieval
        ans = generate_answer_with_context(
            model, processor, c["image"], c["question"], c["retr_cap"], caption=c["caption"]
        )
        results["rag_caption_full"].append(vqa_accuracy(ans, c["gt_all"]))

    print("\n" + "=" * 60)
    print(f"Ablation results (n={num_samples}, k={top_k}, alpha={alpha}, cap_w={caption_weight}, KB={kb_size})")
    print("=" * 60)
    for name, scores in results.items():
        acc = sum(scores) / len(scores) * 100
        print(f"  {name:30s}  {acc:6.2f}%")

    # Save
    out = {
        "config": {"num_samples": num_samples, "top_k": top_k, "alpha": alpha,
                   "caption_weight": caption_weight, "kb_size": kb_size},
        "accuracies": {k: sum(v)/len(v) for k, v in results.items()},
    }
    out_path = Path("results/ablation_caption.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--top_k",       type=int, default=3)
    p.add_argument("--alpha",       type=float, default=1.0)
    p.add_argument("--caption_weight", type=float, default=0.3)
    args = p.parse_args()
    run_ablation(args.num_samples, args.top_k, args.alpha, args.caption_weight)
