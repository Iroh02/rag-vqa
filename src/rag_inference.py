"""Run RAG-augmented BLIP-2 inference on VQAv2."""

import argparse
import json
import time

from tqdm import tqdm

from src.config import (
    RAG_ALPHA,
    RAG_CANDIDATE_K,
    RAG_TAU,
    RAG_TOP_K,
    RESULTS_DIR,
)
from src.dataset import load_vqav2
from src.evaluate import evaluate_predictions, print_results, save_results
from src.model import (
    generate_answer,
    generate_answer_with_context,
    generate_answer_with_hints,
    load_finetuned_model,
    load_model,
)
from src.rag_retriever import RAGRetriever


def run_rag_inference(num_samples=500, top_k=RAG_TOP_K, alpha=RAG_ALPHA,
                      candidate_k=RAG_CANDIDATE_K, tau=RAG_TAU, quantize=True,
                      checkpoint=None):
    """Run RAG-augmented inference on VQAv2 validation set.

    Args:
        checkpoint: Path to LoRA checkpoint. When provided, uses hints prompt
                    format instead of Q/A few-shot format.

    Outputs:
        - predictions.json: baseline-compatible [{question_id, answer}]
        - debug_context.json: per-question retrieved examples and scores
    """
    use_hints = checkpoint is not None

    # Load retriever (CLIP on CPU + FAISS indices)
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()

    # Load BLIP-2 (4-bit on GPU), with optional LoRA adapter
    if checkpoint:
        model, processor = load_finetuned_model(checkpoint, quantize=quantize)
    else:
        model, processor = load_model(quantize=quantize)

    # Load validation data (offset by KB size to avoid data leakage)
    kb_size = retriever.image_index.ntotal
    print(f"\nLoading VQAv2 validation set ({num_samples} samples, offset={kb_size})...")
    dataset = load_vqav2(split="validation", num_samples=num_samples, offset=kb_size)

    predictions = []
    debug_context = []
    rag_used_count = 0
    start_time = time.time()

    for i in tqdm(range(len(dataset)), desc="RAG inference"):
        item = dataset[i]
        image = item["image"].convert("RGB")
        question = item["question"]
        question_id = item.get("question_id", i)

        # Retrieve
        retrieved = retriever.retrieve(
            image, question, top_k=top_k, alpha=alpha, candidate_k=candidate_k,
        )
        use_rag = retriever.should_use_rag(retrieved, tau=tau)

        # Generate
        if use_rag:
            if use_hints:
                answer = generate_answer_with_hints(
                    model, processor, image, question, retrieved,
                )
            else:
                answer = generate_answer_with_context(
                    model, processor, image, question, retrieved,
                )
            rag_used_count += 1
        else:
            answer = generate_answer(model, processor, image, question)

        # Save prediction in baseline-compatible format
        predictions.append({
            "question_id": question_id,
            "answer": answer,
        })

        # Save debug info
        debug_context.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "rag_used": use_rag,
            "retrieved": [
                {
                    "question": r["question"],
                    "answer": r["best_answer"],
                    "score": round(r["score"], 4),
                    "img_score": round(r["img_score"], 4),
                    "q_score": round(r["q_score"], 4),
                }
                for r in retrieved
            ],
        })

    elapsed = time.time() - start_time
    rag_pct = 100 * rag_used_count / len(dataset) if dataset else 0
    print(f"\nRAG inference complete: {len(predictions)} samples in {elapsed:.1f}s")
    print(f"RAG used: {rag_used_count}/{len(dataset)} ({rag_pct:.1f}%)")
    print(f"Config: K={top_k}, alpha={alpha}, candidate_k={candidate_k}, tau={tau}")

    # Save predictions (baseline-compatible format)
    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_path}")

    # Save debug context
    debug_path = RESULTS_DIR / "debug_context.json"
    with open(debug_path, "w") as f:
        json.dump(debug_context, f, indent=2)
    print(f"Debug context saved to {debug_path}")

    # Evaluate
    results = evaluate_predictions(predictions, dataset)
    results["rag_config"] = {
        "top_k": top_k, "alpha": alpha, "candidate_k": candidate_k,
        "tau": tau, "kb_size": retriever.image_index.ntotal,
        "rag_used_pct": rag_pct,
    }
    print_results(results)
    save_results(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG-VQA inference")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=RAG_TOP_K)
    parser.add_argument("--alpha", type=float, default=RAG_ALPHA)
    parser.add_argument("--candidate_k", type=int, default=RAG_CANDIDATE_K)
    parser.add_argument("--tau", type=float, default=RAG_TAU)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint path (uses hints prompt format)")
    args = parser.parse_args()

    run_rag_inference(
        num_samples=args.num_samples,
        top_k=args.top_k,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        tau=args.tau,
        quantize=not args.no_quantize,
        checkpoint=args.checkpoint,
    )
