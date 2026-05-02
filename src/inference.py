"""Run BLIP-2 inference on VQAv2 and evaluate."""

import argparse
import json
import time

import torch
from tqdm import tqdm

from src.config import RESULTS_DIR
from src.dataset import load_vqav2
from src.evaluate import evaluate_predictions, print_results, save_results
from src.model import generate_answer, load_model


def run_inference(num_samples=500, quantize=True, offset=0):
    """Run BLIP-2 inference on VQAv2 validation set.

    Args:
        num_samples: Number of samples to evaluate
        quantize: Whether to use 4-bit quantization
        offset: Number of samples to skip (for disjoint splits with RAG KB)

    Returns:
        Evaluation results dict
    """
    # Load model
    model, processor = load_model(quantize=quantize)

    # Load dataset
    print(f"\nLoading VQAv2 validation set ({num_samples} samples, offset={offset})...")
    dataset = load_vqav2(split="validation", num_samples=num_samples, offset=offset)
    print(f"Loaded {len(dataset)} samples")

    # Run inference
    predictions = []
    start_time = time.time()

    for i in tqdm(range(len(dataset)), desc="Running inference"):
        item = dataset[i]
        image = item["image"].convert("RGB")
        question = item["question"]
        question_id = item.get("question_id", i)

        answer = generate_answer(model, processor, image, question)

        predictions.append({
            "question_id": question_id,
            "answer": answer,
        })

        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(dataset)}] {rate:.1f} samples/sec")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(predictions)} samples in {elapsed:.1f}s "
          f"({len(predictions)/elapsed:.1f} samples/sec)")

    # Save raw predictions
    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_path}")

    # Evaluate
    results = evaluate_predictions(predictions, dataset)
    print_results(results)
    save_results(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLIP-2 VQA inference")
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="Number of validation samples to evaluate"
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Number of samples to skip (for disjoint splits with RAG KB)"
    )
    parser.add_argument(
        "--no_quantize", action="store_true",
        help="Disable 4-bit quantization (uses more VRAM)"
    )
    args = parser.parse_args()

    run_inference(
        num_samples=args.num_samples,
        quantize=not args.no_quantize,
        offset=args.offset,
    )
