"""VQA evaluation metrics."""

import json
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.config import RESULTS_DIR


def normalize_answer(answer):
    """Normalize answer string for comparison.

    Follows the VQAv2 evaluation protocol:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    """
    answer = str(answer).lower().strip()

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Remove extra whitespace
    answer = " ".join(answer.split())

    return answer


def vqa_accuracy(predicted, ground_truth_answers):
    """Compute VQA accuracy for a single prediction.

    Uses the official VQAv2 metric:
        accuracy = min(1, count_of_matching_answers / 3)

    Args:
        predicted: Predicted answer string
        ground_truth_answers: List of 10 human-provided answer strings

    Returns:
        Float accuracy score between 0 and 1
    """
    predicted_norm = normalize_answer(predicted)
    gt_normalized = [normalize_answer(a) for a in ground_truth_answers]

    matching = sum(1 for gt in gt_normalized if gt == predicted_norm)
    return min(1.0, matching / 3.0)


def evaluate_predictions(predictions, dataset):
    """Evaluate a list of predictions against the dataset.

    Args:
        predictions: List of dicts with "question_id" and "answer"
        dataset: HuggingFace VQAv2 dataset

    Returns:
        Dict with overall accuracy and per-question-type accuracy
    """
    # Build lookup from question_id to ground truth
    gt_lookup = {}
    for item in dataset:
        qid = item.get("question_id", None)
        if qid is not None:
            gt_lookup[qid] = {
                "answers": [a["answer"] for a in item["answers"]],
                "question": item["question"],
                "question_type": item.get("question_type", "unknown"),
            }

    total_accuracy = 0.0
    per_type_accuracy = defaultdict(list)
    results = []

    for pred in predictions:
        qid = pred["question_id"]
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]
        acc = vqa_accuracy(pred["answer"], gt["answers"])
        total_accuracy += acc

        qtype = gt["question_type"]
        per_type_accuracy[qtype].append(acc)

        results.append({
            "question_id": qid,
            "question": gt["question"],
            "predicted": pred["answer"],
            "ground_truth": gt["answers"][:3],
            "accuracy": acc,
            "question_type": qtype,
        })

    n = len(results)
    overall = total_accuracy / n if n > 0 else 0.0

    # Per question type
    type_results = {}
    for qtype, accs in sorted(per_type_accuracy.items()):
        type_results[qtype] = {
            "accuracy": sum(accs) / len(accs),
            "count": len(accs),
        }

    return {
        "overall_accuracy": overall,
        "num_evaluated": n,
        "per_type_accuracy": type_results,
        "detailed_results": results,
    }


def save_results(results, filename="eval_results.json"):
    """Save evaluation results to JSON."""
    output_path = RESULTS_DIR / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return output_path


def print_results(results):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 60)
    print("VQA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} "
          f"({results['overall_accuracy']*100:.2f}%)")
    print(f"Samples Evaluated: {results['num_evaluated']}")

    if results["per_type_accuracy"]:
        print(f"\nPer Question Type:")
        print("-" * 45)
        for qtype, stats in sorted(
            results["per_type_accuracy"].items(),
            key=lambda x: -x[1]["accuracy"],
        ):
            print(f"  {qtype:<25} {stats['accuracy']*100:6.2f}%  (n={stats['count']})")

    # Show some examples
    print(f"\nSample Predictions:")
    print("-" * 60)
    for r in results["detailed_results"][:10]:
        status = "correct" if r["accuracy"] >= 1.0 else "wrong"
        print(f"  Q: {r['question']}")
        print(f"  Predicted: {r['predicted']} | GT: {r['ground_truth']} [{status}]")
        print()


if __name__ == "__main__":
    # Quick test of the evaluation logic
    print("Testing VQA evaluation metric...")

    # Test cases
    test_cases = [
        ("yes", ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes"]),
        ("cat", ["cat", "cat", "cat", "dog", "dog", "dog", "dog", "dog", "dog", "dog"]),
        ("blue", ["red", "red", "red", "red", "red", "red", "red", "red", "red", "red"]),
        ("2", ["2", "two", "2", "2", "two", "2", "2", "2", "two", "2"]),
    ]

    for pred, gts in test_cases:
        acc = vqa_accuracy(pred, gts)
        print(f"  Predicted: '{pred}' | GTs: {gts[:3]}... | Accuracy: {acc:.4f}")
