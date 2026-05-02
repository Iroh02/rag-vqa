"""Visualize VQA predictions — great for showing results at midterm."""

import json
import random

import matplotlib.pyplot as plt
from PIL import Image

from src.config import RESULTS_DIR
from src.dataset import load_vqav2
from src.evaluate import normalize_answer


def visualize_predictions(dataset, predictions, num_samples=8, save_path=None):
    """Show a grid of images with questions, predicted answers, and ground truth.

    Args:
        dataset: HuggingFace VQAv2 dataset
        predictions: List of dicts with "question_id" and "answer"
        num_samples: Number of samples to show
        save_path: If set, save figure to this path
    """
    # Build prediction lookup
    pred_lookup = {p["question_id"]: p["answer"] for p in predictions}

    # Collect samples that have predictions
    samples = []
    for item in dataset:
        qid = item.get("question_id")
        if qid in pred_lookup:
            gt_answers = [a["answer"] for a in item["answers"]]
            pred = pred_lookup[qid]
            pred_norm = normalize_answer(pred)
            gt_norm = [normalize_answer(a) for a in gt_answers]
            is_correct = pred_norm in gt_norm
            samples.append({
                "image": item["image"],
                "question": item["question"],
                "predicted": pred,
                "ground_truth": gt_answers[:3],
                "is_correct": is_correct,
            })

    # Select a mix of correct and incorrect
    correct = [s for s in samples if s["is_correct"]]
    incorrect = [s for s in samples if not s["is_correct"]]

    n_correct = min(num_samples // 2, len(correct))
    n_incorrect = min(num_samples - n_correct, len(incorrect))
    n_correct = min(num_samples - n_incorrect, len(correct))

    selected = (random.sample(correct, n_correct) +
                random.sample(incorrect, n_incorrect))
    random.shuffle(selected)

    # Plot
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, "__len__") else [row])]

    for i, ax in enumerate(axes):
        if i < len(selected):
            s = selected[i]
            ax.imshow(s["image"])
            color = "green" if s["is_correct"] else "red"
            ax.set_title(
                f"Q: {s['question']}\n"
                f"Pred: {s['predicted']}\n"
                f"GT: {', '.join(s['ground_truth'])}",
                fontsize=9,
                color=color,
                wrap=True,
            )
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.savefig(RESULTS_DIR / "predictions_visualization.png",
                    dpi=150, bbox_inches="tight")
        print(f"Figure saved to {RESULTS_DIR / 'predictions_visualization.png'}")

    plt.show()


def plot_accuracy_by_type(results, save_path=None):
    """Bar chart of accuracy by question type.

    Args:
        results: Output from evaluate_predictions()
        save_path: If set, save figure to this path
    """
    type_acc = results.get("per_type_accuracy", {})
    if not type_acc:
        print("No per-type accuracy data available")
        return

    # Sort by accuracy descending, take top 15
    sorted_types = sorted(type_acc.items(), key=lambda x: -x[1]["accuracy"])[:15]
    names = [t[0] for t in sorted_types]
    accs = [t[1]["accuracy"] * 100 for t in sorted_types]
    counts = [t[1]["count"] for t in sorted_types]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), accs, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(names, counts)], fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("VQA Accuracy by Question Type (BLIP-2 Baseline)")
    ax.invert_yaxis()

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=8)

    plt.tight_layout()

    path = save_path or (RESULTS_DIR / "accuracy_by_type.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}")
    plt.show()


if __name__ == "__main__":
    # Load saved results and visualize
    results_path = RESULTS_DIR / "eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        plot_accuracy_by_type(results)
    else:
        print(f"No results found at {results_path}")
        print("Run inference.py first to generate predictions")
