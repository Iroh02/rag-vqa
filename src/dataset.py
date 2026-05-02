"""VQAv2 dataset loading and preprocessing."""

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import CACHE_DIR, VQAV2_DATASET


def load_vqav2(split="validation", num_samples=None, offset=0):
    """Load VQAv2 dataset from HuggingFace using streaming.

    Uses streaming to avoid downloading the entire dataset (~20GB).
    Materializes only the requested number of samples into memory.

    Args:
        split: "validation", "testdev", or "test"
        num_samples: Number of samples to load
        offset: Number of samples to skip before loading (for disjoint splits)

    Returns:
        List of dataset items (dicts with image, question, answers, etc.)
    """
    print(f"Streaming VQAv2 {split} set (offset={offset})...")
    stream = load_dataset(VQAV2_DATASET, split=split, streaming=True)

    samples = []
    target = (offset + num_samples) if num_samples is not None else None
    for i, item in enumerate(tqdm(stream, total=target, desc="Loading samples")):
        if target is not None and i >= target:
            break
        if i < offset:
            continue
        samples.append(item)

    print(f"Loaded {len(samples)} samples")
    return samples


def get_best_answer(answers):
    """Get the most common answer from a list of annotator answers."""
    answer_counts = {}
    for ans in answers:
        text = ans["answer"]
        answer_counts[text] = answer_counts.get(text, 0) + 1
    return max(answer_counts, key=answer_counts.get)


class VQADataset(Dataset):
    """PyTorch dataset wrapper for VQAv2."""

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = item["image"].convert("RGB")
        question = item["question"]
        best_answer = get_best_answer(item["answers"])

        # Process inputs for BLIP-2
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return {
            "inputs": inputs,
            "question": question,
            "answer": best_answer,
            "all_answers": [a["answer"] for a in item["answers"]],
            "image": image,
            "question_id": item.get("question_id", idx),
        }


if __name__ == "__main__":
    # Quick test: load a small subset
    print("Loading VQAv2 validation set (10 samples)...")
    samples = load_vqav2(split="validation", num_samples=10)
    print(f"\nSample 0:")
    print(f"  Question: {samples[0]['question']}")
    print(f"  Answers: {[a['answer'] for a in samples[0]['answers']]}")
    print(f"  Best answer: {get_best_answer(samples[0]['answers'])}")
    print(f"  Question type: {samples[0].get('question_type', 'N/A')}")
    print(f"  Image size: {samples[0]['image'].size}")
