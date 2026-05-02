"""Fine-tune BLIP-2 on VQAv2 using QLoRA.

Supports two modes:
  1. Raw VQAv2: python run.py train --num_samples 5000
  2. RAG JSONL: python run.py rag-finetune --train_jsonl data/rag_train.jsonl
"""

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
)

from src.config import (
    CHECKPOINTS_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
)
from src.dataset import load_vqav2


# --- RAG fine-tuning defaults (8GB VRAM safe) ---
RAG_BATCH_SIZE = 2
RAG_GRAD_ACCUM = 8
RAG_LR = 1e-4
RAG_MAX_LENGTH = 128


def setup_model_for_training():
    """Load BLIP-2 with QLoRA configuration for fine-tuning.

    Returns:
        (model, processor) tuple with LoRA adapters applied
    """
    print("Loading processor...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)

    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA — target the language model's attention layers
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # Attention projection layers in OPT
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return model, processor


# ---- Dataset classes ----

class SimpleVQADataset:
    """Wraps raw VQAv2 samples for training."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        answers = item["answers"]
        answer_counts = {}
        for ans in answers:
            text = ans["answer"]
            answer_counts[text] = answer_counts.get(text, 0) + 1
        best_answer = max(answer_counts, key=answer_counts.get)
        return {
            "image": item["image"],
            "question": item["question"],
            "prompt": f"Question: {item['question']} Answer:",
            "answer": best_answer,
            "all_answers": [a["answer"] for a in answers],
        }


class RAGTrainDataset:
    """Pairs JSONL prompts with VQAv2 images for RAG fine-tuning."""

    def __init__(self, jsonl_path, vqa_samples):
        self.vqa_samples = vqa_samples
        self.entries = []
        self.meta = {}

        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("_meta"):
                    self.meta = entry
                    continue
                self.entries.append(entry)

        if len(self.entries) != len(self.vqa_samples):
            raise ValueError(
                f"JSONL has {len(self.entries)} entries but VQA has "
                f"{len(self.vqa_samples)} samples. Use matching offset/num_samples."
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = self.vqa_samples[idx]["image"].convert("RGB")
        return {
            "image": image,
            "prompt": entry["prompt"],
            "answer": entry["answer"],
            "all_answers": entry.get("all_answers", [entry["answer"]]),
            "question_id": entry.get("question_id", idx),
        }


# ---- Collate functions ----

def collate_fn(batch, processor, max_length=MAX_LENGTH):
    """Collate for both raw VQAv2 and JSONL training."""
    images = [item["image"].convert("RGB") if hasattr(item["image"], "convert")
              else item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]

    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = processor.tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )

    inputs["labels"] = labels["input_ids"]
    return inputs


# ---- Evaluation ----

def evaluate_epoch(model, processor, eval_dataset, max_samples=100):
    """Quick VQA accuracy eval using pre-built prompts.

    Args:
        model: Fine-tuned BLIP-2 model (switched to eval mode internally)
        processor: BLIP-2 processor
        eval_dataset: RAGTrainDataset or SimpleVQADataset with all_answers
        max_samples: Max samples to evaluate

    Returns:
        Float accuracy (0-1)
    """
    from src.evaluate import vqa_accuracy
    from src.model import _postprocess_answer

    model.eval()
    total_acc = 0.0
    n = min(len(eval_dataset), max_samples)

    for i in tqdm(range(n), desc="Eval", leave=False):
        item = eval_dataset[i]
        image = item["image"]
        prompt = item["prompt"]

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            model.device, torch.float16
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=10, num_beams=5, early_stopping=True,
            )

        answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Remove prompt echo
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        for pfx in ["Answer:", "answer:"]:
            if answer.startswith(pfx):
                answer = answer[len(pfx):].strip()

        # Extract question from prompt for postprocessing
        q_lines = [l for l in prompt.split("\n") if l.startswith("Question:")]
        question = (q_lines[-1].replace("Question:", "").replace("Answer:", "").strip()
                    if q_lines else "")
        answer = _postprocess_answer(answer, question)

        acc = vqa_accuracy(answer, item["all_answers"])
        total_acc += acc

    model.train()
    return total_acc / n if n > 0 else 0.0


# ---- Training functions ----

def train(num_train_samples=5000, num_epochs=NUM_EPOCHS):
    """Run QLoRA fine-tuning on raw VQAv2 (original behavior)."""
    model, processor = setup_model_for_training()

    print(f"\nLoading VQAv2 training set ({num_train_samples} samples)...")
    train_data = load_vqav2(split="train", num_samples=num_train_samples)
    train_dataset = SimpleVQADataset(train_data)

    _run_training_loop(model, processor, train_dataset, num_epochs=num_epochs)
    return model, processor


def train_rag(train_jsonl, eval_jsonl=None, num_epochs=NUM_EPOCHS,
              batch_size=RAG_BATCH_SIZE, grad_accum=RAG_GRAD_ACCUM,
              lr=RAG_LR):
    """Run QLoRA fine-tuning on RAG-augmented JSONL prompts.

    Args:
        train_jsonl: Path to training JSONL (from rag_make_train_jsonl.py)
        eval_jsonl: Optional path to eval JSONL (for per-epoch eval)
        num_epochs: Number of training epochs
        batch_size: Batch size (default 2 for 8GB VRAM)
        grad_accum: Gradient accumulation steps
        lr: Learning rate
    """
    model, processor = setup_model_for_training()

    # Load training data: JSONL for prompts, VQAv2 for images
    print(f"\nLoading training JSONL: {train_jsonl}")
    train_meta = _read_jsonl_meta(train_jsonl)
    train_offset = train_meta["offset"]
    train_n = train_meta["num_samples"]

    print(f"Loading VQAv2 images (offset={train_offset}, n={train_n})...")
    train_vqa = load_vqav2(split="validation", num_samples=train_n, offset=train_offset)
    train_dataset = RAGTrainDataset(train_jsonl, train_vqa)

    # Load eval data if provided
    eval_dataset = None
    if eval_jsonl:
        print(f"\nLoading eval JSONL: {eval_jsonl}")
        eval_meta = _read_jsonl_meta(eval_jsonl)
        eval_offset = eval_meta["offset"]
        eval_n = eval_meta["num_samples"]
        print(f"Loading VQAv2 eval images (offset={eval_offset}, n={eval_n})...")
        eval_vqa = load_vqav2(split="validation", num_samples=eval_n, offset=eval_offset)
        eval_dataset = RAGTrainDataset(eval_jsonl, eval_vqa)

    _run_training_loop(
        model, processor, train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        checkpoint_prefix="rag_checkpoint",
    )
    return model, processor


def _read_jsonl_meta(jsonl_path):
    """Read metadata from first line of JSONL."""
    with open(jsonl_path) as f:
        meta = json.loads(f.readline())
    if not meta.get("_meta"):
        raise ValueError(f"First line of {jsonl_path} is not metadata")
    return meta


def _run_training_loop(model, processor, train_dataset, eval_dataset=None,
                       num_epochs=NUM_EPOCHS, batch_size=RAG_BATCH_SIZE,
                       grad_accum=RAG_GRAD_ACCUM, lr=RAG_LR,
                       checkpoint_prefix="checkpoint"):
    """Shared training loop for both raw VQAv2 and RAG JSONL training."""
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor, max_length=RAG_MAX_LENGTH),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nStarting training:")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    if eval_dataset:
        print(f"  Eval samples: {len(eval_dataset)}")
    print()

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(model.device) if hasattr(v, "to") else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{epoch_loss / num_batches:.4f}"})

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, "
              f"time={epoch_time:.1f}s")

        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{checkpoint_prefix}_epoch_{epoch+1}"
        model.save_pretrained(str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")

        # Eval hook
        if eval_dataset:
            acc = evaluate_epoch(model, processor, eval_dataset)
            print(f"Epoch {epoch+1} eval accuracy: {acc*100:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 with QLoRA")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train_jsonl", type=str, default=None,
                        help="Path to RAG training JSONL (enables RAG fine-tuning)")
    parser.add_argument("--eval_jsonl", type=str, default=None,
                        help="Path to RAG eval JSONL (enables per-epoch eval)")
    parser.add_argument("--batch_size", type=int, default=RAG_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=RAG_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=RAG_LR)
    args = parser.parse_args()

    if args.train_jsonl:
        train_rag(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )
    else:
        train(num_train_samples=args.num_samples, num_epochs=args.epochs)
