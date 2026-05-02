"""BLIP-2 model loading and inference."""

import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.config import DEVICE, MODEL_NAME


def load_model(quantize=True):
    """Load BLIP-2 model and processor.

    Args:
        quantize: If True, load in 4-bit quantization (saves VRAM)

    Returns:
        (model, processor) tuple
    """
    print(f"Loading processor from {MODEL_NAME}...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)

    print(f"Loading model from {MODEL_NAME}...")
    if quantize:
        from transformers import BitsAndBytesConfig

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
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
    return model, processor


def generate_answer(model, processor, image, question, max_new_tokens=20):
    """Generate an answer for a single image-question pair.

    Args:
        model: BLIP-2 model
        processor: BLIP-2 processor
        image: PIL Image
        question: Question string
        max_new_tokens: Max tokens to generate

    Returns:
        Generated answer string
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device, torch.float16
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=5,
            early_stopping=True,
        )

    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Remove prompt prefix if model echoes it back
    for prefix in [f"Question: {question} Answer:", "Answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    answer = _postprocess_answer(answer, question)
    return answer


def _postprocess_answer(answer, question):
    """Clean up model output to match VQAv2 short-answer format."""
    # Strip trailing punctuation
    answer = answer.rstrip(".!,")

    # Detect if this is a yes/no question
    q_lower = question.lower().strip()
    is_yesno = any(q_lower.startswith(w) for w in [
        "is ", "are ", "do ", "does ", "can ", "has ", "have ",
        "was ", "were ", "will ", "could ", "should ", "would ",
    ])

    lower = answer.lower().strip()

    if is_yesno:
        # For yes/no questions, extract the core yes/no
        first_word = lower.split()[0].rstrip(".,!") if lower.split() else ""
        if first_word == "yes":
            return "yes"
        if first_word == "no":
            return "no"

    # Remove common verbose prefixes
    for prefix in ["it is ", "it's ", "they are ", "there are ", "there is ",
                    "he is ", "she is ", "he's ", "she's ", "they're "]:
        if lower.startswith(prefix):
            answer = answer[len(prefix):]
            break

    # Remove leading "a ", "an ", "the "
    for article in ["a ", "an ", "the "]:
        if answer.lower().startswith(article):
            answer = answer[len(article):]
            break

    return answer.strip()


def build_hints_prompt(question, retrieved_examples):
    """Build the Hints-style prompt for RAG-VQA.

    This format is used for both fine-tuning and inference.
    """
    lines = ["Answer with 1-3 words. Hints may be wrong."]
    if retrieved_examples:
        lines.append("Hints:")
        for ex in retrieved_examples:
            answer = ex.get("best_answer") or ex.get("answer", "")
            lines.append(f"- Similar image suggests: {answer}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _clean_hints_output(raw_answer):
    """Remove repetition artifacts from fine-tuned model output.

    The fine-tuned model sometimes generates repeated tokens like
    'noyesnonoyesno' or 'cloudycloudyyes'. We extract the first clean phrase.
    """
    # Split into words and deduplicate consecutive repeats
    words = raw_answer.split()
    if not words:
        return raw_answer

    deduped = [words[0]]
    for w in words[1:]:
        if w.lower() != deduped[-1].lower():
            deduped.append(w)

    # Cap at 3 words
    clean = " ".join(deduped[:3])

    # If still looks like a run-on (no spaces, > 10 chars), take first 'word'
    # by splitting at the point where a repeated pattern starts
    if " " not in raw_answer and len(raw_answer) > 10:
        # Try to find a short prefix that repeats: e.g. "nono..." -> "no"
        for length in range(2, 8):
            prefix = raw_answer[:length]
            if raw_answer.startswith(prefix * 2):
                return prefix
        # Fallback: just take first 6 chars trimmed to a reasonable length
        clean = raw_answer[:6]

    return clean.strip()


def generate_answer_with_hints(model, processor, image, question,
                                retrieved_examples, max_new_tokens=7):
    """Generate answer using Hints-style RAG prompt.

    Use with fine-tuned models trained on hints-format prompts.
    For pre-trained models, use generate_answer_with_context instead.
    """
    prompt = build_hints_prompt(question, retrieved_examples)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device, torch.float16
    )

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = generated_ids[0, prompt_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip any echoed "Answer:" prefix
    for prefix in ["Answer:", "answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # Stop at newline (model sometimes continues into next line)
    answer = answer.split("\n")[0].strip()

    # Remove repetition artifacts
    answer = _clean_hints_output(answer)

    answer = _postprocess_answer(answer, question)
    return answer


def load_finetuned_model(checkpoint_path, quantize=True):
    """Load BLIP-2 base model + LoRA adapter from checkpoint.

    Args:
        checkpoint_path: Path to saved LoRA adapter directory
        quantize: Whether to use 4-bit quantization

    Returns:
        (model, processor) tuple
    """
    from peft import PeftModel

    model, processor = load_model(quantize=quantize)
    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    print("Fine-tuned model loaded.")
    return model, processor


def generate_answer_with_context(model, processor, image, question,
                                  retrieved_examples, max_new_tokens=10):
    """Generate answer using retrieved examples as few-shot context.

    Args:
        model: BLIP-2 model
        processor: BLIP-2 processor
        image: PIL Image
        question: question string
        retrieved_examples: list of dicts with 'question' and 'best_answer'
        max_new_tokens: max tokens to generate

    Returns:
        Generated answer string
    """
    # Build few-shot context
    context_lines = []
    for ex in retrieved_examples:
        context_lines.append(f"Q: {ex['question']} A: {ex['best_answer']}")
    context_lines.append(f"Q: {question} A:")
    prompt = "\n".join(context_lines)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device, torch.float16
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            early_stopping=True,
        )

    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Remove prompt echo
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    last_line = f"Q: {question} A:"
    if answer.startswith(last_line):
        answer = answer[len(last_line):].strip()
    for prefix in ["A:", "a:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    answer = _postprocess_answer(answer, question)
    return answer


def batch_generate(model, processor, images, questions, max_new_tokens=20):
    """Generate answers for a batch of image-question pairs.

    Args:
        model: BLIP-2 model
        processor: BLIP-2 processor
        images: List of PIL Images
        questions: List of question strings
        max_new_tokens: Max tokens to generate

    Returns:
        List of answer strings
    """
    prompts = [f"Question: {q} Answer:" for q in questions]
    inputs = processor(
        images=images, text=prompts, return_tensors="pt", padding=True
    ).to(model.device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            early_stopping=True,
        )

    answers = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [a.strip() for a in answers]


if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    # Quick test with a sample image
    print("Testing BLIP-2 inference...")
    model, processor = load_model(quantize=True)

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    questions = [
        "What do you see in this image?",
        "What colors are present?",
    ]

    for q in questions:
        answer = generate_answer(model, processor, image, q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print()
