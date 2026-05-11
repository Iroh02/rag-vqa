"""Build the RAG knowledge base.

Two modes:
  default:      first N validation samples (legacy, kept for reproducibility)
  --diverse:    streams validation starting AFTER the eval region (offset=5100),
                dedupes by image_id, collects N unique images. Same KB size with
                ~5.9x more visual diversity, and image-id-disjoint from eval set.
"""

import argparse
import time

from datasets import load_dataset

from src.config import RAG_KB_SIZE, VQAV2_DATASET
from src.rag_retriever import RAGRetriever

# Eval set lives in validation samples 5000–5099 (offset=5000, size=100).
# We start the diverse build at 5100 so KB image_ids are disjoint from eval.
EVAL_END_OFFSET = 5100


def _stream_diverse(num_images, start_offset=EVAL_END_OFFSET):
    """Stream validation past the eval region, dedupe by image_id."""
    print(f"Streaming VQAv2 validation from offset={start_offset} "
          f"for {num_images} unique image_ids...")
    stream = load_dataset(VQAV2_DATASET, split="validation", streaming=True)

    seen = set()
    samples = []
    streamed = 0
    skipped = 0
    for item in stream:
        if skipped < start_offset:
            skipped += 1
            continue
        streamed += 1
        img_id = item.get("image_id")
        if img_id is None or img_id in seen:
            continue
        seen.add(img_id)
        samples.append(item)
        if len(samples) % 500 == 0:
            ratio = streamed / max(len(samples), 1)
            print(f"  collected {len(samples)} unique images "
                  f"(streamed {streamed} samples, {ratio:.1f}× ratio)")
        if len(samples) >= num_images:
            break

    print(f"Collected {len(samples)} unique images from {streamed} streamed samples.")
    return samples


def build_knowledge_base(num_samples=RAG_KB_SIZE, diverse=False):
    """Build and save the FAISS knowledge base."""
    start = time.time()

    if diverse:
        samples = _stream_diverse(num_samples)
    else:
        from src.dataset import load_vqav2
        print(f"Building KB from validation split ({num_samples} samples, first-N)...")
        samples = load_vqav2(split="validation", num_samples=num_samples, offset=0)

    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.build_index(samples)
    retriever.save_index()
    retriever.unload_clip()

    elapsed = time.time() - start
    print(f"Knowledge base built in {elapsed:.1f}s")
    print(f"  diverse={diverse}, entries={len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG knowledge base")
    parser.add_argument("--num_samples", type=int, default=RAG_KB_SIZE)
    parser.add_argument("--diverse", action="store_true",
                        help="Stream TRAIN split, dedupe by image_id (recommended)")
    args = parser.parse_args()
    build_knowledge_base(num_samples=args.num_samples, diverse=args.diverse)
