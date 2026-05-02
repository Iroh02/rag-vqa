"""Build the RAG knowledge base from VQAv2 validation set.

Uses the first N samples from the validation split as the knowledge base.
Evaluation samples should use offset=N to avoid data leakage.
"""

import argparse
import time

from src.config import RAG_KB_SIZE
from src.dataset import load_vqav2
from src.rag_retriever import RAGRetriever


def build_knowledge_base(num_samples=RAG_KB_SIZE):
    """Build and save the FAISS knowledge base."""
    print(f"Building RAG knowledge base ({num_samples} samples from validation)...")
    print("(First N validation samples = KB, evaluation uses offset=N)")
    start = time.time()

    samples = load_vqav2(split="validation", num_samples=num_samples, offset=0)

    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.build_index(samples)
    retriever.save_index()
    retriever.unload_clip()

    elapsed = time.time() - start
    print(f"Knowledge base built in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG knowledge base")
    parser.add_argument("--num_samples", type=int, default=RAG_KB_SIZE)
    args = parser.parse_args()
    build_knowledge_base(num_samples=args.num_samples)
