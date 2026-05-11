"""RAG retriever: CLIP embeddings + FAISS index for VQA."""

import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import (
    CLIP_MODEL_NAME,
    RAG_ALPHA,
    RAG_CANDIDATE_K,
    RAG_INDEX_DIR,
    RAG_TAU,
    RAG_TOP_K,
)


class RAGRetriever:
    """Retrieves similar VQA examples using CLIP embeddings and FAISS."""

    def __init__(self, index_dir=None, device="cpu"):
        self.index_dir = Path(index_dir or RAG_INDEX_DIR)
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self.image_index = None
        self.question_index = None
        self.metadata = None

    def load_clip(self):
        """Load CLIP model and processor."""
        print(f"Loading CLIP ({CLIP_MODEL_NAME}) on {self.device}...")
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.clip_model.eval()
        print("CLIP loaded.")

    def unload_clip(self):
        """Free CLIP from memory."""
        del self.clip_model
        del self.clip_processor
        self.clip_model = None
        self.clip_processor = None
        torch.cuda.empty_cache()

    def encode_images(self, images, batch_size=32):
        """Encode PIL images with CLIP. Returns L2-normalized numpy array."""
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.clip_processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                emb = self.clip_model.get_image_features(pixel_values=pixel_values)
                if not isinstance(emb, torch.Tensor):
                    emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb[0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    def encode_questions(self, questions, batch_size=64):
        """Encode question strings with CLIP. Returns L2-normalized numpy array."""
        all_embs = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            inputs = self.clip_processor(
                text=batch, return_tensors="pt", padding=True, truncation=True,
                max_length=77,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            with torch.no_grad():
                emb = self.clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                if not isinstance(emb, torch.Tensor):
                    emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb[0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    def build_index(self, samples):
        """Build FAISS indices from VQAv2 samples.

        Args:
            samples: list of dicts with image, question, answers keys
        """
        from src.dataset import get_best_answer

        images = [s["image"].convert("RGB") for s in tqdm(samples, desc="Prep images")]
        questions = [s["question"] for s in samples]

        self.metadata = []
        for s in samples:
            self.metadata.append({
                "question":      s["question"],
                "best_answer":   get_best_answer(s["answers"]),
                "question_type": s.get("question_type", "unknown"),
                "image_id":      s.get("image_id"),
                "question_id":   s.get("question_id"),
                "all_answers":   [a["answer"] for a in s["answers"]],
            })

        print("Encoding images with CLIP...")
        image_embs = self.encode_images(images)
        print("Encoding questions with CLIP...")
        question_embs = self.encode_questions(questions)

        dim = image_embs.shape[1]
        self.image_index = faiss.IndexFlatIP(dim)
        self.image_index.add(image_embs)
        self.question_index = faiss.IndexFlatIP(dim)
        self.question_index.add(question_embs)

        print(f"Built indices: {len(samples)} entries, dim={dim}")

    def save_index(self):
        """Save FAISS indices and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.image_index, str(self.index_dir / "image.index"))
        faiss.write_index(self.question_index, str(self.index_dir / "question.index"))
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved indices to {self.index_dir}")

    def load_index(self):
        """Load FAISS indices and metadata from disk."""
        self.image_index = faiss.read_index(str(self.index_dir / "image.index"))
        self.question_index = faiss.read_index(str(self.index_dir / "question.index"))
        with open(self.index_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded indices: {self.image_index.ntotal} entries")

    def _load_cross_encoder(self):
        """Lazy-load the cross-encoder model for re-ranking."""
        if not hasattr(self, "_ce_model") or self._ce_model is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            print("Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
            self._ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder loaded.")
        return self._ce_model

    def retrieve(self, image, question, top_k=RAG_TOP_K, alpha=RAG_ALPHA,
                 candidate_k=RAG_CANDIDATE_K, rerank=False, rerank_weight=0.3,
                 caption=None, caption_weight=0.0):
        """Retrieve top-K similar examples using fused image+question similarity.

        Pipeline:
          1. CLIP bi-encoder retrieves top candidate_k by image+question (+caption) similarity
          2. (optional) Cross-encoder re-ranks candidates by (query_Q, retrieved_Q+A)
          3. Returns top_k after final scoring

        Score fusion:
            score = (1 - caption_weight) * (α * img_score + (1-α) * q_score)
                  + caption_weight * caption_text_vs_question_score

        Args:
            rerank: If True, apply cross-encoder re-ranking after CLIP retrieval
            rerank_weight: Weight of cross-encoder score in final combined score
            caption: Optional BLIP-2 caption string. If provided, used as a third
                     retrieval signal (CLIP-text encoded vs the question_index).
            caption_weight: Weight of caption-based retrieval in [0, 1]

        Returns:
            list of dicts: {question, best_answer, score, img_score, q_score, ce_score, cap_score}
        """
        img_emb = self.encode_images([image])
        q_emb = self.encode_questions([question])

        search_k = min(candidate_k, self.image_index.ntotal)
        img_scores, img_ids = self.image_index.search(img_emb, search_k)
        q_scores, q_ids = self.question_index.search(q_emb, search_k)

        img_score_map = defaultdict(float)
        q_score_map = defaultdict(float)
        cap_score_map = defaultdict(float)

        for j in range(search_k):
            img_score_map[int(img_ids[0][j])] = float(img_scores[0][j])
        for j in range(search_k):
            q_score_map[int(q_ids[0][j])] = float(q_scores[0][j])

        # Caption-augmented retrieval: encode caption as text, search question_index
        if caption and caption_weight > 0:
            cap_emb = self.encode_questions([caption])
            cap_scores, cap_ids = self.question_index.search(cap_emb, search_k)
            for j in range(search_k):
                cap_score_map[int(cap_ids[0][j])] = float(cap_scores[0][j])

        all_ids = set(img_score_map.keys()) | set(q_score_map.keys()) | set(cap_score_map.keys())

        # Fused CLIP score
        ranked = []
        for idx in all_ids:
            i_s = img_score_map.get(idx, 0.0)
            q_s = q_score_map.get(idx, 0.0)
            c_s = cap_score_map.get(idx, 0.0)
            base = alpha * i_s + (1 - alpha) * q_s
            fused = (1 - caption_weight) * base + caption_weight * c_s
            ranked.append((idx, fused, i_s, q_s, c_s))

        ranked.sort(key=lambda x: -x[1])

        # Cross-encoder re-ranking: score (query_question, retrieved_Q + " " + retrieved_A)
        if rerank and ranked:
            ce = self._load_cross_encoder()
            pairs = [
                (question, f"{self.metadata[idx]['question']} {self.metadata[idx]['best_answer']}")
                for idx, *_ in ranked
            ]
            ce_scores = ce.predict(pairs)
            ce_min, ce_max = ce_scores.min(), ce_scores.max()
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
            ce_norm = (ce_scores - ce_min) / ce_range

            ranked = [
                (idx, clip_s + rerank_weight * float(ce_n), i_s, q_s, c_s, float(ce_n))
                for (idx, clip_s, i_s, q_s, c_s), ce_n in zip(ranked, ce_norm)
            ]
            ranked.sort(key=lambda x: -x[1])
        else:
            ranked = [(idx, s, i_s, q_s, c_s, 0.0) for idx, s, i_s, q_s, c_s in ranked]

        results = []
        for row in ranked[:top_k]:
            idx, fused, i_s, q_s, c_s, ce_s = row
            entry = self.metadata[idx].copy()
            entry["score"]     = fused
            entry["img_score"] = i_s
            entry["q_score"]   = q_s
            entry["cap_score"] = c_s
            entry["ce_score"]  = ce_s
            results.append(entry)

        return results

    def should_use_rag(self, retrieved, tau=RAG_TAU):
        """Confidence gating: returns True if best score >= tau."""
        if not retrieved:
            return False
        return retrieved[0]["score"] >= tau
