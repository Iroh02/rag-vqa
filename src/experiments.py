"""
Ablation experiments for RAG-VQA.

Run individual experiments:
    python -m src.experiments --exp random_baseline
    python -m src.experiments --exp oracle
    python -m src.experiments --exp perfect_hint
    python -m src.experiments --exp kb_ablation
    python -m src.experiments --exp topk_ablation
    python -m src.experiments --exp answer_type
    python -m src.experiments --exp helpful_harmful
    python -m src.experiments --exp all

Run everything:
    python -m src.experiments --exp all
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    RAG_INDEX_DIR,
    RESULTS_DIR,
)
from src.dataset import load_vqav2, get_best_answer
from src.evaluate import normalize_answer, vqa_accuracy
from src.model import (
    generate_answer,
    generate_answer_with_context,
    load_model,
)
from src.rag_retriever import RAGRetriever

# ── Output directories ────────────────────────────────────────────────────────
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
PLOTS_DIR = RESULTS_DIR / "plots"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared constants ──────────────────────────────────────────────────────────
# IMPORTANT: Retrieval KB and eval set must be image-disjoint.
# Eval uses validation split; KB is built from the train split by default.
# Explicit image_id exclusion is applied inside build_kb — do not rely on
# offset arithmetic alone to guarantee disjointness.
EVAL_SAMPLES = 100      # matches published baseline/RAG numbers — do not change default
EVAL_OFFSET  = 0        # start of eval window in validation split


# =============================================================================
# 1. BUILD IMAGE-DISJOINT KNOWLEDGE BASE
# =============================================================================

def build_kb(num_samples: int,
             split: str = "train",
             eval_image_ids: set = None,
             index_dir: Path = None) -> Path:
    """Build a FAISS knowledge base that is explicitly image-disjoint from the eval set.

    Prefers the train split (different split entirely from validation eval).
    If eval_image_ids is provided, any sample whose image_id appears in that
    set is removed before indexing — this is the primary leakage guard, not
    offset arithmetic.

    Args:
        num_samples:    Desired KB size after filtering.
        split:          Dataset split to draw from. Default "train" keeps a
                        hard split boundary from the validation eval set.
        eval_image_ids: Set of image_id values in the eval set. Samples
                        sharing an image_id with eval are excluded.
        index_dir:      Save path (default: data/rag_index/kb_{num_samples}).

    Returns:
        Path to the saved index directory.
    """
    if index_dir is None:
        index_dir = DATA_DIR / "rag_index" / f"kb_{num_samples}"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load generously so we still have num_samples after filtering
    load_n = num_samples * 2 if eval_image_ids else num_samples
    print(f"\n[build_kb] Loading {load_n} samples from {split} split...")
    candidates = load_vqav2(split=split, num_samples=load_n, offset=0)

    if eval_image_ids:
        before = len(candidates)
        candidates = [s for s in candidates
                      if str(s.get("image_id", "")) not in eval_image_ids]
        print(f"[build_kb] Removed {before - len(candidates)} samples that "
              f"share image_id with eval set.")

    if len(candidates) < num_samples:
        print(f"[build_kb] WARNING: only {len(candidates)} samples available "
              f"after filtering (wanted {num_samples}). Using all available.")
    samples = candidates[:num_samples]

    print(f"[build_kb] Indexing {len(samples)} samples → {index_dir}")
    retriever = RAGRetriever(index_dir=index_dir)
    retriever.load_clip()
    retriever.build_index(samples)
    retriever.save_index()
    retriever.unload_clip()

    print(f"[build_kb] Saved index to {index_dir}")
    return index_dir


def build_all_kb_sizes(sizes=(500, 1000, 2000, 5000, 10000, 20000),
                        eval_image_ids: set = None):
    """Build one KB for each requested size. Pass eval_image_ids to exclude
    any eval sample from all knowledge bases."""
    for size in sizes:
        kb_dir = DATA_DIR / "rag_index" / f"kb_{size}"
        if (kb_dir / "image.index").exists():
            print(f"[build_all_kb_sizes] KB {size} already exists, skipping.")
            continue
        build_kb(num_samples=size, eval_image_ids=eval_image_ids)


# =============================================================================
# 2+3. COMPUTE AND SAVE CLIP EMBEDDINGS / FAISS INDEX
#      (handled inside RAGRetriever.build_index + save_index above)
# =============================================================================

# The RAGRetriever already handles embedding + indexing.  The functions below
# show the raw steps explicitly in case you need to inspect or reuse them.

def compute_and_save_embeddings(samples, out_path: Path) -> np.ndarray:
    """Embed images with CLIP and save as .npy.  Returns array (N, 512)."""
    retriever = RAGRetriever()
    retriever.load_clip()
    images = [s["image"].convert("RGB") for s in samples]
    embeddings = retriever.encode_images(images, batch_size=32)
    np.save(out_path, embeddings)
    retriever.unload_clip()
    print(f"Saved embeddings: {out_path}  shape={embeddings.shape}")
    return embeddings


def build_faiss_index_from_embeddings(embeddings: np.ndarray,
                                       out_path: Path) -> faiss.Index:
    """Build IndexFlatIP on L2-normalised embeddings and save."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index.add(normed.astype(np.float32))
    faiss.write_index(index, str(out_path))
    print(f"Saved FAISS index: {out_path}  ({index.ntotal} vectors, dim={dim})")
    return index


# =============================================================================
# 4. LOAD FAISS AND RETRIEVE TOP-K
# =============================================================================

def load_retriever(index_dir: Path = None) -> RAGRetriever:
    """Load a saved RAGRetriever (FAISS + metadata) from disk."""
    if index_dir is None:
        index_dir = RAG_INDEX_DIR
    retriever = RAGRetriever(index_dir=index_dir)
    retriever.load_clip()
    retriever.load_index()
    return retriever


def retrieve_top_k(retriever: RAGRetriever, image, question: str,
                   top_k: int = 3, alpha: float = 1.0) -> list:
    """Return top-k retrieved examples as a list of dicts."""
    return retriever.retrieve(image, question, top_k=top_k, alpha=alpha)


# =============================================================================
# 5. RANDOM RETRIEVAL BASELINE  (5 seeds → averaged)
# =============================================================================

def run_random_baseline(model, processor, eval_samples: list,
                        kb_metadata: list,
                        top_k: int = 3,
                        n_seeds: int = 5,
                        num_eval: int = EVAL_SAMPLES) -> dict:
    """Replace FAISS retrieval with random KB sampling.

    Runs `n_seeds` independent random draws and averages accuracy.
    This controls for the benefit of prompt-formatting alone.

    Returns:
        dict with per-seed accuracy and mean/std.
    """
    print(f"\n[random_baseline] top_k={top_k}, seeds={n_seeds}, n={num_eval}")
    seed_accs = []

    for seed in range(n_seeds):
        random.seed(seed)
        accs = []
        for item in tqdm(eval_samples[:num_eval],
                         desc=f"  seed {seed}", leave=False):
            image   = item["image"].convert("RGB")
            question = item["question"]
            gt_answers = [a["answer"] for a in item["answers"]]

            # Random retrieval — ignore query entirely
            rand_examples = random.sample(kb_metadata, min(top_k, len(kb_metadata)))
            retrieved = [{"best_answer": ex["best_answer"],
                          "question":    ex["question"]}
                         for ex in rand_examples]

            answer = generate_answer_with_context(
                model, processor, image, question,
                retrieved, max_new_tokens=10
            )
            accs.append(vqa_accuracy(answer, gt_answers))

        seed_acc = sum(accs) / len(accs)
        seed_accs.append(seed_acc)
        print(f"  seed {seed}: {seed_acc*100:.2f}%")

    result = {
        "experiment":  "random_baseline",
        "top_k":       top_k,
        "n_seeds":     n_seeds,
        "seed_accs":   seed_accs,
        "mean_acc":    float(np.mean(seed_accs)),
        "std_acc":     float(np.std(seed_accs)),
        "num_eval":    num_eval,
    }
    print(f"  Mean: {result['mean_acc']*100:.2f}% ± {result['std_acc']*100:.2f}%")
    _save_csv(result, EXPERIMENTS_DIR / "random_baseline.csv",
              rows=[{"seed": i, "accuracy": a}
                    for i, a in enumerate(seed_accs)] +
                   [{"seed": "mean", "accuracy": result["mean_acc"]},
                    {"seed": "std",  "accuracy": result["std_acc"]}])
    return result


# =============================================================================
# 6. RETRIEVAL ORACLE@k
# =============================================================================

def compute_oracle(eval_samples: list,
                   retriever: RAGRetriever,
                   k_values: tuple = (1, 3, 5, 10),
                   num_eval: int = EVAL_SAMPLES) -> dict:
    """Measure retrieval quality: does top-k contain a correct answer?

    This is RETRIEVAL ORACLE@k — it measures how often the retrieved set
    contains a useful answer, using the official VQAv2 soft accuracy metric.
    For each sample, the oracle score at k is the maximum VQA soft accuracy
    over the top-k retrieved answers against all ground-truth annotations.
    Averaged over all eval samples.

    This experiment does NOT load BLIP-2. It only requires the retriever.

    Distinct from Perfect-Hint Context Test:
      Oracle@k  → does retrieval FIND the right answer?
      Perfect-hint → does the MODEL USE the right answer when given it?

    Returns:
        dict mapping k → mean oracle soft accuracy (0–1).
    """
    print(f"\n[oracle] k_values={k_values}, n={num_eval}")
    # Accumulate per-sample oracle scores at each k
    oracle_scores = {k: [] for k in k_values}
    max_k = max(k_values)
    per_sample_rows = []

    for item in tqdm(eval_samples[:num_eval], desc="Oracle@k"):
        image      = item["image"].convert("RGB")
        question   = item["question"]
        gt_answers = [a["answer"] for a in item["answers"]]
        question_id = item.get("question_id", "")
        image_id    = str(item.get("image_id", ""))

        # Retrieve max_k candidates once; score reused at each k cut-off
        retrieved = retriever.retrieve(image, question,
                                       top_k=max_k, alpha=1.0)

        retrieved_answer_strs = [ex["best_answer"] for ex in retrieved]
        retrieved_score_strs  = [f"{ex.get('score', ex.get('img_score', 0)):.4f}"
                                  for ex in retrieved]

        sample_oracle = {}
        for k in k_values:
            # VQA soft accuracy: max over top-k retrieved answers
            max_score = max(
                (vqa_accuracy(ex["best_answer"], gt_answers)
                 for ex in retrieved[:k]),
                default=0.0,
            )
            oracle_scores[k].append(max_score)
            sample_oracle[f"oracle@{k}"] = max_score

        per_sample_rows.append({
            "question_id":       question_id,
            "image_id":          image_id,
            "question":          question,
            "ground_truth":      get_best_answer(item["answers"]),
            "retrieved_answers": " | ".join(retrieved_answer_strs[:max_k]),
            "retrieved_scores":  " | ".join(retrieved_score_strs[:max_k]),
            **sample_oracle,
        })

    oracle = {k: float(np.mean(oracle_scores[k])) for k in k_values}
    for k, v in oracle.items():
        print(f"  Oracle@{k}: {v*100:.2f}%")

    # Summary CSV
    _save_csv({}, EXPERIMENTS_DIR / "oracle_summary.csv",
              rows=[{"k": k, "oracle_soft_acc": oracle[k]} for k in k_values])
    # Per-sample detail CSV
    _save_csv({}, EXPERIMENTS_DIR / "oracle_detail.csv", rows=per_sample_rows)
    return oracle


# =============================================================================
# 7. PERFECT-HINT CONTEXT TEST
# =============================================================================

def run_perfect_hint_test(model, processor, eval_samples: list,
                           num_eval: int = EVAL_SAMPLES) -> dict:
    """Test whether BLIP-2 can use a correct answer when directly injected as a hint.

    PERFECT-HINT CONTEXT TEST — measures the *model's* ability to exploit
    context, not retrieval quality.  The ground-truth answer is injected in
    place of any retrieved example.  If accuracy is low here, the architecture
    (frozen LM backbone, Q-Former bottleneck) is limiting context utilisation,
    regardless of how good retrieval is.

    Distinct from Retrieval Oracle@k:
      Oracle@k     → does RETRIEVAL find the right answer?
      Perfect-hint → does the MODEL output it when given it explicitly?

    Returns:
        dict with aggregate accuracy and per-sample CSV detail.
    """
    print(f"\n[perfect_hint] n={num_eval}")
    accs = []
    per_sample_rows = []

    for item in tqdm(eval_samples[:num_eval], desc="Perfect hint"):
        image       = item["image"].convert("RGB")
        question    = item["question"]
        gt_answers  = [a["answer"] for a in item["answers"]]
        best_gt     = get_best_answer(item["answers"])
        question_id = item.get("question_id", "")
        image_id    = str(item.get("image_id", ""))

        # Inject the ground-truth answer as the sole hint
        perfect_hint = [{"best_answer": best_gt, "question": question}]
        answer = generate_answer_with_context(
            model, processor, image, question,
            perfect_hint, max_new_tokens=10
        )
        acc = vqa_accuracy(answer, gt_answers)
        accs.append(acc)

        per_sample_rows.append({
            "question_id":    question_id,
            "image_id":       image_id,
            "question":       question,
            "ground_truth":   best_gt,
            "injected_hint":  best_gt,
            "prediction":     answer,
            "vqa_acc":        acc,
            "hint_used":      int(acc > 0),
        })

    mean_acc     = sum(accs) / len(accs)
    correct_count = sum(1 for a in accs if a > 0)
    print(f"  Perfect-hint accuracy: {mean_acc*100:.2f}%")
    print(f"  Hint exploited: {correct_count}/{num_eval} "
          f"({100*correct_count/num_eval:.1f}%)")

    result = {
        "experiment":    "perfect_hint",
        "accuracy":      mean_acc,
        "correct_count": correct_count,
        "num_eval":      num_eval,
        "pct_hint_used": correct_count / num_eval,
    }
    _save_csv({}, EXPERIMENTS_DIR / "perfect_hint_summary.csv",
              rows=[{"metric": k, "value": v}
                    for k, v in result.items() if k != "experiment"])
    _save_csv({}, EXPERIMENTS_DIR / "perfect_hint_detail.csv",
              rows=per_sample_rows)
    return result


# =============================================================================
# 8. KB SIZE ABLATION
# =============================================================================

def run_kb_ablation(model, processor, eval_samples: list,
                    kb_sizes: tuple = (500, 1000, 2000, 5000, 10000, 20000),
                    top_k: int = 3,
                    alpha: float = 1.0,
                    num_eval: int = EVAL_SAMPLES) -> dict:
    """Evaluate accuracy at each KB size.

    Each KB must already exist under data/rag_index/kb_{size}/.
    Call build_all_kb_sizes() first if they don't.

    Returns:
        dict mapping kb_size → accuracy.
    """
    print(f"\n[kb_ablation] sizes={kb_sizes}, top_k={top_k}, alpha={alpha}")
    results = {}

    for size in kb_sizes:
        index_dir = DATA_DIR / "rag_index" / f"kb_{size}"
        if not (index_dir / "image.index").exists():
            print(f"  KB {size} not found at {index_dir}, skipping.")
            continue

        retriever = RAGRetriever(index_dir=index_dir)
        retriever.load_clip()
        retriever.load_index()

        accs = []
        for item in tqdm(eval_samples[:num_eval],
                         desc=f"  KB={size}", leave=False):
            image      = item["image"].convert("RGB")
            question   = item["question"]
            gt_answers = [a["answer"] for a in item["answers"]]

            retrieved = retriever.retrieve(image, question,
                                           top_k=top_k, alpha=alpha)
            answer    = generate_answer_with_context(
                model, processor, image, question,
                retrieved, max_new_tokens=10
            )
            accs.append(vqa_accuracy(answer, gt_answers))

        retriever.unload_clip()

        acc = sum(accs) / len(accs)
        results[size] = acc
        print(f"  KB={size:>6}: {acc*100:.2f}%")

    full_result = {
        "experiment": "kb_ablation",
        "top_k": top_k, "alpha": alpha, "num_eval": num_eval,
        **{str(k): v for k, v in results.items()},
    }
    _save_csv(full_result, EXPERIMENTS_DIR / "kb_ablation.csv",
              rows=[{"kb_size": k, "accuracy": v} for k, v in results.items()])
    return results


# =============================================================================
# 9. TOP-K ABLATION
# =============================================================================

def run_topk_ablation(model, processor, eval_samples: list,
                      retriever: RAGRetriever,
                      k_values: tuple = (1, 3, 5, 10),
                      alpha: float = 1.0,
                      num_eval: int = EVAL_SAMPLES) -> dict:
    """Evaluate accuracy for each top-k value using a fixed KB and alpha.

    Returns:
        dict mapping k → accuracy.
    """
    print(f"\n[topk_ablation] k_values={k_values}, alpha={alpha}")
    results = {}

    for k in k_values:
        accs = []
        for item in tqdm(eval_samples[:num_eval],
                         desc=f"  k={k}", leave=False):
            image      = item["image"].convert("RGB")
            question   = item["question"]
            gt_answers = [a["answer"] for a in item["answers"]]

            retrieved = retriever.retrieve(image, question,
                                           top_k=k, alpha=alpha)
            answer    = generate_answer_with_context(
                model, processor, image, question,
                retrieved, max_new_tokens=10
            )
            accs.append(vqa_accuracy(answer, gt_answers))

        acc = sum(accs) / len(accs)
        results[k] = acc
        print(f"  k={k}: {acc*100:.2f}%")

    full_result = {
        "experiment": "topk_ablation",
        "alpha": alpha, "num_eval": num_eval,
        **{str(k): v for k, v in results.items()},
    }
    _save_csv(full_result, EXPERIMENTS_DIR / "topk_ablation.csv",
              rows=[{"k": k, "accuracy": v} for k, v in results.items()])
    return results


# =============================================================================
# 10. ANSWER-TYPE BREAKDOWN
# =============================================================================

def run_answer_type_breakdown(baseline_preds: list, rag_preds: list,
                               eval_samples: list) -> dict:
    """Break accuracy down by answer type.

    Args:
        baseline_preds: list of predicted answer strings (zero-shot).
        rag_preds:      list of predicted answer strings (RAG).
        eval_samples:   matching VQAv2 sample dicts.

    Returns:
        dict mapping answer_type → {"baseline": acc, "rag": acc, "n": count}.
    """
    print(f"\n[answer_type_breakdown]")

    def _bucket(item):
        answer_type   = item.get("answer_type", "other")
        question_type = item.get("question_type", "")
        question      = item.get("question", "").lower()
        if answer_type == "yes/no":
            return "yes/no"
        if "color" in question_type or "what color" in question:
            return "color"
        if "how many" in question or "number" in question_type:
            return "number"
        return "other"

    buckets = {}
    for item, base_pred, rag_pred in zip(eval_samples, baseline_preds, rag_preds):
        gt_answers  = [a["answer"] for a in item["answers"]]
        bucket      = _bucket(item)
        base_acc    = vqa_accuracy(base_pred, gt_answers)
        rag_acc_val = vqa_accuracy(rag_pred, gt_answers)

        if bucket not in buckets:
            buckets[bucket] = {"baseline": [], "rag": [], "n": 0}
        buckets[bucket]["baseline"].append(base_acc)
        buckets[bucket]["rag"].append(rag_acc_val)
        buckets[bucket]["n"] += 1

    results = {}
    rows = []
    for bucket, data in buckets.items():
        base_mean = sum(data["baseline"]) / len(data["baseline"])
        rag_mean  = sum(data["rag"])      / len(data["rag"])
        results[bucket] = {
            "baseline": base_mean,
            "rag":      rag_mean,
            "delta":    rag_mean - base_mean,
            "n":        data["n"],
        }
        rows.append({"answer_type": bucket, "n": data["n"],
                     "baseline_acc": base_mean, "rag_acc": rag_mean,
                     "delta": rag_mean - base_mean})
        print(f"  {bucket:<10} n={data['n']:>4}  "
              f"base={base_mean*100:.1f}%  "
              f"rag={rag_mean*100:.1f}%  "
              f"Δ={((rag_mean-base_mean)*100):+.1f}pp")

    _save_csv({"experiment": "answer_type"}, EXPERIMENTS_DIR / "answer_type.csv",
              rows=rows)
    return results


# =============================================================================
# 11. HELPFUL / HARMFUL / NEUTRAL
# =============================================================================

def run_helpful_harmful(baseline_preds: list, rag_preds: list,
                         eval_samples: list) -> dict:
    """Categorise each sample as helpful, harmful, or neutral.

    Definitions:
        helpful      — RAG correct,  baseline wrong
        harmful      — RAG wrong,    baseline correct
        neutral_good — both correct
        neutral_bad  — both wrong

    Returns:
        dict with counts, percentages, and per-sample labels.
    """
    print(f"\n[helpful_harmful]")
    labels = []
    counts = {"helpful": 0, "harmful": 0, "neutral_good": 0, "neutral_bad": 0}

    for item, base_pred, rag_pred in zip(eval_samples, baseline_preds, rag_preds):
        gt_answers  = [a["answer"] for a in item["answers"]]
        base_ok     = vqa_accuracy(base_pred, gt_answers) > 0
        rag_ok      = vqa_accuracy(rag_pred,  gt_answers) > 0

        if rag_ok and not base_ok:
            label = "helpful"
        elif not rag_ok and base_ok:
            label = "harmful"
        elif rag_ok and base_ok:
            label = "neutral_good"
        else:
            label = "neutral_bad"

        counts[label] += 1
        labels.append({
            "question_id":  item.get("question_id", ""),
            "image_id":     str(item.get("image_id", "")),
            "question":     item["question"],
            "ground_truth": get_best_answer(item["answers"]),
            "baseline_pred": base_pred,
            "rag_pred":     rag_pred,
            "baseline_acc": vqa_accuracy(base_pred, gt_answers),
            "rag_acc":      vqa_accuracy(rag_pred, gt_answers),
            "label":        label,
        })

    total = len(labels)
    pcts  = {k: v / total for k, v in counts.items()}
    for k, v in counts.items():
        print(f"  {k:<14}: {v:>4} ({pcts[k]*100:.1f}%)")

    result = {"experiment": "helpful_harmful",
              "total": total, **counts, **{f"{k}_pct": v for k, v in pcts.items()}}
    _save_csv(result, EXPERIMENTS_DIR / "helpful_harmful_summary.csv",
              rows=[{"category": k, "count": counts[k], "pct": pcts[k]}
                    for k in counts])
    _save_csv(result, EXPERIMENTS_DIR / "helpful_harmful_detail.csv",
              rows=labels)
    return result


# =============================================================================
# 12. RETRIEVAL CONFIDENCE VS ACCURACY
# =============================================================================

def run_confidence_vs_accuracy(retrieval_meta: list,
                                eval_samples: list,
                                n_buckets: int = 5) -> dict:
    """Does higher retrieval similarity predict higher RAG accuracy?

    Uses the per-sample retrieval metadata from _collect_predictions.
    Buckets samples by top-1 similarity score and by top1−top2 margin,
    then computes mean RAG accuracy per bucket.

    Args:
        retrieval_meta: list of dicts returned by _collect_predictions.
        eval_samples:   matching VQAv2 sample dicts (for gt_answers).
        n_buckets:      number of equal-width buckets.

    Returns:
        dict with bucketed results for top1_score and margin.
    """
    print(f"\n[confidence_vs_accuracy] n={len(retrieval_meta)}, buckets={n_buckets}")

    per_sample_rows = []
    scores, margins, rag_accs = [], [], []

    for meta, item in zip(retrieval_meta, eval_samples):
        gt_answers = [a["answer"] for a in item["answers"]]
        rag_acc    = vqa_accuracy(meta["rag_pred"], gt_answers)
        scores.append(meta["top1_score"])
        margins.append(meta["margin"])
        rag_accs.append(rag_acc)
        per_sample_rows.append({
            "question_id":   meta["question_id"],
            "image_id":      meta["image_id"],
            "question":      meta["question"],
            "ground_truth":  meta["ground_truth"],
            "prediction":    meta["rag_pred"],
            "retrieved_answers": meta["retrieved_answers"],
            "retrieved_scores":  meta["retrieved_scores"],
            "top1_score":    meta["top1_score"],
            "margin":        meta["margin"],
            "rag_correct":   int(rag_acc > 0),
            "rag_vqa_acc":   rag_acc,
        })

    def _bucket_results(values, accs, label):
        arr   = np.array(values)
        lo, hi = arr.min(), arr.max()
        edges = np.linspace(lo, hi, n_buckets + 1)
        rows  = []
        for i in range(n_buckets):
            mask      = (arr >= edges[i]) & (arr < edges[i + 1])
            if i == n_buckets - 1:            # include right edge in last bucket
                mask  = (arr >= edges[i]) & (arr <= edges[i + 1])
            bucket_accs = [accs[j] for j in range(len(accs)) if mask[j]]
            rows.append({
                f"{label}_lo":    edges[i],
                f"{label}_hi":    edges[i + 1],
                "n":              len(bucket_accs),
                "mean_rag_acc":   float(np.mean(bucket_accs)) if bucket_accs else 0.0,
            })
        return rows

    score_rows  = _bucket_results(scores,  rag_accs, "top1_score")
    margin_rows = _bucket_results(margins, rag_accs, "margin")

    for row in score_rows:
        print(f"  top1∈[{row['top1_score_lo']:.3f},{row['top1_score_hi']:.3f}] "
              f"n={row['n']:>3}  acc={row['mean_rag_acc']*100:.1f}%")

    _save_csv({}, EXPERIMENTS_DIR / "confidence_vs_accuracy_detail.csv",
              rows=per_sample_rows)
    _save_csv({}, EXPERIMENTS_DIR / "confidence_vs_accuracy_score_buckets.csv",
              rows=score_rows)
    _save_csv({}, EXPERIMENTS_DIR / "confidence_vs_accuracy_margin_buckets.csv",
              rows=margin_rows)

    return {"score_buckets": score_rows, "margin_buckets": margin_rows}


def plot_confidence_vs_accuracy(conf_result: dict) -> None:
    """Two side-by-side plots: top-1 score buckets and margin buckets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, xlabel, title in [
        (ax1, "score_buckets",  "Top-1 similarity score",  "Retrieval Score vs. Accuracy"),
        (ax2, "margin_buckets", "Top1 − Top2 margin",      "Score Margin vs. Accuracy"),
    ]:
        rows  = conf_result[key]
        lo_key = [k for k in rows[0] if k.endswith("_lo")][0]
        labels = [f"{r[lo_key]:.2f}" for r in rows]
        means  = [r["mean_rag_acc"] * 100 for r in rows]
        ns     = [r["n"] for r in rows]

        bars = ax.bar(labels, means, color="#0891b2", alpha=0.85)
        ax.bar_label(bars, labels=[f"{m:.1f}%" for m in means], padding=3, fontsize=8)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, -4,
                    f"n={n}", ha="center", va="top", fontsize=7, color="#64748b")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean RAG VQA Accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(means) * 1.2 if means else 100)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "confidence_vs_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 13. CSV HELPERS
# =============================================================================

def _save_csv(result: dict, path: Path, rows: list) -> None:
    """Write `rows` (list of dicts) to a CSV at `path`."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def save_prediction_pair(baseline_preds: list, rag_preds: list,
                          eval_samples: list,
                          retrieval_meta: list = None) -> None:
    """Save baseline vs RAG predictions side by side for manual inspection."""
    rows = []
    for i, (item, base_pred, rag_pred) in enumerate(
            zip(eval_samples, baseline_preds, rag_preds)):
        gt_answers = [a["answer"] for a in item["answers"]]
        meta       = retrieval_meta[i] if retrieval_meta else {}
        rows.append({
            "question_id":       item.get("question_id", ""),
            "image_id":          str(item.get("image_id", "")),
            "question":          item["question"],
            "ground_truth":      get_best_answer(item["answers"]),
            "baseline_pred":     base_pred,
            "rag_pred":          rag_pred,
            "baseline_acc":      vqa_accuracy(base_pred, gt_answers),
            "rag_acc":           vqa_accuracy(rag_pred, gt_answers),
            "retrieved_answers": meta.get("retrieved_answers", ""),
            "retrieved_scores":  meta.get("retrieved_scores", ""),
            "top1_score":        meta.get("top1_score", ""),
        })
    _save_csv({}, EXPERIMENTS_DIR / "predictions_comparison.csv", rows)


# =============================================================================
# 13. MATPLOTLIB PLOTS
# =============================================================================

def plot_alpha_curve(alpha_results: dict, baseline_acc: float) -> None:
    """Line plot: accuracy vs alpha.

    Args:
        alpha_results: dict mapping alpha (float) → accuracy (float).
        baseline_acc: zero-shot accuracy for horizontal reference line.
    """
    alphas = sorted(alpha_results.keys())
    accs   = [alpha_results[a] * 100 for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, accs, marker="o", linewidth=2, color="#2563eb", label="RAG")
    ax.axhline(baseline_acc * 100, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline_acc*100:.1f}%)")
    ax.set_xlabel("Alpha (image weight)")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Image vs. Text Retrieval Weight")
    ax.set_xticks(alphas)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "alpha_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_kb_ablation(kb_results: dict, baseline_acc: float) -> None:
    """Line plot: accuracy vs KB size.

    Args:
        kb_results:   dict mapping kb_size (int) → accuracy (float).
        baseline_acc: zero-shot accuracy for reference.
    """
    sizes = sorted(kb_results.keys())
    accs  = [kb_results[s] * 100 for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, accs, marker="o", linewidth=2, color="#16a34a", label="RAG")
    ax.axhline(baseline_acc * 100, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline_acc*100:.1f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("Knowledge Base Size")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Scaling the Visual Knowledge Base")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "kb_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_topk_ablation(topk_results: dict, baseline_acc: float) -> None:
    """Bar chart: accuracy vs top-k.

    Args:
        topk_results: dict mapping k (int) → accuracy (float).
        baseline_acc: zero-shot accuracy for reference.
    """
    ks   = sorted(topk_results.keys())
    accs = [topk_results[k] * 100 for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(k) for k in ks], accs, color="#7c3aed", width=0.5)
    ax.axhline(baseline_acc * 100, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline_acc*100:.1f}%)")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Top-k Retrieved Examples")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Number of Retrieved Hints")
    ax.legend()
    ax.set_ylim(0, max(accs) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "topk_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_oracle(oracle_results: dict, rag_acc: float, baseline_acc: float) -> None:
    """Bar chart: Oracle@k recall vs RAG accuracy vs baseline.

    Args:
        oracle_results: dict mapping k → oracle recall (0–1).
        rag_acc:        best RAG accuracy (for gap annotation).
        baseline_acc:   zero-shot accuracy.
    """
    ks     = sorted(oracle_results.keys())
    recalls = [oracle_results[k] * 100 for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"@{k}" for k in ks], recalls, color="#0891b2", width=0.5,
           label="Oracle recall")
    ax.axhline(rag_acc * 100, color="#f97316", linestyle="--",
               linewidth=1.5, label=f"RAG accuracy ({rag_acc*100:.1f}%)")
    ax.axhline(baseline_acc * 100, color="#dc2626", linestyle=":",
               linewidth=1.5, label=f"Baseline ({baseline_acc*100:.1f}%)")
    ax.set_xlabel("Retrieval depth k")
    ax.set_ylabel("Recall / Accuracy (%)")
    ax.set_title("Retrieval Oracle@k vs. System Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "oracle.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_helpful_harmful(hh_result: dict) -> None:
    """Horizontal stacked bar showing helpful/harmful/neutral split."""
    categories = ["helpful", "harmful", "neutral_good", "neutral_bad"]
    colors     = ["#16a34a", "#dc2626", "#2563eb", "#9ca3af"]
    labels     = ["Helpful", "Harmful", "Neutral+", "Neutral−"]
    counts     = [hh_result[c] for c in categories]
    total      = hh_result["total"]
    pcts       = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 2))
    left = 0
    for pct, color, label in zip(pcts, colors, labels):
        ax.barh(0, pct, left=left, color=color, label=f"{label} ({pct:.1f}%)")
        if pct > 4:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%",
                    ha="center", va="center", color="white", fontsize=9,
                    fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% of evaluation samples")
    ax.set_title("Effect of RAG: Helpful / Harmful / Neutral")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.6), ncol=4, fontsize=8)
    fig.tight_layout()
    path = PLOTS_DIR / "helpful_harmful.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_answer_type(type_results: dict) -> None:
    """Grouped bar chart: baseline vs RAG accuracy by answer type."""
    types   = list(type_results.keys())
    base_v  = [type_results[t]["baseline"] * 100 for t in types]
    rag_v   = [type_results[t]["rag"]      * 100 for t in types]
    x       = np.arange(len(types))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width/2, base_v, width, label="Baseline", color="#94a3b8")
    b2 = ax.bar(x + width/2, rag_v,  width, label="RAG",      color="#2563eb")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Accuracy by Answer Type")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "answer_type.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# MAIN — wire everything together
# =============================================================================

def _load_models(quantize: bool = True):
    """Load BLIP-2 model + processor once and reuse."""
    return load_model(quantize=quantize)


def _collect_predictions(model, processor, retriever,
                          eval_samples, top_k=3, alpha=1.0,
                          num_eval=EVAL_SAMPLES):
    """Collect baseline and RAG predictions with full retrieval metadata.

    Returns:
        baseline_preds: list of answer strings (zero-shot).
        rag_preds:      list of answer strings (RAG).
        retrieval_meta: list of dicts with per-sample retrieval info:
            question_id, image_id, question, ground_truth,
            retrieved_answers, retrieved_scores, top1_score, margin.
    """
    baseline_preds, rag_preds, retrieval_meta = [], [], []
    for item in tqdm(eval_samples[:num_eval], desc="Collecting predictions"):
        image       = item["image"].convert("RGB")
        question    = item["question"]
        question_id = item.get("question_id", "")
        image_id    = str(item.get("image_id", ""))

        base_ans = generate_answer(model, processor, image, question)
        baseline_preds.append(base_ans)

        retrieved = retriever.retrieve(image, question, top_k=top_k, alpha=alpha)
        rag_ans   = generate_answer_with_context(
            model, processor, image, question, retrieved, max_new_tokens=10
        )
        rag_preds.append(rag_ans)

        scores = [ex.get("score", ex.get("img_score", 0.0)) for ex in retrieved]
        top1   = scores[0] if scores else 0.0
        margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
        retrieval_meta.append({
            "question_id":       question_id,
            "image_id":          image_id,
            "question":          question,
            "ground_truth":      get_best_answer(item["answers"]),
            "baseline_pred":     base_ans,
            "rag_pred":          rag_ans,
            "retrieved_answers": " | ".join(ex["best_answer"] for ex in retrieved),
            "retrieved_scores":  " | ".join(f"{s:.4f}" for s in scores),
            "top1_score":        top1,
            "margin":            margin,
        })

    return baseline_preds, rag_preds, retrieval_meta


def run_all(num_eval: int = EVAL_SAMPLES, quantize: bool = True):
    """Run the full ablation suite end-to-end."""
    BASELINE_ACC = 0.4233   # from published zero-shot run (n=100)
    RAG_BEST_ACC = 0.5133   # from published RAG α=1.0 run (n=100)

    # ── Load eval set (validation split, image-disjoint from KB) ──
    print("Loading eval set...")
    eval_samples = load_vqav2(split="validation",
                               num_samples=num_eval,
                               offset=EVAL_OFFSET)

    # ── Oracle@k does NOT need BLIP-2 — run first, cheaply ──
    retriever = load_retriever()
    oracle_result = compute_oracle(eval_samples, retriever, num_eval=num_eval)
    plot_oracle(oracle_result, RAG_BEST_ACC, BASELINE_ACC)

    # ── Load BLIP-2 for all generation experiments ──
    model, processor = _load_models(quantize=quantize)

    # ── Collect paired predictions once — reused by multiple experiments ──
    print("\nCollecting baseline + RAG predictions...")
    baseline_preds, rag_preds, retrieval_meta = _collect_predictions(
        model, processor, retriever, eval_samples,
        top_k=3, alpha=1.0, num_eval=num_eval
    )
    save_prediction_pair(baseline_preds, rag_preds,
                          eval_samples[:num_eval], retrieval_meta)

    # ── Load KB metadata for random baseline ──
    meta_path = RAG_INDEX_DIR / "metadata.pkl"
    with open(meta_path, "rb") as f:
        kb_metadata = pickle.load(f)

    # ── Run experiments that need the model ──
    random_result = run_random_baseline(model, processor, eval_samples,
                                         kb_metadata, num_eval=num_eval)
    hint_result   = run_perfect_hint_test(model, processor, eval_samples,
                                           num_eval=num_eval)
    topk_results  = run_topk_ablation(model, processor, eval_samples,
                                       retriever, num_eval=num_eval)

    # ── Post-processing experiments (use cached predictions) ──
    type_results = run_answer_type_breakdown(
        baseline_preds, rag_preds, eval_samples[:num_eval])
    hh_result    = run_helpful_harmful(
        baseline_preds, rag_preds, eval_samples[:num_eval])
    conf_result  = run_confidence_vs_accuracy(
        retrieval_meta, eval_samples[:num_eval])

    # ── KB ablation builds its own retrievers — free CLIP first ──
    retriever.unload_clip()
    kb_results = run_kb_ablation(model, processor, eval_samples,
                                  num_eval=num_eval)

    # ── Plot everything ──
    print("\nGenerating plots...")
    plot_topk_ablation(topk_results, BASELINE_ACC)
    plot_kb_ablation(kb_results, BASELINE_ACC)
    plot_helpful_harmful(hh_result)
    plot_answer_type(type_results)
    plot_confidence_vs_accuracy(conf_result)

    print(f"\nAll experiments complete.")
    print(f"CSVs:  {EXPERIMENTS_DIR}")
    print(f"Plots: {PLOTS_DIR}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-VQA ablation experiments")
    parser.add_argument(
        "--exp", required=True,
        choices=["build_kb", "random_baseline", "oracle", "perfect_hint",
                 "kb_ablation", "topk_ablation", "answer_type",
                 "helpful_harmful", "confidence", "all"],
        help="Which experiment to run",
    )
    parser.add_argument("--num_eval", type=int, default=EVAL_SAMPLES,
                        help=f"Eval samples (default {EVAL_SAMPLES} = published baseline)")
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--kb_sizes", nargs="+", type=int,
                        default=[500, 1000, 2000, 5000, 10000, 20000])
    args = parser.parse_args()

    if args.exp == "build_kb":
        # Load eval set first so we can exclude its image_ids from the KB
        print("Loading eval set to collect image_ids for leakage prevention...")
        eval_samples_for_ids = load_vqav2(split="validation",
                                           num_samples=args.num_eval,
                                           offset=EVAL_OFFSET)
        eval_ids = {str(s.get("image_id", "")) for s in eval_samples_for_ids}
        build_all_kb_sizes(sizes=args.kb_sizes, eval_image_ids=eval_ids)

    elif args.exp == "all":
        run_all(num_eval=args.num_eval, quantize=not args.no_quantize)

    elif args.exp == "oracle":
        # Oracle@k does NOT require BLIP-2 — only loads CLIP + FAISS
        print("Loading eval set...")
        eval_samples = load_vqav2(split="validation",
                                   num_samples=args.num_eval,
                                   offset=EVAL_OFFSET)
        retriever = load_retriever()
        oracle = compute_oracle(eval_samples, retriever, num_eval=args.num_eval)
        plot_oracle(oracle, rag_acc=0.5133, baseline_acc=0.4233)

    else:
        # All remaining experiments require BLIP-2
        print("Loading eval set...")
        eval_samples = load_vqav2(split="validation",
                                   num_samples=args.num_eval,
                                   offset=EVAL_OFFSET)
        retriever = load_retriever()
        model, processor = _load_models(quantize=not args.no_quantize)

        if args.exp == "random_baseline":
            meta_path = RAG_INDEX_DIR / "metadata.pkl"
            with open(meta_path, "rb") as f:
                kb_metadata = pickle.load(f)
            run_random_baseline(model, processor, eval_samples,
                                 kb_metadata, num_eval=args.num_eval)

        elif args.exp == "perfect_hint":
            run_perfect_hint_test(model, processor, eval_samples,
                                   num_eval=args.num_eval)

        elif args.exp == "topk_ablation":
            results = run_topk_ablation(model, processor, eval_samples,
                                         retriever, num_eval=args.num_eval)
            plot_topk_ablation(results, baseline_acc=0.4233)

        elif args.exp == "kb_ablation":
            retriever.unload_clip()
            results = run_kb_ablation(model, processor, eval_samples,
                                       num_eval=args.num_eval)
            plot_kb_ablation(results, baseline_acc=0.4233)

        elif args.exp in ("answer_type", "helpful_harmful", "confidence"):
            print("Collecting predictions...")
            baseline_preds, rag_preds, retrieval_meta = _collect_predictions(
                model, processor, retriever, eval_samples,
                num_eval=args.num_eval
            )
            save_prediction_pair(baseline_preds, rag_preds,
                                  eval_samples[:args.num_eval], retrieval_meta)

            if args.exp == "answer_type":
                results = run_answer_type_breakdown(
                    baseline_preds, rag_preds, eval_samples[:args.num_eval])
                plot_answer_type(results)

            elif args.exp == "helpful_harmful":
                results = run_helpful_harmful(
                    baseline_preds, rag_preds, eval_samples[:args.num_eval])
                plot_helpful_harmful(results)

            elif args.exp == "confidence":
                results = run_confidence_vs_accuracy(
                    retrieval_meta, eval_samples[:args.num_eval])
                plot_confidence_vs_accuracy(results)
