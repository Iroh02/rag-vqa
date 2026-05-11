"""Two analyses on the same 100-sample eval, no new BLIP-2 inference:

  1. CLEAN ORACLE@K on the 20k diverse KB (replaces the leaky 5k oracle plot)
     — for each sample, retrieve top-10 at α=1.0 and check whether GT appears
       in the retrieved answers at k = 1, 3, 5, 10.

  2. RETRIEVAL vs GENERATOR DECOMPOSITION (the bottleneck question)
     — for every wrong rag_k3 prediction, classify into:
        retrieval_failure  : GT not in any top-3 retrieved answer
        generator_failure  : GT in top-3 retrieved BUT the model picked something else
       and for every correct rag_k3:
        retrieval_then_lm  : GT in retrieved AND model picked it
        generator_smart    : GT NOT in retrieved AND model still got it right

Outputs:
  results/experiments/clean_oracle_at_k.csv
  results/experiments/bottleneck_decomposition.csv
  results/experiments/bottleneck_decomposition.md
  results/plots/clean_oracle_at_k.png
  results/plots/bottleneck_decomposition.png

Run:
  python -m scripts.clean_oracle_and_decomposition
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.rag_retriever import RAGRetriever


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").strip().lower())


def in_retrieved(gt_all: list[str], retrieved_answers: list[str]) -> bool:
    """True iff any annotator answer appears (whole-word) in any retrieved answer."""
    rs = [_norm(r) for r in retrieved_answers]
    for g in gt_all:
        gn = _norm(g)
        if not gn:
            continue
        for r in rs:
            if r == gn:
                return True
            # whole-word substring
            if re.search(r"\b" + re.escape(gn) + r"\b", r):
                return True
    return False


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("Loading retriever (CLIP + 20k FAISS)...")
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()
    kb = retriever.image_index.ntotal
    print(f"  KB size: {kb}")

    print("\nLoading 100 eval samples (offset=5000)...")
    dataset = load_vqav2(split="validation", num_samples=100, offset=5000)

    # Re-retrieve top-10 per sample at α=1.0 (matches the headline RAG config)
    print("\nRetrieving top-10 per sample at α=1.0...")
    per_sample = []
    for i, item in enumerate(dataset):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]
        qid      = str(item.get("question_id", i))
        retr10 = retriever.retrieve(image, question, top_k=10, alpha=1.0, candidate_k=30)
        per_sample.append({
            "qid":      qid,
            "question": question,
            "gt_all":   gt_all,
            "retrieved_answers": [
                (r.get("best_answer") or r.get("answer") or "") for r in retr10
            ],
        })
        if (i + 1) % 20 == 0:
            print(f"  retrieved {i+1}/100", flush=True)

    # ── 1. clean oracle@k ────────────────────────────────────────────────────
    print("\nComputing clean oracle@k on the 20k diverse KB...")
    oracle = {}
    for k in (1, 3, 5, 10):
        hits = sum(in_retrieved(s["gt_all"], s["retrieved_answers"][:k]) for s in per_sample)
        oracle[k] = hits / len(per_sample)
        print(f"  oracle@{k:<2} = {oracle[k]*100:6.2f}%   ({hits}/{len(per_sample)})")

    # ── 2. retrieval vs generator decomposition ──────────────────────────────
    # Need rag_k3 predictions — load from final_ablations_detail.csv
    print("\nLoading rag_k3 predictions from final_ablations_detail.csv...")
    detail = ROOT / "results" / "experiments" / "final_ablations_detail.csv"
    if not detail.exists():
        print("  ERROR: final_ablations_detail.csv missing. Run scripts.final_ablations first.")
        return 1
    rag_pred_by_qid = {r["qid"]: r["rag_k3"] for r in csv.DictReader(open(detail, encoding="utf-8"))}

    print(f"  loaded rag_k3 for {len(rag_pred_by_qid)} samples")

    print("\nDecomposing 100 cases...")
    cats = {
        "retrieval_then_lm":  [],   # GT in retrieved AND rag correct
        "generator_smart":    [],   # GT not in retrieved BUT rag correct
        "generator_failure":  [],   # GT in retrieved BUT rag wrong
        "retrieval_failure":  [],   # GT not in retrieved AND rag wrong
    }

    for s in per_sample:
        qid = s["qid"]
        gt_all = s["gt_all"]
        retr3 = s["retrieved_answers"][:3]
        rag = rag_pred_by_qid.get(qid, "")
        rag_correct = vqa_accuracy(rag, gt_all) >= 1.0
        gt_in_top3  = in_retrieved(gt_all, retr3)

        if   rag_correct and     gt_in_top3:  bucket = "retrieval_then_lm"
        elif rag_correct and not gt_in_top3:  bucket = "generator_smart"
        elif not rag_correct and     gt_in_top3:  bucket = "generator_failure"
        else:                                     bucket = "retrieval_failure"
        cats[bucket].append({
            "qid": qid, "question": s["question"],
            "gt": "|".join(gt_all[:3]),
            "rag": rag, "retrieved_top3": " || ".join(retr3),
        })

    n = sum(len(v) for v in cats.values())
    print(f"\nBOTTLENECK DECOMPOSITION (n={n})")
    print("-" * 76)
    for cat in ["retrieval_then_lm", "generator_smart", "generator_failure", "retrieval_failure"]:
        c = len(cats[cat])
        print(f"  {cat:<22}  {c:>3} ({c/n*100:5.1f}%)")
    correct = len(cats["retrieval_then_lm"]) + len(cats["generator_smart"])
    wrong   = len(cats["generator_failure"]) + len(cats["retrieval_failure"])
    print()
    print(f"  → Total correct (rag_k3): {correct}/{n}  ({correct/n*100:.1f}%)  "
          f"sanity check vs reported 59.33%")
    print(f"  → Of {wrong} wrong cases:")
    if wrong:
        gf = len(cats["generator_failure"])
        rf = len(cats["retrieval_failure"])
        print(f"      generator_failure (model ignored evidence): {gf}/{wrong}  ({gf/wrong*100:.1f}%)")
        print(f"      retrieval_failure (evidence not present)  : {rf}/{wrong}  ({rf/wrong*100:.1f}%)")

    # ── save outputs ─────────────────────────────────────────────────────────
    exp = ROOT / "results" / "experiments"
    plots = ROOT / "results" / "plots"
    exp.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    # Oracle CSV
    oracle_csv = exp / "clean_oracle_at_k.csv"
    with open(oracle_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "oracle_at_k_pct", "kb", "alpha", "n_samples", "notes"])
        for k in (1, 3, 5, 10):
            w.writerow([k, round(oracle[k]*100, 2), "20k_diverse", 1.0, len(per_sample),
                        "GT appears as whole-word in any top-k retrieved answer (image-id-disjoint)"])
    print(f"\nSaved {oracle_csv}")

    # Decomposition CSV (aggregate + per-sample)
    decomp_csv = exp / "bottleneck_decomposition.csv"
    with open(decomp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "pct_of_total", "pct_of_wrong"])
        for cat in ["retrieval_then_lm", "generator_smart", "generator_failure", "retrieval_failure"]:
            c = len(cats[cat])
            pct_all = c / n * 100 if n else 0
            pct_wrong = c / wrong * 100 if wrong and cat in ("generator_failure", "retrieval_failure") else ""
            w.writerow([cat, c, round(pct_all, 2), round(pct_wrong, 2) if pct_wrong != "" else ""])
    print(f"Saved {decomp_csv}")

    # Per-sample detail
    decomp_detail = exp / "bottleneck_decomposition_detail.csv"
    with open(decomp_detail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "category", "question", "gt", "rag_k3", "retrieved_top3"])
        for cat, items in cats.items():
            for it in items:
                w.writerow([it["qid"], cat, it["question"], it["gt"], it["rag"], it["retrieved_top3"]])
    print(f"Saved {decomp_detail}")

    # Markdown summary
    md = exp / "bottleneck_decomposition.md"
    md_lines = ["# Clean Oracle@k + Bottleneck Decomposition"]
    md_lines.append("")
    md_lines.append("100-sample eval (validation offset=5000) · 20k diverse KB (image-id-disjoint) · α=1.0.")
    md_lines.append("")
    md_lines.append("## Clean oracle@k on the 20k KB")
    md_lines.append("")
    md_lines.append("| k | clean 20k oracle@k | leaky 5k val oracle@k (legacy) |")
    md_lines.append("|---:|---:|---:|")
    for k, leaky in [(1, 24.00), (3, 60.33), (5, 83.33), (10, 92.33)]:
        md_lines.append(f"| {k} | **{oracle[k]*100:.2f}%** | {leaky:.2f}% |")
    md_lines.append("")
    md_lines.append("The legacy `oracle@10 = 92.33%` was image-id-leakage-driven on the 5k validation-built KB. "
                    "The clean number on the image-id-disjoint 20k KB is much lower — that's the **real** "
                    "ceiling of any selector that picks among the top-k retrieved answers.")
    md_lines.append("")
    md_lines.append("## Bottleneck decomposition (rag_k3, n=100)")
    md_lines.append("")
    md_lines.append("| Category | Count | % of all | % of wrong |")
    md_lines.append("|---|---:|---:|---:|")
    for cat in ["retrieval_then_lm", "generator_smart", "generator_failure", "retrieval_failure"]:
        c = len(cats[cat])
        pa = c / n * 100
        pw = c / wrong * 100 if wrong and cat in ("generator_failure", "retrieval_failure") else None
        pwt = f"{pw:.1f}%" if pw is not None else "—"
        md_lines.append(f"| {cat} | {c} | {pa:.1f}% | {pwt} |")
    md_lines.append("")
    md_lines.append("**Definitions:**")
    md_lines.append("- `retrieval_then_lm` — GT in retrieved AND model picked it (working as designed)")
    md_lines.append("- `generator_smart`  — GT NOT in retrieved AND model still got it right (image-grounded baseline strength)")
    md_lines.append("- `generator_failure` — GT IS in retrieved BUT model picked wrong (the **'evidence ignored'** failure mode)")
    md_lines.append("- `retrieval_failure` — GT not in retrieved (the **retrieval ceiling** — needs more KB / better retriever)")
    md_lines.append("")
    if wrong:
        gf, rf = len(cats["generator_failure"]), len(cats["retrieval_failure"])
        md_lines.append(f"**Of the {wrong} wrong cases:**")
        md_lines.append(f"- {gf} ({gf/wrong*100:.0f}%) are generator-bound — fixing requires a fine-tuned generator or better evidence-aware selection.")
        md_lines.append(f"- {rf} ({rf/wrong*100:.0f}%) are retrieval-bound — fixing requires bigger/better KB or a trainable retriever.")
        md_lines.append("")
        bigger = "generator-bound" if gf > rf else "retrieval-bound"
        md_lines.append(f"**Headline bottleneck: {bigger}.**")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {md}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. oracle@k clean vs legacy
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ks = [1, 3, 5, 10]
        clean = [oracle[k]*100 for k in ks]
        legacy = [24.00, 60.33, 83.33, 92.33]
        x = list(range(len(ks)))
        ax.bar([i-0.18 for i in x], clean,  width=0.36, color="#3fb950", label="Clean 20k diverse KB (this run)")
        ax.bar([i+0.18 for i in x], legacy, width=0.36, color="#f0883e", label="Legacy 5k val KB (image-id leakage)")
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel("Oracle@k (%)")
        ax.set_title("Oracle@k — GT appears in top-k retrieved answers")
        ax.legend(loc="upper left")
        for i, v in enumerate(clean):
            ax.text(i-0.18, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#3fb950", fontweight="bold")
        for i, v in enumerate(legacy):
            ax.text(i+0.18, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#f0883e")
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(plots / "clean_oracle_at_k.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {plots/'clean_oracle_at_k.png'}")

        # 2. bottleneck decomposition stacked bar
        fig, ax = plt.subplots(figsize=(9, 3.6))
        labels = ["", ""]
        cats_correct = [len(cats["retrieval_then_lm"]), len(cats["generator_smart"])]
        cats_wrong   = [len(cats["generator_failure"]), len(cats["retrieval_failure"])]
        # one-bar layout: correct vs wrong segments
        ax.barh([0], [len(cats["retrieval_then_lm"])],
                color="#3fb950", label=f"retrieval_then_lm  ({len(cats['retrieval_then_lm'])})")
        ax.barh([0], [len(cats["generator_smart"])],
                left=len(cats["retrieval_then_lm"]),
                color="#58a6ff", label=f"generator_smart  ({len(cats['generator_smart'])})")
        ax.barh([0], [len(cats["generator_failure"])],
                left=len(cats["retrieval_then_lm"])+len(cats["generator_smart"]),
                color="#f0883e", label=f"generator_failure  ({len(cats['generator_failure'])})")
        ax.barh([0], [len(cats["retrieval_failure"])],
                left=len(cats["retrieval_then_lm"])+len(cats["generator_smart"])+len(cats["generator_failure"]),
                color="#f85149", label=f"retrieval_failure  ({len(cats['retrieval_failure'])})")
        ax.set_xlim(0, n)
        ax.set_xlabel(f"Sample count (n={n})")
        ax.set_yticks([])
        ax.set_title("rag_k3 outcome decomposition — where the 100 cases fall")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(plots / "bottleneck_decomposition.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {plots/'bottleneck_decomposition.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\n*** DONE ***")
    print(f"Clean oracle@1={oracle[1]*100:.2f}% · @3={oracle[3]*100:.2f}% · @5={oracle[5]*100:.2f}% · @10={oracle[10]*100:.2f}%")
    print(f"Decomposition: {len(cats['retrieval_then_lm'])} retrieval_then_lm · "
          f"{len(cats['generator_smart'])} generator_smart · "
          f"{len(cats['generator_failure'])} generator_failure · "
          f"{len(cats['retrieval_failure'])} retrieval_failure")


if __name__ == "__main__":
    main()
