"""Final ablation suite — 5 measurements on the same 100-sample eval.

Single-pass design: for each sample, compute all retrievals + generations once,
then derive every ablation from the cached predictions.

Ablations:
  1. Evidence-aware selector       (rule-based override of current_prompt RAG)
  2. Type-conditional gate         (yes/no → baseline; open-ended → RAG)
  3. Clean 20k k-sweep             (k = 1, 3, 5, 10 with α=1.0)
  4. Cross-encoder rerank          (α=1.0, k=3, rerank=True)
  5. Self-consistency vote         (majority of α ∈ {0.0, 0.5, 1.0}, k=3)

Outputs:
  results/experiments/final_ablations.csv
  results/experiments/final_ablations.md
  results/plots/final_ablations.png

Run (with API stopped, GPU free):
  python -m scripts.final_ablations
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.model import load_model, generate_answer, generate_with_template
from src.prompt_templates import compute_prompt_helpers
from src.rag_retriever import RAGRetriever


YES_NO_PREFIXES = (
    "is ", "are ", "do ", "does ", "can ", "has ", "have ", "had ",
    "was ", "were ", "will ", "could ", "should ", "would ", "isn't ", "aren't ",
)


def is_yes_no(question: str) -> bool:
    return question.lower().lstrip().startswith(YES_NO_PREFIXES)


def evidence_aware_select(question: str, retrieved: list, raw_rag: str) -> tuple[str, str]:
    """Demo-style evidence-aware selector.
    Rule 1: exact-question match + img_score >= 0.90 -> retrieved answer
    Rule 2: jaccard(question, retr_q) >= 0.6 + img_score >= 0.90 -> retrieved answer
    Rule 3: majority retrieved answer (>=2 votes) -> majority
    Rule 4: keep raw RAG
    """
    h = compute_prompt_helpers(question, retrieved)
    rqs = h["retrieved_questions"]
    ras = h["retrieved_answers"]
    qn = question.strip().lower()

    # Rule 1: exact match
    for r in retrieved[:3]:
        rq = (r.get("question") or "").strip().lower()
        if rq == qn and r.get("img_score", 0.0) >= 0.90:
            return r.get("best_answer") or r.get("answer", ""), "exact_q_match_img090"

    # Rule 2: high jaccard + high img_score
    best_idx, best_jacc = -1, 0.0
    for i, rq in enumerate(rqs):
        # Quick jaccard on content words
        from src.prompt_templates import _toks  # type: ignore
        a, b = _toks(question), _toks(rq)
        if not a or not b:
            continue
        j = len(a & b) / len(a | b)
        if j > best_jacc:
            best_jacc, best_idx = j, i
    if best_idx >= 0 and best_jacc >= 0.6 and retrieved[best_idx].get("img_score", 0.0) >= 0.90:
        return ras[best_idx], f"jaccard{best_jacc:.2f}_img090"

    # Rule 3: majority (>=2 votes)
    counts = Counter(a for a in ras if a)
    if counts:
        top_a, top_n = counts.most_common(1)[0]
        if top_n >= 2:
            return top_a, "majority>=2"

    return raw_rag, "raw_rag"


def majority_vote(*answers: str, fallback: str = "") -> str:
    """Pick the most common, prefer earlier in tie-break."""
    valid = [a for a in answers if a and a.strip()]
    if not valid:
        return fallback
    cnt = Counter(valid)
    top, n = cnt.most_common(1)[0]
    if n >= 2:
        return top
    return valid[0]   # all unique → fall back to first (alpha=1.0 = headline)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--offset",      type=int, default=5000)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # ── 1. Load models ────────────────────────────────────────────────────────
    print("Loading retriever (CLIP + 20k FAISS)...", flush=True)
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()
    print(f"  KB size: {retriever.image_index.ntotal}")

    print("\nLoading BLIP-2 (4-bit)...", flush=True)
    model, processor = load_model(quantize=True)

    # ── 2. Load eval samples ──────────────────────────────────────────────────
    print(f"\nLoading {args.num_samples} eval samples (offset={args.offset})...", flush=True)
    dataset = load_vqav2(split="validation", num_samples=args.num_samples, offset=args.offset)

    # ── 3. Single-pass cache + generations ────────────────────────────────────
    print("\n=== single-pass: 1 baseline + 7 RAG generations per sample ===", flush=True)
    rows = []
    t0 = time.time()
    for i, item in enumerate(dataset):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]
        qid      = str(item.get("question_id", i))
        img_id   = str(item.get("image_id", ""))

        # Baseline (no retrieval)
        baseline = generate_answer(model, processor, image, question)

        # 7 retrieval configs
        retr_k1   = retriever.retrieve(image, question, top_k=1,  alpha=1.0)
        retr_k3   = retriever.retrieve(image, question, top_k=3,  alpha=1.0)   # headline
        retr_k5   = retriever.retrieve(image, question, top_k=5,  alpha=1.0)
        retr_k10  = retriever.retrieve(image, question, top_k=10, alpha=1.0)
        retr_a05  = retriever.retrieve(image, question, top_k=3,  alpha=0.5)
        retr_a00  = retriever.retrieve(image, question, top_k=3,  alpha=0.0)
        retr_rerk = retriever.retrieve(image, question, top_k=3,  alpha=1.0, rerank=True)

        # 7 generations with current_prompt
        rag_k1,    _ = generate_with_template(model, processor, image, question, retr_k1,   "current_prompt")
        rag_k3,    _ = generate_with_template(model, processor, image, question, retr_k3,   "current_prompt")
        rag_k5,    _ = generate_with_template(model, processor, image, question, retr_k5,   "current_prompt")
        rag_k10,   _ = generate_with_template(model, processor, image, question, retr_k10,  "current_prompt")
        rag_a05,   _ = generate_with_template(model, processor, image, question, retr_a05,  "current_prompt")
        rag_a00,   _ = generate_with_template(model, processor, image, question, retr_a00,  "current_prompt")
        rag_rerk,  _ = generate_with_template(model, processor, image, question, retr_rerk, "current_prompt")

        # Derive the ablations
        # 1. Evidence-aware selector applied to k=3 baseline
        eaw, eaw_reason = evidence_aware_select(question, retr_k3, rag_k3)
        # 2. Type-conditional gate
        type_gated = baseline if is_yes_no(question) else rag_k3
        # 5. Self-consistency vote (α 0/0.5/1.0)
        sc_vote = majority_vote(rag_k3, rag_a05, rag_a00, fallback=rag_k3)

        rows.append({
            "qid": qid, "image_id": img_id, "question": question,
            "gt": "|".join(gt_all[:3]), "gt_all": gt_all,
            "is_yes_no": is_yes_no(question),
            "baseline":  baseline,
            "rag_k1":    rag_k1,
            "rag_k3":    rag_k3,
            "rag_k5":    rag_k5,
            "rag_k10":   rag_k10,
            "rag_a05":   rag_a05,
            "rag_a00":   rag_a00,
            "rag_rerank":rag_rerk,
            "eaw":       eaw,
            "eaw_reason":eaw_reason,
            "type_gated":type_gated,
            "sc_vote":   sc_vote,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dataset) - i - 1)
            print(f"  {i+1}/{len(dataset)}  ({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    # ── 4. Compute accuracies ────────────────────────────────────────────────
    fields = [
        ("baseline",      "Baseline (frozen, no retrieval)"),
        ("rag_k1",        "RAG · α=1.0 · k=1"),
        ("rag_k3",        "★ RAG · α=1.0 · k=3 (headline reproduce)"),
        ("rag_k5",        "RAG · α=1.0 · k=5"),
        ("rag_k10",       "RAG · α=1.0 · k=10"),
        ("rag_a05",       "RAG · α=0.5 · k=3"),
        ("rag_a00",       "RAG · α=0.0 · k=3"),
        ("rag_rerank",    "RAG · α=1.0 · k=3 + cross-encoder rerank"),
        ("eaw",           "RAG + Evidence-Aware Selector"),
        ("type_gated",    "Type-conditional gate (yes/no → baseline)"),
        ("sc_vote",       "Self-consistency vote (α ∈ {0.0, 0.5, 1.0})"),
    ]
    summary = {}
    for col, _ in fields:
        scores = [vqa_accuracy(r[col], r["gt_all"]) for r in rows]
        summary[col] = sum(scores) / len(scores)

    # Per-type breakdown for type-conditional gate analysis
    yn  = [r for r in rows if r["is_yes_no"]]
    op  = [r for r in rows if not r["is_yes_no"]]
    type_breakdown = {}
    for col in ["baseline", "rag_k3", "type_gated"]:
        type_breakdown[col] = {
            "yes_no":    sum(vqa_accuracy(r[col], r["gt_all"]) for r in yn) / max(len(yn), 1),
            "open":      sum(vqa_accuracy(r[col], r["gt_all"]) for r in op) / max(len(op), 1),
            "n_yn":      len(yn),
            "n_open":    len(op),
        }

    # Helped/hurt analysis vs current (rag_k3) for the new ablations
    cur_correct = [vqa_accuracy(r["rag_k3"], r["gt_all"]) >= 1.0 for r in rows]
    def helped_hurt(col):
        new_correct = [vqa_accuracy(r[col], r["gt_all"]) >= 1.0 for r in rows]
        helped = sum(1 for c, n in zip(cur_correct, new_correct) if not c and n)
        hurt   = sum(1 for c, n in zip(cur_correct, new_correct) if c and not n)
        return helped, hurt

    # ── 5. Print + save ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FINAL ABLATIONS — 100-sample eval, 20k diverse KB")
    print("=" * 78)
    print(f"{'configuration':<58} {'acc':>7}  {'Δ vs k=3':>9}  helped/hurt")
    print("-" * 78)
    cur = summary["rag_k3"]
    for col, label in fields:
        a = summary[col] * 100
        d = (summary[col] - cur) * 100
        if col not in ("baseline", "rag_k3"):
            h, ht = helped_hurt(col)
            extra = f"  {h}/{ht}"
        else:
            extra = ""
        print(f"{label:<58} {a:6.2f}%  {d:+8.2f}{extra}")

    print()
    print("Type breakdown (n=27 yes/no, n=73 open-ended):")
    for col in ["baseline", "rag_k3", "type_gated"]:
        b = type_breakdown[col]
        print(f"  {col:<14} yes/no={b['yes_no']*100:6.2f}%   open={b['open']*100:6.2f}%")

    # CSV
    exp = ROOT / "results" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    csv_path = exp / "final_ablations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "accuracy", "delta_vs_k3", "helped_vs_k3", "hurt_vs_k3", "notes"])
        for col, label in fields:
            a = summary[col]
            d = a - summary["rag_k3"]
            if col not in ("baseline", "rag_k3"):
                h, ht = helped_hurt(col)
            else:
                h, ht = "—", "—"
            note = "headline" if col == "rag_k3" else ""
            w.writerow([label, round(a, 4), round(d, 4), h, ht, note])
    print(f"\nSaved {csv_path}")

    # Detail CSV
    detail = exp / "final_ablations_detail.csv"
    with open(detail, "w", newline="", encoding="utf-8") as f:
        out_rows = []
        for r in rows:
            r2 = {k: v for k, v in r.items() if k != "gt_all"}
            out_rows.append(r2)
        cols = list(out_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Saved {detail}")

    # Markdown summary
    md = exp / "final_ablations.md"
    md_lines = []
    md_lines.append("# Final ablations — RAG-VQA")
    md_lines.append("")
    md_lines.append(f"Eval set: {args.num_samples} validation samples (offset={args.offset}) · 20k diverse KB · BLIP-2 frozen, 4-bit · `current_prompt`.")
    md_lines.append("")
    md_lines.append("| Configuration | Accuracy | Δ vs k=3 | Helped | Hurt |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for col, label in fields:
        a = summary[col] * 100
        d = (summary[col] - cur) * 100
        if col not in ("baseline", "rag_k3"):
            h, ht = helped_hurt(col)
        else:
            h, ht = "—", "—"
        flag = " ★" if col == "rag_k3" else ""
        md_lines.append(f"| {label}{flag} | {a:.2f}% | {d:+.2f} | {h} | {ht} |")
    md_lines.append("")
    md_lines.append("## Type breakdown (yes/no vs open-ended)")
    md_lines.append("")
    md_lines.append("| Configuration | Yes/No (n=27) | Open-ended (n=73) |")
    md_lines.append("|---|---:|---:|")
    for col, label in [("baseline", "Baseline"), ("rag_k3", "RAG headline (k=3, α=1.0)"), ("type_gated", "Type-conditional gate")]:
        b = type_breakdown[col]
        md_lines.append(f"| {label} | {b['yes_no']*100:.2f}% | {b['open']*100:.2f}% |")
    md_lines.append("")

    # Pick winner
    sortable = sorted(summary.items(), key=lambda kv: -kv[1])
    winner_col, winner_acc = sortable[0]
    md_lines.append(f"**Winner: {dict(fields)[winner_col]} at {winner_acc*100:.2f}%** (Δ {(winner_acc-cur)*100:+.2f} vs the k=3 headline).")
    md_lines.append("")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {md}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [label for col, label in fields]
        accs  = [summary[col] * 100 for col, _ in fields]
        cols_ord = [col for col, _ in fields]
        baseline_v = summary["baseline"] * 100
        cur_v      = summary["rag_k3"] * 100
        winner_idx = cols_ord.index(winner_col)

        colors = []
        for i, col in enumerate(cols_ord):
            if col == "baseline":     colors.append("#8b949e")
            elif col == "rag_k3":     colors.append("#58a6ff")
            elif col == winner_col:   colors.append("#3fb950")
            else:                     colors.append("#21262d")

        fig, ax = plt.subplots(figsize=(11, 6.5))
        bars = ax.barh(names, accs, color=colors, edgecolor="#444")
        for b, a in zip(bars, accs):
            ax.text(a + 0.4, b.get_y() + b.get_height()/2, f"{a:.2f}%", va="center", fontsize=10)
        ax.axvline(baseline_v, color="#f85149", linestyle="--", alpha=0.7,
                   label=f"baseline {baseline_v:.1f}%")
        ax.axvline(cur_v, color="#58a6ff", linestyle=":", alpha=0.7,
                   label=f"headline RAG k=3 {cur_v:.1f}%")
        ax.set_xlabel("VQA soft accuracy (%)")
        ax.set_title(f"Final ablations — n={args.num_samples}, 20k diverse KB · winner: {winner_col} @ {winner_acc*100:.2f}%")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        ax.set_xlim(0, max(accs) + 8)
        plt.tight_layout()
        plot_path = ROOT / "results" / "plots" / "final_ablations.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\n*** Winner: {winner_col} at {winner_acc*100:.2f}%  (Δ {(winner_acc-cur)*100:+.2f} vs k=3 headline) ***")


if __name__ == "__main__":
    main()
