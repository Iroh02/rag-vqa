"""Final system-layer experiment: 4 improvements stacked.

Improvements (in priority order):
  A. Type-conditional router      — yes/no → baseline; open-ended → RAG
  B. Evidence-aware selector      — rule-based override of raw RAG
  C. Answer normalization         — lowercase + articles + candidate collapse
  D. Question-aware Q&A reranking — α=0.75, candidate_k=50, fusion=0.75·img+0.25·q

Outputs:
  results/experiments/question_aware_rerank.csv
  results/experiments/type_router.csv
  results/experiments/evidence_selector.csv
  results/experiments/answer_normalizer.csv
  results/experiments/final_system_layer.csv     (all stacked)
  results/experiments/final_system_layer.md
  results/plots/final_system_layer.png

Run (with API stopped):
  python -m scripts.final_system_layer
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.answer_normalizer import normalize as norm_answer
from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.model import load_model, generate_answer, generate_with_template
from src.prompt_templates import compute_prompt_helpers
from src.rag_retriever import RAGRetriever


YN_PREFIXES = (
    "is ", "are ", "do ", "does ", "can ", "has ", "have ", "had ",
    "was ", "were ", "will ", "could ", "should ", "would ",
)


def is_yes_no(q: str) -> bool:
    return q.lower().lstrip().startswith(YN_PREFIXES)


def _toks_content(s: str) -> set:
    """Content-word tokens (ignore stopwords)."""
    stop = {"is","the","a","an","this","that","what","where","when","how","why",
            "who","in","on","of","to","at","for","with","do","does","are","there",
            "it","be","was","were","has","have","had","can","could","should",
            "would","will"}
    return {t for t in re.findall(r"[a-z]+", (s or "").lower()) if t not in stop}


def _jaccard(a: str, b: str) -> float:
    A, B = _toks_content(a), _toks_content(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def evidence_aware_select(question, retrieved, raw_rag):
    """Same selector logic as the demo toggle."""
    rqs = [r.get("question", "") for r in retrieved[:3]]
    ras = [(r.get("best_answer") or r.get("answer") or "") for r in retrieved[:3]]
    qn  = question.strip().lower()

    # Rule 1: exact question match + img >= 0.90
    for r in retrieved[:3]:
        rq = (r.get("question") or "").strip().lower()
        if rq == qn and r.get("img_score", 0.0) >= 0.90:
            return (r.get("best_answer") or r.get("answer", "")), "exact_q_match_img090"

    # Rule 2: high jaccard + high img
    best_idx, best_j = -1, 0.0
    for i, rq in enumerate(rqs):
        j = _jaccard(question, rq)
        if j > best_j:
            best_j, best_idx = j, i
    if best_idx >= 0 and best_j >= 0.6 and retrieved[best_idx].get("img_score", 0.0) >= 0.90:
        return ras[best_idx], f"jaccard_{best_j:.2f}_img090"

    # Rule 3: majority (≥2 votes)
    cnt = Counter(a for a in ras if a)
    if cnt:
        top, n = cnt.most_common(1)[0]
        if n >= 2:
            return top, "majority>=2"

    return raw_rag, "raw_rag"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--offset",      type=int, default=5000)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading retriever (CLIP + 20k FAISS)...")
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()
    print(f"  KB size: {retriever.image_index.ntotal}")

    print("\nLoading BLIP-2 (4-bit)...")
    model, processor = load_model(quantize=True)

    # ── eval samples ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.num_samples} samples (offset={args.offset})...")
    dataset = load_vqav2(split="validation", num_samples=args.num_samples, offset=args.offset)

    # ── single-pass: 1 baseline + 2 RAG generations per sample ───────────────
    # (current α=1.0 k=3) and (question-aware α=0.75 candidate_k=50, top_k=3)
    print("\n=== single-pass: baseline + RAG·k=3 + RAG·QA-rerank ===")
    rows = []
    t0 = time.time()
    for i, item in enumerate(dataset):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]
        qid      = str(item.get("question_id", i))
        img_id   = str(item.get("image_id", ""))

        baseline = generate_answer(model, processor, image, question)

        # Retrieval — current headline
        retr_cur = retriever.retrieve(image, question, top_k=3, alpha=1.0, candidate_k=30)

        # Retrieval — question-aware reranked: 0.75*img + 0.25*q over top-50 candidates
        retr_qar = retriever.retrieve(image, question, top_k=3, alpha=0.75, candidate_k=50)

        # Generations using current_prompt
        rag_cur, _ = generate_with_template(model, processor, image, question, retr_cur, "current_prompt")
        rag_qar, _ = generate_with_template(model, processor, image, question, retr_qar, "current_prompt")

        rows.append({
            "qid":      qid,
            "image_id": img_id,
            "question": question,
            "gt_all":   gt_all,
            "is_yn":    is_yes_no(question),
            "baseline": baseline,
            "rag_cur":  rag_cur,
            "rag_qar":  rag_qar,
            "retr_cur": retr_cur,
            "retr_qar": retr_qar,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dataset) - i - 1)
            print(f"  {i+1}/{len(dataset)}  ({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    # ── derive all 4 layers ──────────────────────────────────────────────────
    print("\nDeriving layers...")
    for r in rows:
        # A. Type-conditional router
        r["type_router"] = r["baseline"] if r["is_yn"] else r["rag_cur"]
        r["type_router_qar"] = r["baseline"] if r["is_yn"] else r["rag_qar"]

        # B. Evidence-aware selector — applied to current and to QAR
        r["selector_cur"], r["selector_cur_reason"] = evidence_aware_select(
            r["question"], r["retr_cur"], r["rag_cur"]
        )
        r["selector_qar"], r["selector_qar_reason"] = evidence_aware_select(
            r["question"], r["retr_qar"], r["rag_qar"]
        )

        # candidates from QAR retrieval (used by normalizer)
        helpers_qar = compute_prompt_helpers(r["question"], r["retr_qar"])
        r["candidates_qar"] = helpers_qar["candidate_answers"]
        helpers_cur = compute_prompt_helpers(r["question"], r["retr_cur"])
        r["candidates_cur"] = helpers_cur["candidate_answers"]

        # C. Answer normalization — applied to baseline, current RAG, and QAR RAG
        r["norm_baseline"], r["norm_baseline_reason"] = norm_answer(r["baseline"], r["candidates_cur"])
        r["norm_rag_cur"],  r["norm_rag_cur_reason"]  = norm_answer(r["rag_cur"],  r["candidates_cur"])
        r["norm_rag_qar"],  r["norm_rag_qar_reason"]  = norm_answer(r["rag_qar"],  r["candidates_qar"])

        # Stacked: type_router → selector → normalize (using QAR retrieval)
        stage1 = r["baseline"] if r["is_yn"] else r["rag_qar"]
        if r["is_yn"]:
            stage2 = stage1   # don't apply selector for yes/no — baseline
            stage2_reason = "yn_baseline"
        else:
            stage2, stage2_reason = evidence_aware_select(r["question"], r["retr_qar"], stage1)
        stage3, stage3_reason = norm_answer(stage2, r["candidates_qar"])
        r["stacked"] = stage3
        r["stacked_pipeline"] = f"router={stage1!r} | selector={stage2!r} ({stage2_reason}) | norm={stage3!r} ({stage3_reason})"

    # ── compute accuracies ───────────────────────────────────────────────────
    def acc(col):  return sum(vqa_accuracy(r[col], r["gt_all"]) for r in rows) / len(rows)
    def acc_yn(col):  return sum(vqa_accuracy(r[col], r["gt_all"]) for r in rows if r["is_yn"]) / max(sum(1 for r in rows if r["is_yn"]), 1)
    def acc_op(col):  return sum(vqa_accuracy(r[col], r["gt_all"]) for r in rows if not r["is_yn"]) / max(sum(1 for r in rows if not r["is_yn"]), 1)

    summary = {}
    for col in ["baseline", "rag_cur", "rag_qar",
                "type_router", "type_router_qar",
                "selector_cur", "selector_qar",
                "norm_baseline", "norm_rag_cur", "norm_rag_qar",
                "stacked"]:
        summary[col] = {
            "overall": acc(col),
            "yes_no":  acc_yn(col),
            "open":    acc_op(col),
        }

    # ── print + save ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL SYSTEM LAYER — strict VQA accuracy on 100-sample eval")
    print("=" * 90)
    print(f"{'configuration':<40}{'overall':>10}{'yes/no':>10}{'open':>10}{'Δ vs RAG':>12}")
    print("-" * 90)
    cur = summary["rag_cur"]["overall"]
    label_for = {
        "baseline":         "Baseline (no retrieval)",
        "rag_cur":          "★ RAG α=1.0 k=3 (headline)",
        "rag_qar":          "D. Question-aware rerank (α=0.75, c50)",
        "type_router":      "A. Type-router on RAG",
        "type_router_qar":  "A. Type-router on QAR",
        "selector_cur":     "B. Evidence-aware selector on RAG",
        "selector_qar":     "B. Evidence-aware selector on QAR",
        "norm_baseline":    "C. Normalizer on baseline",
        "norm_rag_cur":     "C. Normalizer on RAG",
        "norm_rag_qar":     "C. Normalizer on QAR",
        "stacked":          "★★ A→B→C on QAR (full stack)",
    }
    for col in ["baseline", "rag_cur", "rag_qar",
                "type_router", "type_router_qar",
                "selector_cur", "selector_qar",
                "norm_baseline", "norm_rag_cur", "norm_rag_qar",
                "stacked"]:
        s = summary[col]
        d = (s["overall"] - cur) * 100
        print(f"{label_for[col]:<40}{s['overall']*100:>9.2f}%{s['yes_no']*100:>9.2f}%{s['open']*100:>9.2f}%{d:>+11.2f}")

    # Per-layer CSVs
    exp = ROOT / "results" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)

    def write_csv(path, header_rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(header_rows)

    # 1. question_aware_rerank.csv
    write_csv(exp / "question_aware_rerank.csv", [
        ["configuration", "overall", "yes_no", "open", "delta_vs_rag", "notes"],
        ["RAG α=1.0 k=3 (headline)",         round(summary["rag_cur"]["overall"], 4),
         round(summary["rag_cur"]["yes_no"], 4), round(summary["rag_cur"]["open"], 4), 0.0,  "control"],
        ["Question-aware rerank α=0.75 c50", round(summary["rag_qar"]["overall"], 4),
         round(summary["rag_qar"]["yes_no"], 4), round(summary["rag_qar"]["open"], 4),
         round(summary["rag_qar"]["overall"] - cur, 4), "0.75·img + 0.25·q over top-50"],
    ])
    print(f"\nSaved {exp/'question_aware_rerank.csv'}")

    # 2. type_router.csv
    write_csv(exp / "type_router.csv", [
        ["configuration", "overall", "yes_no", "open", "delta_vs_rag"],
        ["Baseline",                         round(summary["baseline"]["overall"], 4),
         round(summary["baseline"]["yes_no"], 4), round(summary["baseline"]["open"], 4), round(summary["baseline"]["overall"]-cur, 4)],
        ["RAG α=1.0 k=3",                    round(summary["rag_cur"]["overall"], 4),
         round(summary["rag_cur"]["yes_no"], 4), round(summary["rag_cur"]["open"], 4),  0.0],
        ["A. Type-router (yn→base, open→RAG)", round(summary["type_router"]["overall"], 4),
         round(summary["type_router"]["yes_no"], 4), round(summary["type_router"]["open"], 4),
         round(summary["type_router"]["overall"] - cur, 4)],
    ])
    print(f"Saved {exp/'type_router.csv'}")

    # 3. evidence_selector.csv
    write_csv(exp / "evidence_selector.csv", [
        ["configuration", "overall", "yes_no", "open", "delta_vs_rag", "notes"],
        ["RAG α=1.0 k=3",                          round(summary["rag_cur"]["overall"], 4),
         round(summary["rag_cur"]["yes_no"], 4), round(summary["rag_cur"]["open"], 4), 0.0, "control"],
        ["B. Evidence-aware selector on RAG",      round(summary["selector_cur"]["overall"], 4),
         round(summary["selector_cur"]["yes_no"], 4), round(summary["selector_cur"]["open"], 4),
         round(summary["selector_cur"]["overall"] - cur, 4), "exact_match → high_jaccard → majority → raw"],
        ["B. Evidence-aware selector on QAR",      round(summary["selector_qar"]["overall"], 4),
         round(summary["selector_qar"]["yes_no"], 4), round(summary["selector_qar"]["open"], 4),
         round(summary["selector_qar"]["overall"] - cur, 4), "selector applied to question-aware retrieval"],
    ])
    print(f"Saved {exp/'evidence_selector.csv'}")

    # 4. answer_normalizer.csv
    write_csv(exp / "answer_normalizer.csv", [
        ["configuration", "overall", "yes_no", "open", "delta_vs_rag", "notes"],
        ["Baseline (raw)",                round(summary["baseline"]["overall"], 4),
         round(summary["baseline"]["yes_no"], 4), round(summary["baseline"]["open"], 4),
         round(summary["baseline"]["overall"] - cur, 4), "raw"],
        ["C. Normalizer on baseline",     round(summary["norm_baseline"]["overall"], 4),
         round(summary["norm_baseline"]["yes_no"], 4), round(summary["norm_baseline"]["open"], 4),
         round(summary["norm_baseline"]["overall"] - cur, 4), "lowercase + articles + candidate collapse"],
        ["RAG α=1.0 k=3 (raw)",           round(summary["rag_cur"]["overall"], 4),
         round(summary["rag_cur"]["yes_no"], 4), round(summary["rag_cur"]["open"], 4), 0.0, "control"],
        ["C. Normalizer on RAG",          round(summary["norm_rag_cur"]["overall"], 4),
         round(summary["norm_rag_cur"]["yes_no"], 4), round(summary["norm_rag_cur"]["open"], 4),
         round(summary["norm_rag_cur"]["overall"] - cur, 4), ""],
        ["C. Normalizer on QAR",          round(summary["norm_rag_qar"]["overall"], 4),
         round(summary["norm_rag_qar"]["yes_no"], 4), round(summary["norm_rag_qar"]["open"], 4),
         round(summary["norm_rag_qar"]["overall"] - cur, 4), ""],
    ])
    print(f"Saved {exp/'answer_normalizer.csv'}")

    # 5. final_system_layer.csv (full stack)
    write_csv(exp / "final_system_layer.csv", [
        ["configuration", "overall", "yes_no", "open", "delta_vs_rag", "notes"],
        ["Baseline",                                        round(summary["baseline"]["overall"], 4),
         round(summary["baseline"]["yes_no"], 4), round(summary["baseline"]["open"], 4),
         round(summary["baseline"]["overall"] - cur, 4), ""],
        ["RAG α=1.0 k=3 (headline)",                        round(summary["rag_cur"]["overall"], 4),
         round(summary["rag_cur"]["yes_no"], 4), round(summary["rag_cur"]["open"], 4), 0.0, "headline"],
        ["+ A. Type-router (yn → baseline)",                round(summary["type_router"]["overall"], 4),
         round(summary["type_router"]["yes_no"], 4), round(summary["type_router"]["open"], 4),
         round(summary["type_router"]["overall"] - cur, 4), ""],
        ["+ B. Evidence-aware selector",                    round(summary["selector_cur"]["overall"], 4),
         round(summary["selector_cur"]["yes_no"], 4), round(summary["selector_cur"]["open"], 4),
         round(summary["selector_cur"]["overall"] - cur, 4), ""],
        ["+ C. Answer normalizer (on RAG)",                 round(summary["norm_rag_cur"]["overall"], 4),
         round(summary["norm_rag_cur"]["yes_no"], 4), round(summary["norm_rag_cur"]["open"], 4),
         round(summary["norm_rag_cur"]["overall"] - cur, 4), ""],
        ["+ D. Question-aware rerank (replaces retrieval)", round(summary["rag_qar"]["overall"], 4),
         round(summary["rag_qar"]["yes_no"], 4), round(summary["rag_qar"]["open"], 4),
         round(summary["rag_qar"]["overall"] - cur, 4), ""],
        ["★★ Stacked: D → A → B → C",                       round(summary["stacked"]["overall"], 4),
         round(summary["stacked"]["yes_no"], 4), round(summary["stacked"]["open"], 4),
         round(summary["stacked"]["overall"] - cur, 4), "full final-system layer"],
    ])
    print(f"Saved {exp/'final_system_layer.csv'}")

    # detail CSV — one row per sample
    detail_path = exp / "final_system_layer_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        cols = ["qid", "image_id", "question", "is_yn", "gt", "baseline", "rag_cur", "rag_qar",
                "type_router", "selector_cur", "norm_rag_cur", "stacked",
                "candidates_cur", "candidates_qar",
                "selector_cur_reason", "norm_rag_cur_reason", "stacked_pipeline"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "qid":          r["qid"],
                "image_id":     r["image_id"],
                "question":     r["question"],
                "is_yn":        r["is_yn"],
                "gt":           "|".join(r["gt_all"][:3]),
                "baseline":     r["baseline"],
                "rag_cur":      r["rag_cur"],
                "rag_qar":      r["rag_qar"],
                "type_router":  r["type_router"],
                "selector_cur": r["selector_cur"],
                "norm_rag_cur": r["norm_rag_cur"],
                "stacked":      r["stacked"],
                "candidates_cur": ", ".join(r["candidates_cur"]),
                "candidates_qar": ", ".join(r["candidates_qar"]),
                "selector_cur_reason": r["selector_cur_reason"],
                "norm_rag_cur_reason": r["norm_rag_cur_reason"],
                "stacked_pipeline":    r["stacked_pipeline"],
            })
    print(f"Saved {detail_path}")

    # Markdown summary
    md = exp / "final_system_layer.md"
    md_lines = ["# Final system layer — RAG-VQA improvements"]
    md_lines.append("")
    md_lines.append(f"100-sample eval (offset={args.offset}) · 20k diverse KB · BLIP-2 frozen, 4-bit · `current_prompt`.")
    md_lines.append("")
    md_lines.append("Same eval set, no new training, no KB rebuild. Strict VQA accuracy.")
    md_lines.append("")
    md_lines.append("| Configuration | Overall | Yes/No | Open | Δ vs RAG |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for col in ["baseline", "rag_cur", "rag_qar",
                "type_router", "type_router_qar",
                "selector_cur", "selector_qar",
                "norm_baseline", "norm_rag_cur", "norm_rag_qar",
                "stacked"]:
        s = summary[col]
        d = (s["overall"] - cur) * 100
        flag = " ★" if col == "rag_cur" else (" ★★" if col == "stacked" else "")
        md_lines.append(f"| {label_for[col]}{flag} | {s['overall']*100:.2f}% | {s['yes_no']*100:.2f}% | {s['open']*100:.2f}% | {d:+.2f} |")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {md}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = ["baseline", "rag_cur", "rag_qar", "type_router", "selector_cur",
                 "norm_rag_cur", "stacked"]
        names = [label_for[c] for c in order]
        accs  = [summary[c]["overall"] * 100 for c in order]
        colors = []
        for c in order:
            if c == "baseline":          colors.append("#8b949e")
            elif c == "rag_cur":         colors.append("#58a6ff")
            elif c == "stacked":         colors.append("#3fb950")
            else:                        colors.append("#21262d")

        fig, ax = plt.subplots(figsize=(11, 5.5))
        bars = ax.barh(names, accs, color=colors, edgecolor="#444")
        for b, a in zip(bars, accs):
            ax.text(a + 0.4, b.get_y() + b.get_height()/2, f"{a:.2f}%", va="center", fontsize=10)
        ax.axvline(summary["baseline"]["overall"]*100, color="#f85149", linestyle="--", alpha=0.7, label=f"baseline {summary['baseline']['overall']*100:.1f}%")
        ax.axvline(cur*100, color="#58a6ff", linestyle=":", alpha=0.7, label=f"RAG headline {cur*100:.1f}%")
        ax.set_xlabel("VQA strict accuracy (%)")
        ax.set_title(f"Final system layer · n={args.num_samples} · 20k diverse KB")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        ax.set_xlim(0, max(accs) + 8)
        plt.tight_layout()
        plot = ROOT / "results" / "plots" / "final_system_layer.png"
        plt.savefig(plot, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\n*** Stacked accuracy: {summary['stacked']['overall']*100:.2f}%  "
          f"(Δ {(summary['stacked']['overall']-cur)*100:+.2f} vs RAG headline) ***")


if __name__ == "__main__":
    main()
