"""
Offline analysis of RAG-VQA results using cached prediction/eval JSON files.
Requires NO internet — works entirely from results/*.json.

Produces:
  results/experiments/oracle_summary.csv
  results/experiments/answer_type.csv
  results/experiments/helpful_harmful.csv
  results/experiments/confidence_buckets.csv
  results/experiments/alpha_sweep.csv
  results/plots/oracle.png
  results/plots/answer_type.png
  results/plots/helpful_harmful.png
  results/plots/confidence.png
  results/plots/alpha_sweep.png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
EXP_DIR = RESULTS_DIR / "experiments"
PLOTS_DIR = RESULTS_DIR / "plots"
EXP_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(s):
    import re
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())

def vqa_acc(pred, gt_list):
    p = normalize(pred)
    matches = sum(1 for g in gt_list if normalize(g) == p)
    return min(1.0, matches / 3.0)

def _write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path}")

# ── load cached data ──────────────────────────────────────────────────────────

def load_eval(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def load_debug(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# =============================================================================
# 1. ALPHA / TAU SWEEP
# =============================================================================

def run_alpha_sweep():
    print("\n[alpha_sweep]")
    configs = [
        ("baseline", "eval_results_baseline.json", 0.0, "N/A"),
        ("a=0.0,t=0.0", "eval_results_rag_a00_t00.json", 0.0, 0.0),
        ("a=0.5,t=0.0", "eval_results_rag_a05_t00.json", 0.5, 0.0),
        ("a=0.5,t=0.3", "eval_results_rag_a05_t03.json", 0.5, 0.3),
        ("a=0.5,t=0.5", "eval_results_rag_a05_t05.json", 0.5, 0.5),
        ("a=1.0,t=0.0", "eval_results_rag_a10_t00.json", 1.0, 0.0),
        ("a=1.0,t=0.5", "eval_results_rag_a10_t05.json", 1.0, 0.5),
        ("a=1.0,t=0.75","eval_results_rag_a10_t075.json",1.0, 0.75),
    ]

    rows = []
    for label, fname, alpha, tau in configs:
        r = load_eval(fname)
        if r is None:
            continue
        acc = r["overall_accuracy"]
        rows.append({"label": label, "alpha": alpha, "tau": tau, "accuracy": acc})
        print(f"  {label}: {acc*100:.2f}%")

    _write_csv(EXP_DIR / "alpha_sweep.csv", rows)

    # Plot: accuracy vs alpha (group by tau)
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_acc = next(r["accuracy"] for r in rows if r["label"] == "baseline")
    ax.axhline(baseline_acc, color="gray", linestyle="--", label=f"Baseline {baseline_acc*100:.1f}%")

    rag_rows = [r for r in rows if r["label"] != "baseline"]
    alphas = sorted(set(r["alpha"] for r in rag_rows))
    taus = sorted(set(r["tau"] for r in rag_rows))
    markers = ["o", "s", "^", "D"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, tau in enumerate(taus):
        pts = sorted([r for r in rag_rows if r["tau"] == tau], key=lambda x: x["alpha"])
        xs = [p["alpha"] for p in pts]
        ys = [p["accuracy"] * 100 for p in pts]
        ax.plot(xs, ys, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], label=f"τ={tau}")

    ax.set_xlabel("Alpha (image retrieval weight)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RAG Accuracy vs. Alpha / Tau")
    ax.legend()
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "alpha_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  saved {PLOTS_DIR}/alpha_sweep.png")
    return rows


# =============================================================================
# 2. ORACLE @ k
# =============================================================================

def run_oracle():
    print("\n[oracle@k]")
    debug = load_debug("debug_context_rag_a10_t00.json")
    baseline_eval = load_eval("eval_results_baseline.json")
    if debug is None or baseline_eval is None:
        print("  missing files, skipping")
        return

    # Build ground_truth lookup by question_id
    gt_lookup = {d["question_id"]: d["ground_truth"]
                 for d in baseline_eval["detailed_results"]}

    k_values = [1, 3, 5]
    oracle_scores = {k: [] for k in k_values}
    detail_rows = []

    for item in debug:
        qid = item["question_id"]
        gt = gt_lookup.get(qid, [])
        if not gt:
            continue

        retrieved = item.get("retrieved", [])
        retrieved_answers = [ex.get("answer", "") for ex in retrieved]
        retrieved_scores  = [ex.get("img_score", ex.get("score", 0.0)) for ex in retrieved]

        row = {
            "question_id": qid,
            "question": item["question"],
            "ground_truth": " | ".join(gt),
            "retrieved_answers": " | ".join(retrieved_answers),
            "retrieved_scores": " | ".join(f"{s:.4f}" for s in retrieved_scores),
        }
        for k in k_values:
            top_k = retrieved_answers[:k]
            score = max((vqa_acc(a, gt) for a in top_k), default=0.0)
            oracle_scores[k].append(score)
            row[f"oracle@{k}"] = score
        detail_rows.append(row)

    oracle = {k: float(np.mean(oracle_scores[k])) for k in k_values}
    for k, v in oracle.items():
        print(f"  Oracle@{k}: {v*100:.2f}%")

    summary_rows = [{"k": k, "oracle_acc": oracle[k]} for k in k_values]
    _write_csv(EXP_DIR / "oracle_summary.csv", summary_rows)
    _write_csv(EXP_DIR / "oracle_detail.csv", detail_rows)

    # Plot
    baseline_acc = load_eval("eval_results_baseline.json")["overall_accuracy"]
    rag_acc      = load_eval("eval_results_rag_a10_t00.json")["overall_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ks = list(k_values)
    accs = [oracle[k] * 100 for k in ks]
    ax.bar(ks, accs, color="#2196F3", width=0.6, label="Oracle@k")
    ax.axhline(baseline_acc * 100, color="gray", linestyle="--",
               label=f"Baseline {baseline_acc*100:.1f}%")
    ax.axhline(rag_acc * 100, color="#FF9800", linestyle="--",
               label=f"RAG {rag_acc*100:.1f}%")
    ax.set_xlabel("k (top-k retrieved)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Retrieval Oracle@k vs. Baseline & RAG")
    ax.legend()
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "oracle.png", dpi=150)
    plt.close(fig)
    print(f"  saved {PLOTS_DIR}/oracle.png")
    return oracle


# =============================================================================
# 3. ANSWER-TYPE BREAKDOWN (yes/no vs. open-ended)
# =============================================================================

def run_answer_type():
    print("\n[answer_type]")
    YES_NO_STARTS = {"is ", "are ", "do ", "does ", "can ", "has ", "have ",
                     "was ", "were ", "will ", "could ", "should ", "would "}

    def is_yesno(qt):
        return any(qt.strip().lower().startswith(p) for p in YES_NO_STARTS)

    baseline_eval = load_eval("eval_results_baseline.json")
    rag_eval      = load_eval("eval_results_rag_a10_t00.json")
    if baseline_eval is None or rag_eval is None:
        print("  missing files, skipping")
        return

    rag_lookup = {d["question_id"]: d
                  for d in rag_eval["detailed_results"]}

    buckets = {"yes/no": {"baseline": [], "rag": []},
               "open":   {"baseline": [], "rag": []}}

    rows = []
    for bd in baseline_eval["detailed_results"]:
        qid = bd["question_id"]
        rd  = rag_lookup.get(qid)
        if rd is None:
            continue
        qt   = bd.get("question_type", "")
        kind = "yes/no" if is_yesno(qt) else "open"
        b_acc = bd["accuracy"]
        r_acc = rd["accuracy"]
        buckets[kind]["baseline"].append(b_acc)
        buckets[kind]["rag"].append(r_acc)
        rows.append({
            "question_id": qid,
            "question": bd["question"],
            "type": kind,
            "question_type": qt,
            "baseline_acc": b_acc,
            "rag_acc": r_acc,
        })

    summary = []
    for kind, data in buckets.items():
        b = np.mean(data["baseline"]) if data["baseline"] else 0.0
        r = np.mean(data["rag"]) if data["rag"] else 0.0
        n = len(data["baseline"])
        summary.append({"type": kind, "baseline": b, "rag": r, "n": n})
        print(f"  {kind} (n={n}): baseline={b*100:.1f}% rag={r*100:.1f}%")

    _write_csv(EXP_DIR / "answer_type.csv", rows)
    _write_csv(EXP_DIR / "answer_type_summary.csv", summary)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w/2, [s["baseline"]*100 for s in summary], w,
           label="Baseline", color="#90A4AE")
    ax.bar(x + w/2, [s["rag"]*100 for s in summary], w,
           label="RAG (α=1.0)", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['type']}\n(n={s['n']})" for s in summary])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "answer_type.png", dpi=150)
    plt.close(fig)
    print(f"  saved {PLOTS_DIR}/answer_type.png")
    return summary


# =============================================================================
# 4. HELPFUL / HARMFUL / NEUTRAL
# =============================================================================

def run_helpful_harmful():
    print("\n[helpful_harmful]")
    baseline_eval = load_eval("eval_results_baseline.json")
    rag_eval      = load_eval("eval_results_rag_a10_t00.json")
    if baseline_eval is None or rag_eval is None:
        print("  missing files, skipping")
        return

    rag_lookup = {d["question_id"]: d for d in rag_eval["detailed_results"]}

    helpful = harmful = neutral = 0
    rows = []
    for bd in baseline_eval["detailed_results"]:
        qid   = bd["question_id"]
        rd    = rag_lookup.get(qid)
        if rd is None:
            continue
        b_acc = bd["accuracy"]
        r_acc = rd["accuracy"]
        delta = r_acc - b_acc
        if delta > 0:
            verdict = "helpful"
            helpful += 1
        elif delta < 0:
            verdict = "harmful"
            harmful += 1
        else:
            verdict = "neutral"
            neutral += 1
        rows.append({
            "question_id": qid,
            "question": bd["question"],
            "ground_truth": " | ".join(bd["ground_truth"]),
            "baseline_pred": bd["predicted"],
            "rag_pred": rd["predicted"],
            "baseline_acc": b_acc,
            "rag_acc": r_acc,
            "delta": delta,
            "verdict": verdict,
        })

    total = helpful + harmful + neutral
    print(f"  helpful={helpful} ({helpful/total*100:.1f}%)  "
          f"harmful={harmful} ({harmful/total*100:.1f}%)  "
          f"neutral={neutral} ({neutral/total*100:.1f}%)")

    _write_csv(EXP_DIR / "helpful_harmful.csv", rows)

    # Plot
    labels = ["Helpful", "Harmful", "Neutral"]
    counts = [helpful, harmful, neutral]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{cnt} ({cnt/total*100:.0f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of samples")
    ax.set_title("RAG Effect: Helpful / Harmful / Neutral vs Baseline")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "helpful_harmful.png", dpi=150)
    plt.close(fig)
    print(f"  saved {PLOTS_DIR}/helpful_harmful.png")
    return {"helpful": helpful, "harmful": harmful, "neutral": neutral, "total": total}


# =============================================================================
# 5. CONFIDENCE VS ACCURACY  (top1 image retrieval score as proxy)
# =============================================================================

def run_confidence():
    print("\n[confidence_vs_accuracy]")
    debug = load_debug("debug_context_rag_a10_t00.json")
    rag_eval = load_eval("eval_results_rag_a10_t00.json")
    if debug is None or rag_eval is None:
        print("  missing files, skipping")
        return

    rag_lookup = {d["question_id"]: d["accuracy"] for d in rag_eval["detailed_results"]}

    rows = []
    for item in debug:
        qid = item["question_id"]
        acc = rag_lookup.get(qid)
        if acc is None:
            continue
        retrieved = item.get("retrieved", [])
        scores = [ex.get("img_score", ex.get("score", 0.0)) for ex in retrieved]
        top1   = scores[0] if scores else 0.0
        margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
        rows.append({
            "question_id": qid,
            "question": item["question"],
            "top1_score": top1,
            "margin": margin,
            "rag_acc": acc,
        })

    if not rows:
        print("  no data")
        return

    _write_csv(EXP_DIR / "confidence_detail.csv", rows)

    # Bucket by top1_score
    n_buckets = 5
    top1s  = np.array([r["top1_score"] for r in rows])
    accs   = np.array([r["rag_acc"]    for r in rows])
    margins = np.array([r["margin"]    for r in rows])

    def bucket_stats(vals, accs, n):
        edges = np.linspace(vals.min(), vals.max() + 1e-9, n + 1)
        bdata = []
        for i in range(n):
            mask = (vals >= edges[i]) & (vals < edges[i+1])
            if mask.sum() == 0:
                continue
            bdata.append({
                "bucket": f"{edges[i]:.2f}–{edges[i+1]:.2f}",
                "n": int(mask.sum()),
                "mean_acc": float(accs[mask].mean()),
            })
        return bdata

    score_buckets  = bucket_stats(top1s, accs, n_buckets)
    margin_buckets = bucket_stats(margins, accs, n_buckets)

    _write_csv(EXP_DIR / "confidence_score_buckets.csv", score_buckets)
    _write_csv(EXP_DIR / "confidence_margin_buckets.csv", margin_buckets)
    print(f"  score range: [{top1s.min():.3f}, {top1s.max():.3f}]")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, bdata, xlabel, title in [
        (axes[0], score_buckets,  "Top-1 Retrieval Score",  "Accuracy by Top-1 Score"),
        (axes[1], margin_buckets, "Top-1 − Top-2 Margin",   "Accuracy by Score Margin"),
    ]:
        if not bdata:
            continue
        xs  = range(len(bdata))
        ns  = [b["n"] for b in bdata]
        ys  = [b["mean_acc"] * 100 for b in bdata]
        labs = [f"{b['bucket']}\n(n={b['n']})" for b in bdata]
        ax.bar(xs, ys, color="#2196F3", width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence.png", dpi=150)
    plt.close(fig)
    print(f"  saved {PLOTS_DIR}/confidence.png")
    return {"score_buckets": score_buckets, "margin_buckets": margin_buckets}


# =============================================================================
# MAIN
# =============================================================================

def run_all():
    print("=" * 60)
    print("Offline RAG-VQA Analysis")
    print("=" * 60)
    alpha_res  = run_alpha_sweep()
    oracle_res = run_oracle()
    type_res   = run_answer_type()
    hh_res     = run_helpful_harmful()
    conf_res   = run_confidence()

    print("\n" + "=" * 60)
    print("SUMMARY TABLE (for slides)")
    print("=" * 60)
    print(f"  {'Method':<25} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for r in alpha_res or []:
        mark = " *" if r["label"] != "baseline" and r["accuracy"] == max(
            x["accuracy"] for x in (alpha_res or []) if x["label"] != "baseline"
        ) else ""
        print(f"  {r['label']:<25} {r['accuracy']*100:>9.2f}%{mark}")
    if oracle_res:
        print()
        for k, v in oracle_res.items():
            print(f"  Oracle@{k:<21} {v*100:>9.2f}%")
    if hh_res:
        total = hh_res["total"]
        print()
        print(f"  Helpful: {hh_res['helpful']}/{total} ({hh_res['helpful']/total*100:.0f}%)")
        print(f"  Harmful: {hh_res['harmful']}/{total} ({hh_res['harmful']/total*100:.0f}%)")
        print(f"  Neutral: {hh_res['neutral']}/{total} ({hh_res['neutral']/total*100:.0f}%)")


if __name__ == "__main__":
    run_all()
