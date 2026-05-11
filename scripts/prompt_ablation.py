"""Prompt ablation: evaluate 9 RAG prompt templates on the same 100 eval samples
against the 20k diverse KB.

Pipeline:
  1. Stop the API (we need the GPU/model) — handled outside the script.
  2. Load 100 validation samples (offset=5000) — same as headline eval.
  3. For each sample: cache baseline + caption + retrieval (one BLIP-2 + retrieval pass).
  4. For each template × each sample: generate answer with that template's prompt.
  5. Score, save CSVs + summary.md + plot.

Outputs:
  results/experiments/prompt_ablation.csv          (one row per template)
  results/experiments/prompt_ablation_detail.csv   (one row per sample × template)
  results/experiments/prompt_ablation_summary.md
  results/plots/prompt_ablation.png

Run:
  python -m scripts.prompt_ablation                # default 100 samples, all 9 templates
  python -m scripts.prompt_ablation --num_samples 30 --skip caption_aware
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAG_INDEX_DIR
from src.dataset import load_vqav2
from src.evaluate import vqa_accuracy
from src.model import (
    load_model, generate_answer, generate_caption,
    generate_with_template,
)
from src.prompt_templates import TEMPLATES, build_rag_prompt, compute_prompt_helpers
from src.rag_retriever import RAGRetriever


def estimate_tokens(s: str) -> int:
    """Rough BPE token estimate. Good enough for a summary stat."""
    return max(1, len(s) // 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--top_k",       type=int, default=3)
    p.add_argument("--alpha",     type=float, default=1.0)
    p.add_argument("--offset",    type=int,   default=5000)
    p.add_argument("--skip",      nargs="*",  default=[],
                   help="Template names to skip (e.g. 'caption_aware' to save time)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    templates_to_run = [t for t in TEMPLATES if t not in args.skip]
    print(f"Templates: {templates_to_run}")
    print(f"Skipped:   {args.skip}")
    print()

    # ── 1. Load retriever + BLIP-2 ────────────────────────────────────────────
    print("Loading retriever (CLIP + 20k FAISS)...", flush=True)
    retriever = RAGRetriever(device="cpu")
    retriever.load_clip()
    retriever.load_index()
    kb_size = retriever.image_index.ntotal
    print(f"  KB size: {kb_size}", flush=True)

    print("\nLoading BLIP-2 (4-bit)...", flush=True)
    model, processor = load_model(quantize=True)

    # ── 2. Load eval samples ──────────────────────────────────────────────────
    print(f"\nLoading {args.num_samples} eval samples (offset={args.offset})...", flush=True)
    dataset = load_vqav2(split="validation", num_samples=args.num_samples, offset=args.offset)

    # ── 3. Cache pass: baseline + caption + retrieval per sample ──────────────
    print(f"\n=== CACHE PASS ({len(dataset)} samples) ===", flush=True)
    cache = []
    t_cache_start = time.time()
    need_caption = "caption_aware" in templates_to_run
    for i, item in enumerate(dataset):
        image    = item["image"].convert("RGB")
        question = item["question"]
        gt_all   = [a["answer"] for a in item["answers"]]
        qid      = str(item.get("question_id", i))
        img_id   = str(item.get("image_id", ""))

        baseline = generate_answer(model, processor, image, question)
        caption  = generate_caption(model, processor, image) if need_caption else ""
        retrieved = retriever.retrieve(image, question, top_k=args.top_k, alpha=args.alpha)

        cache.append({
            "image": image, "question": question, "gt_all": gt_all,
            "qid": qid, "image_id": img_id,
            "baseline": baseline, "caption": caption, "retrieved": retrieved,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_cache_start
            eta = elapsed / (i + 1) * (len(dataset) - i - 1)
            print(f"  cached {i+1}/{len(dataset)}  ({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    print(f"Cache done in {time.time() - t_cache_start:.0f}s", flush=True)

    # Baseline accuracy (sanity check)
    base_acc = sum(vqa_accuracy(c["baseline"], c["gt_all"]) for c in cache) / len(cache)
    print(f"\nBaseline VQA accuracy: {base_acc*100:.2f}%  (expected ~42.33%)", flush=True)

    # ── 4. Per-template generation ────────────────────────────────────────────
    print(f"\n=== TEMPLATE PASS ({len(templates_to_run)} templates × {len(cache)} samples) ===", flush=True)
    template_results = {}        # template_name -> list of dict per sample
    template_summary = {}        # template_name -> summary stats

    for t_idx, tmpl in enumerate(templates_to_run):
        print(f"\n[{t_idx+1}/{len(templates_to_run)}] {tmpl}", flush=True)
        t_start = time.time()
        rows = []
        scores = []
        prompt_chars = []
        prompt_tokens = []

        for c in cache:
            cap = c["caption"] if tmpl == "caption_aware" else None
            try:
                ans, prompt = generate_with_template(
                    model, processor, c["image"], c["question"], c["retrieved"],
                    tmpl, caption=cap, max_new_tokens=10,
                )
            except Exception as e:
                ans, prompt = "", build_rag_prompt(tmpl, c["question"], c["retrieved"], caption=cap)
                print(f"  [{c['qid']}] FAIL: {e}", flush=True)

            score = vqa_accuracy(ans, c["gt_all"])
            scores.append(score)
            prompt_chars.append(len(prompt))
            prompt_tokens.append(estimate_tokens(prompt))

            helpers = compute_prompt_helpers(c["question"], c["retrieved"])
            rows.append({
                "template_name": tmpl,
                "question_id":   c["qid"],
                "image_id":      c["image_id"],
                "question":      c["question"],
                "ground_truth":  "|".join(c["gt_all"][:3]),
                "prediction":    ans,
                "vqa_score":     round(score, 4),
                "prompt":        prompt,
                "retrieved_questions": " || ".join(helpers["retrieved_questions"]),
                "retrieved_answers":   " || ".join(helpers["retrieved_answers"]),
                "candidate_answers":   helpers["candidate_answers_str"],
                "best_question_matched_answer":   helpers["best_question_matched_answer"],
                "best_question_match_score":      helpers["best_question_match_score"],
                "prompt_chars":   len(prompt),
            })

        acc = sum(scores) / len(scores)
        elapsed = time.time() - t_start
        template_results[tmpl] = rows
        template_summary[tmpl] = {
            "accuracy": acc,
            "num_eval": len(scores),
            "avg_prompt_chars":  sum(prompt_chars)  / len(prompt_chars),
            "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
            "elapsed_sec": elapsed,
        }
        print(f"  accuracy: {acc*100:.2f}%  avg_chars: {sum(prompt_chars)/len(prompt_chars):.0f}  ({elapsed:.0f}s)", flush=True)

    # ── 5. Helpful/harmful vs current_prompt ──────────────────────────────────
    if "current_prompt" in template_results:
        cur_scores = [r["vqa_score"] for r in template_results["current_prompt"]]
        for tmpl, rows in template_results.items():
            if tmpl == "current_prompt":
                template_summary[tmpl]["helped"] = template_summary[tmpl]["hurt"] = template_summary[tmpl]["neutral"] = "—"
                continue
            new_scores = [r["vqa_score"] for r in rows]
            helped = hurt = neutral = 0
            for cur, new in zip(cur_scores, new_scores):
                cur_ok = cur >= 1.0
                new_ok = new >= 1.0
                if not cur_ok and new_ok:   helped += 1
                elif cur_ok and not new_ok: hurt += 1
                else:                        neutral += 1
            template_summary[tmpl]["helped"]  = helped
            template_summary[tmpl]["hurt"]    = hurt
            template_summary[tmpl]["neutral"] = neutral

    # ── 6. Pick winner ────────────────────────────────────────────────────────
    sortable = [(t, s["accuracy"], s["avg_prompt_chars"]) for t, s in template_summary.items()]
    sortable.sort(key=lambda x: (-x[1], x[2]))   # highest acc, then shortest prompt
    winner = sortable[0][0]
    cur_acc = template_summary.get("current_prompt", {}).get("accuracy", 0.0)

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    exp_dir = ROOT / "results" / "experiments"
    plots_dir = ROOT / "results" / "plots"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    sum_path = exp_dir / "prompt_ablation.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["template_name", "accuracy", "delta_vs_baseline", "delta_vs_current_rag",
                    "num_eval", "top_k", "alpha",
                    "avg_prompt_chars", "avg_prompt_tokens",
                    "helped_vs_current", "hurt_vs_current", "neutral_vs_current",
                    "elapsed_sec", "notes"])
        for tmpl, s in template_summary.items():
            note = ""
            if tmpl == winner: note = "★ best"
            if tmpl == "current_prompt": note = "control"
            w.writerow([
                tmpl, round(s["accuracy"], 4),
                round(s["accuracy"] - base_acc, 4),
                round(s["accuracy"] - cur_acc, 4) if cur_acc else 0,
                s["num_eval"], args.top_k, args.alpha,
                round(s["avg_prompt_chars"], 1),
                round(s["avg_prompt_tokens"], 1),
                s.get("helped", "—"), s.get("hurt", "—"), s.get("neutral", "—"),
                round(s["elapsed_sec"], 1),
                note,
            ])
    print(f"\nSaved {sum_path}")

    # Detail CSV
    detail_path = exp_dir / "prompt_ablation_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        if not template_results:
            f.write("")
        else:
            cols = list(next(iter(template_results.values()))[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for rows in template_results.values():
                for r in rows:
                    # truncate prompt to keep CSV manageable
                    rcp = dict(r)
                    rcp["prompt"] = rcp["prompt"][:800]
                    w.writerow(rcp)
    print(f"Saved {detail_path}")

    # Markdown summary
    md = ROOT / "results" / "experiments" / "prompt_ablation_summary.md"
    md_lines = []
    md_lines.append("# Prompt Ablation — RAG-VQA")
    md_lines.append("")
    md_lines.append(f"Eval set: {args.num_samples} validation samples (offset={args.offset}) · "
                    f"top_k={args.top_k} · α={args.alpha} · 20k diverse KB · BLIP-2 frozen, 4-bit.")
    md_lines.append("")
    md_lines.append(f"**Baseline (no retrieval): {base_acc*100:.2f}%**")
    md_lines.append("")
    md_lines.append("| Template | Accuracy | Δ vs baseline | Δ vs current | Helped | Hurt | Neutral | avg chars | avg tokens |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tmpl, _, _ in sortable:
        s = template_summary[tmpl]
        d_b = (s["accuracy"] - base_acc) * 100
        d_c = (s["accuracy"] - cur_acc) * 100 if cur_acc else 0.0
        flag = " ★" if tmpl == winner else (" (control)" if tmpl == "current_prompt" else "")
        md_lines.append(
            f"| {tmpl}{flag} | {s['accuracy']*100:.2f}% | {d_b:+.2f} | {d_c:+.2f} | "
            f"{s.get('helped','—')} | {s.get('hurt','—')} | {s.get('neutral','—')} | "
            f"{s['avg_prompt_chars']:.0f} | {s['avg_prompt_tokens']:.0f} |"
        )
    md_lines.append("")

    md_lines.append(f"**Winner: `{winner}`** (highest accuracy; ties broken by shorter prompt).")
    md_lines.append("")

    # "Where is he looking" specific check
    target_qid = "262148000"
    looking_results = []
    for tmpl, rows in template_results.items():
        for r in rows:
            if r["question_id"] == target_qid:
                looking_results.append((tmpl, r["prediction"], r["vqa_score"]))
    if looking_results:
        md_lines.append('## "Where is he looking?" — generator-ignores-evidence case')
        md_lines.append("")
        md_lines.append("| Template | Prediction | Score |")
        md_lines.append("|---|---|---:|")
        for tmpl, pred, sc in looking_results:
            md_lines.append(f"| {tmpl} | `{pred}` | {sc:.2f} |")
        md_lines.append("")

    md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved {md}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [t for t, _, _ in sortable]
        accs  = [template_summary[t]["accuracy"] * 100 for t in names]
        colors = ["#3fb950" if t == winner else ("#8b949e" if t == "current_prompt" else "#58a6ff")
                  for t in names]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.barh(names, accs, color=colors)
        for b, a in zip(bars, accs):
            ax.text(a + 0.5, b.get_y() + b.get_height()/2, f"{a:.2f}%",
                    va="center", fontsize=10)
        ax.axvline(base_acc * 100, color="#f85149", linestyle="--",
                   label=f"baseline {base_acc*100:.1f}%")
        if "current_prompt" in template_summary:
            ax.axvline(cur_acc * 100, color="#f0883e", linestyle=":",
                       label=f"current_prompt {cur_acc*100:.1f}%")
        ax.set_xlabel("VQA soft accuracy (%)")
        ax.set_title(f"Prompt ablation — RAG-VQA · n={args.num_samples}, top_k={args.top_k}, α={args.alpha}, 20k diverse KB")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        ax.set_xlim(0, max(accs) + 8)
        plt.tight_layout()
        plot_path = plots_dir / "prompt_ablation.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # ── 8. Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROMPT ABLATION RESULTS")
    print("=" * 70)
    print(f"{'template':<28} {'acc':>7}  {'Δ_base':>7}  {'Δ_cur':>7}  helped/hurt/neutral")
    print("-" * 70)
    for tmpl, _, _ in sortable:
        s = template_summary[tmpl]
        d_b = (s["accuracy"] - base_acc) * 100
        d_c = (s["accuracy"] - cur_acc) * 100 if cur_acc else 0.0
        h = s.get("helped", "-"); ht = s.get("hurt", "-"); n = s.get("neutral", "-")
        flag = " *" if tmpl == winner else ""
        print(f"{tmpl:<28} {s['accuracy']*100:6.2f}%  {d_b:+6.2f}  {d_c:+6.2f}   {h}/{ht}/{n}{flag}")
    print()
    print(f"WINNER: {winner}")


if __name__ == "__main__":
    main()
