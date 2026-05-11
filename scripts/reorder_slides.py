"""Reorder slides in presentation/rag_vqa_open_book.html.

Splits the deck on `<!-- SLIDE ... -->` headers, identifies each slide by its
header label, reorders them according to the desired sequence, and inserts a
"Backup Slides" divider between main and backup.

Run:
  python -m scripts.reorder_slides
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "presentation" / "rag_vqa_open_book.html"

# Desired main flow — by slide identifier as it appears in the SLIDE comment.
# Anything not in this list goes to backup, in its current order.
MAIN_ORDER = [
    "1",      # Title (now: 42.33% → 60.33%, +18 pts)
    "2",      # Motivation: closed-book vs open-book
    "3",      # Problem Statement
    "5",      # Architecture
    "6",      # Retrieval Scoring (20k visual memory / CLIP+FAISS)
    "8",      # Main Results (60.33% headline)
    "9",      # Why Image Retrieval Wins (image > text)
    "15c",    # Canonical answer compression / strict vs lenient
    "12",     # Where RAG Helps (open-ended)
    "14b",    # KB Scaling Win
    "15b",    # Final System (type-router)
    "11b",    # Main Limitation (clean oracle + decomposition)
    "16",     # Conclusion
    "16b",    # Demo Preview (pipeline diagram, sets up handoff)
    "17",     # Live Demo Handoff (last main slide)
]

BACKUP_DIVIDER = """<!-- ══════════════════════════════════════════════════════════════════
     BACKUP DIVIDER — DETAILED ABLATIONS & NEGATIVE RESULTS
════════════════════════════════════════════════════════════════════ -->
<div class="slide" data-notes="From here onward: detailed ablations, negative results, legacy plots. Use these only if the audience asks for them.">
  <div class="tag" style="background:rgba(139,148,158,0.12);color:var(--ink-dim);">Backup · Detailed Ablations</div>
  <h2 style="color:var(--ink-dim);">Backup Slides</h2>
  <div style="margin-top:30px;font-size:18px;color:var(--ink-dim);line-height:1.7;max-width:860px;">
    The remaining slides cover detailed ablations, negative results, and legacy
    plots. They are kept for reproducibility and Q&A — not part of the main flow.
    <br/><br/>
    Topics: prompt-template ablation · question-aware reranking · evidence-aware
    selector · self-consistency vote · QLoRA fine-tuning · caption ablation ·
    helpful/harmful breakdown · confidence calibration · top-k legacy · oracle legacy.
  </div>
</div>

"""


def main():
    html = HTML.read_text(encoding="utf-8")

    # Find the deck section start/end
    deck_start = html.index('<div class="deck" id="deck">') + len('<div class="deck" id="deck">')
    deck_end = html.index('</div><!-- end deck -->')
    pre_deck   = html[:deck_start]
    deck_inner = html[deck_start:deck_end]
    post_deck  = html[deck_end:]

    # Split on the SLIDE comment headers. Pattern matches the full multi-line comment.
    # Captures: leading content, then list of (header_comment, body_until_next_header)
    parts = re.split(r"(<!-- ═{50,}\n\s+SLIDE \w+ — [^\n]+\n═{50,} -->\n)", deck_inner)
    # parts is: [leading_whitespace, header1, body1, header2, body2, ...]
    if len(parts) < 3:
        print("ERROR: no slide headers found")
        return 1

    leading = parts[0]
    slides = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        body   = parts[i + 1] if i + 1 < len(parts) else ""
        # Extract slide ID from the header
        m = re.search(r"SLIDE (\w+) — ", header)
        if not m:
            print(f"  WARN: could not extract id from header:\n{header[:200]}")
            continue
        sid = m.group(1)
        slides.append((sid, header + body))
        print(f"  found slide {sid:<5}  ({len(body)} chars)")

    print(f"\nTotal slides: {len(slides)}")

    # Build main list in order; everything else is backup
    by_id = {sid: blk for sid, blk in slides}
    main_blocks = []
    for sid in MAIN_ORDER:
        if sid in by_id:
            main_blocks.append(by_id[sid])
        else:
            print(f"  WARN: main slide id '{sid}' not found")

    main_ids = set(MAIN_ORDER)
    backup_blocks = [blk for sid, blk in slides if sid not in main_ids]

    print(f"\nMain: {len(main_blocks)}  Backup: {len(backup_blocks)}")

    # Reassemble
    new_deck_inner = (
        leading
        + "".join(main_blocks)
        + BACKUP_DIVIDER
        + "".join(backup_blocks)
    )
    new_html = pre_deck + new_deck_inner + post_deck

    # Backup the original
    backup_path = HTML.with_suffix(".html.bak")
    backup_path.write_text(html, encoding="utf-8")
    print(f"\nBacked up original to {backup_path}")

    HTML.write_text(new_html, encoding="utf-8")
    print(f"Wrote reordered deck to {HTML}")
    print(f"\nFinal slide order:")
    for i, sid in enumerate(MAIN_ORDER, 1):
        print(f"  Main {i:>2}.  Slide {sid}")
    print("  ─── BACKUP DIVIDER ───")
    for sid, _ in slides:
        if sid not in main_ids:
            print(f"  Backup     Slide {sid}")


if __name__ == "__main__":
    sys.exit(main() or 0)
