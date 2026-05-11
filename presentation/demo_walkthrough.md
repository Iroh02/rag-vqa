# Demo Walkthrough — Open-Book VQA
**Final presentation flow**

---

## Setup (run once before the talk)

```
# terminal 1 — static server
python -m http.server 8081

# terminal 2 — inference API (only needed if you use the Advanced/Live tab during Q&A)
venv\Scripts\python.exe api.py --port 8000
```

Open `http://localhost:8081/demo.html`. Default tab is **Curated Demo** — instant, presentation-safe.

Open the deck: `presentation/rag_vqa_open_book.html`. Use ←/→ to navigate.

---

## 30-second opening

> *"Standard VQA is closed-book: BLIP-2 sees one image and answers alone. I converted it into open-book VQA by adding a CLIP + FAISS visual memory of 20,000 examples. The model retrieves similar Q&A pairs before answering. The final system improves strict VQA accuracy from 42.33% to 60.33%, and the key finding is that visual retrieval and simple type routing work better than more complex text-based retrieval."*

---

## 3-minute presentation flow

**Slides 1–7 (∼60 s)** — closed-book vs open-book, problem, architecture, retrieval, headline result.

**Slide 8 — Demo preview** (10 s) — *"Now I'll show four real cases from the eval, run through the working system."*

**→ Switch to demo browser tab. Walk through 4 demo cases (∼90 s).** Then back to slides.

**Slide 9 — Canonical answer compression / strict vs lenient** (15 s) — *"Honest framing: a lot of the strict-VQA gain is BLIP-2 being verbose-but-correct on the baseline; RAG compresses to the canonical short form. Under semantic-correctness scoring, the gain is smaller, but the design choices are robust either way."*

**Slides 10–12 (∼30 s)** — open-ended is where RAG helps most, KB scaling is the dominant lever, type-router is the +1 polish.

**Slide 14 — Main limitation (∼15 s)** — *"Of 45 wrong cases, 87% are retrieval-bound and only 13% are generator-bound. The honest priority order for future work is: bigger KB, trainable retriever, then generator fine-tuning."*

**Slide 15 — Conclusion (∼15 s)** — *"42.33% to 60.33% strict VQA, frozen 3B model, no retraining. The biggest lever is KB scale; the simplest polish is a one-line yes/no router."*

---

## Demo flow — exact words for each of the 4 cases

### Case 1 — RAG Success: Visual Memory · *Are the lights on in this room?*

**Click:** sidebar entry *"Retrieved Q&A visibly supports the answer — are the lights on?"*

**Say:**
> *"Vanilla BLIP-2 confidently says 'yes'. The retrieved memory pulls back two related on/off questions from similar bedroom scenes — 'Is the TV on? — no' and 'Is the lamp turned on? — no'. The candidate-priors panel shows 'no' as the majority. The model picks up the pattern and correctly answers 'no'. This is the open-book mechanism working as designed: the model isn't reading the answer off the screen, it's using retrieved context to override a confident wrong prior."*

### Case 2 — Canonical Answer Compression · *What are the people in the background doing?*

**Click:** *"Open-ended format flip — people watching the skateboarder"*

**Say:**
> *"BLIP-2 baseline says 'Watching the skateboarder' — semantically correct, but VQA scoring requires the canonical short form, so it scores zero. RAG compresses this to 'watching' — the canonical answer. This case is honest: the win here is **format compression**, not semantic rescue. Open-ended questions are exactly where this effect is largest, and the eval shows +14 points on open-ended versus the slight drop on yes/no that motivates the type-router."*

### Case 3 — KB Scaling Win · *Is that a folding chair?*

**Click:** *"5k KB couldn't fix this; 20k KB can — is that a folding chair?"*

Point at the **Mode Comparison** table.

**Say:**
> *"Same question, three modes side by side. Baseline: 'yes'. The 5k diverse KB also says 'yes' — its KB is too small to surface enough scene context. The 20k diverse KB retrieves furniture and lighting questions whose answers cluster on 'no', and the model correctly answers 'no'. This is the most important lesson: the dominant lever in this system is **KB scale**, not prompt engineering or selectors. Going from 5k to 20k unique images contributed five of the seventeen total accuracy points."*

### Case 4 — Honest Weakness · *Is it daylight in this picture?*

**Click:** *"Retrieval flips a confident correct yes/no — is it daylight?"*

Point at the **caveat** line.

**Say:**
> *"This is the honest weakness. Baseline correctly says 'yes' — it's clearly daytime. RAG retrieves cow-related questions whose answers cluster on 'no' — the noise misleads BLIP-2 into flipping a correct answer into a wrong one. **Nine of one hundred eval cases look like this**, and they're concentrated on yes/no questions where BLIP-2 was already strong. That's exactly what motivated the type-conditional router: skip retrieval on yes/no, use it on open-ended. With the router on, this case is recovered."*

---

## Q&A backup answers

### "Did RAG really make BLIP-2 smarter?"
> *"Partially. Under strict VQA scoring, RAG adds 17 points absolute. Under lenient scoring that accepts verbose-but-correct answers, the gain shrinks to about 1 point — most of the strict gain is format compression. Both numbers are useful: VQA leaderboards reward the canonical short form, and so do real users."*

### "Why does lenient scoring reduce the gain?"
> *"Because the baseline already knows the answer to ~27 cases that strict VQA marks wrong — answers like 'Watching the skateboarder' contain 'watching' but don't exact-match. Under whole-word substring scoring, those count. Both metrics agree on the relative ordering of design choices though, so KB scaling, image-only retrieval, and the type-router are robustly better than alternatives regardless of metric."*

### "Why is format compression valuable?"
> *"Because every VQA benchmark uses strict scoring, and most downstream applications want short canonical answers. RAG turns a verbose 3B frozen model into one that produces VQA-leaderboard-style outputs without any retraining. That's a real-world useful effect, even if it's not 'the model became smarter' in the textbook sense."*

### "Why image-only retrieval?"
> *"Short VQA questions like 'what color is this?' or 'how many?' are nearly identical across totally different images, so question-text retrieval doesn't discriminate. Image embeddings cluster by visual semantics — same scene type, same object, same colour palette — which correlates much better with what answers should be. The α-sweep confirms this: α=1.0 (image only) gets 59%, α=0.5 (mixed) gets 47%, α=0.0 (text only) gets 44%."*

### "Why did question-aware reranking hurt?"
> *"We weighted retrieval as 0.75·image + 0.25·question over a top-50 candidate set. It dropped accuracy 9 points. The reason is the same — short VQA questions don't discriminate. Adding a 25% question-similarity term diverts retrieval toward visually-different scenes whose questions happen to share words. Image-only is genuinely the right choice on this benchmark, and our experiment quantifies that."*

### "Why did the evidence-aware selector hurt?"
> *"The selector overrides the model's RAG output with a retrieved answer when it's confident the retrieval matches. On this benchmark it overrode correct RAG answers with worse retrieved ones more often than it rescued bad ones. Net effect: −3 points. We keep it as a demo toggle because the per-case logic is interpretable and useful for explaining what the system 'thought it knew', but it's not in the reported number."*

### "Why not show live multi-question?"
> *"It exists — there's an 'Advanced (Live)' tab — but I keep it for Q&A rather than the main flow. The four curated cases each illustrate a specific finding from the controlled eval. Live uploads are out-of-distribution for a 20k VQAv2 KB, so they often hit the τ-gate and fall back to baseline. They're real but they make a less clean story for a 3-minute demo."*

### "What is the final best system?"
> *"Frozen BLIP-2 + 20k diverse image-id-disjoint KB + CLIP image-only retrieval at α=1.0, k=3 + a one-line yes/no type-router. **42.33% → 60.33% strict VQA**. No fine-tuning, no prompt engineering beyond the standard Q/A few-shot format."*

### "What would you improve next?"
> *"Three things, in priority order. First, scale the KB from 20k to 50k+ unique images — the bottleneck decomposition shows 87% of failures are retrieval-bound, so coverage is the highest-leverage change. Second, train a retriever specifically for VQA-relevant features rather than using off-the-shelf CLIP. Third, fine-tune the generator — but that targets the smaller 13% of failures, and our QLoRA experiment showed lightweight adaptation alone causes mode collapse, so this needs partial unfreezing or a different recipe."*

---

## Tone reminders

- **Confident, not apologetic.** Lead with wins, mention negatives only when they prove rigour.
- Say *"controlled 100-sample eval"*, not *"benchmark generalisation"*.
- Say *"format compression"*, not *"the metric is broken"*.
- Say *"image-only retrieval beats question-text retrieval"*, not *"text retrieval failed"*.
- Say *"the dominant bottleneck is retrieval coverage"*, not *"the model is broken"*.

---

## If the demo breaks

- The Curated Demo tab works **fully offline** from cached `results/demo_cases.json`. No backend needed.
- If the Advanced (Live) tab is broken or the API is down: don't open it. The 4 curated cases tell the whole story.
- If you can't open the demo at all: walk through the 4 cases verbally using the script above. The slides have everything you need.

---

*End.*
