# Open-Book VQA — Speaker Notes (3-presenter split)

**Project**: Visual Memory-Augmented BLIP-2 with CLIP + FAISS Retrieval
**Final headline**: 42.33% → 60.33% strict VQA accuracy · +18 pts · frozen 3B model · no retraining

**Total time target**: ~15 minutes deck + ~3 minutes live demo = **~18 minutes total**

---

## Roles

| Presenter | Slides | Section | Time |
|---|---|---|---|
| **Jillian** | 1 – 5 | Setup, motivation, system architecture | ~5 min |
| **Anushree** | 6 – 7, 9 – 10 | Headline result + wins | ~4 min |
| **Nandita** | 8, 11 – 15 | Metric honesty + bottleneck + final system + live demo | ~7-8 min |

Nandita takes the **hardest slides** (8 — strict vs lenient metric, 12 — bottleneck decomposition) plus the **live demo** at the end.

---

# 🟦 PART 1 — JILLIAN (slides 1–5, ~5 minutes)

**Role**: Set up *what we built and why*. Make the closed-book vs open-book idea click.
**Energy**: Opener — confident, clear, slightly slow. Establishes credibility.

---

## SLIDE 1 — Title (~30 s)

**Open with:**
> "Hi everyone, I'm Jillian. Today the three of us — myself, Anushree, and Nandita — are going to walk you through our final project: **Open-Book VQA**. We took a frozen visual question-answering model called BLIP-2, which is a 3-billion parameter model, and we gave it the ability to **look up** similar past examples before answering. The headline is on the slide: we improved strict VQA accuracy from **42.33% to 60.33%** — an 18-point absolute gain — without any retraining. Let me set up why this is interesting."

**Hand-off cue:** "*Let me start with the analogy that motivated this whole project.*"

---

## SLIDE 2 — Motivation (~50 s)

**Open with:**
> "Think of standard VQA as a **closed-book exam**. You give the model an image and a question, and it has to answer purely from memorized weights — no looking anything up. A student in a closed-book exam can only score as well as what they remember. Now imagine the same student in an **open-book exam**: they can consult notes, look up similar problems they've seen before. They usually do better, especially on hard questions."

**Talking points:**
- Frozen models — the weights don't change. They have whatever knowledge they were trained with.
- For hard, specific questions, the closed-book setup is fundamentally limiting.
- **Our project asks**: can we let a frozen model "consult notes" at inference time?
- The notes here are a **visual memory** — past examples of question-answer pairs over images.

**Hand-off cue:** "*Here's the precise task.*"

---

## SLIDE 3 — Problem Statement (~40 s)

**Open with:**
> "The task is **VQAv2** — a standard visual question-answering benchmark. The model gets one image and one natural-language question. It has to output a short answer. Scoring is **strict VQA soft accuracy**: an answer matches against ten human annotators, and you get full credit only if at least three of them said your exact answer."

**Key points:**
- The model is **BLIP-2 OPT-2.7B**, frozen, 4-bit quantized — we are not training it.
- Eval set: **100 validation samples**, fixed across all our experiments — same eval, same KB, only the system around the model changes.
- This is a **controlled evaluation**, not a full benchmark — but the comparisons are internally fair.

**Hand-off cue:** "*Here's how the open-book system is built.*"

---

## SLIDE 4 — Architecture (~70 s)

**Open with:**
> "This is the full pipeline, left to right. The query image and question come in. **CLIP** — a vision-language model — encodes the query image into a 512-dimensional vector. We do an exact inner-product search over a **FAISS index** that holds 20,000 past VQA examples — that's our visual memory. We get back the top-3 most visually similar examples and their answers. We format those as a hints prompt, prepend them to the question, and feed everything to **BLIP-2**, which generates the final answer."

**Walk through the diagram step by step.** Point at each box.

**Talking points:**
- "We use **CLIP ViT-B/32** for retrieval — chosen because CLIP is trained on image-text pairs, so its embeddings cluster by visual semantics."
- "The KB is **20,000 unique images** from VQAv2's validation split, with **image-id-disjoint sampling** — meaning no image in the KB is also in the eval set. This avoids leakage."
- "BLIP-2 itself is frozen, so the only thing we're 'training' here is the retrieval and prompting logic."

**Hand-off cue:** "*The most important design parameter is how we score retrieval.*"

---

## SLIDE 5 — Retrieval Scoring (~70 s)

**Open with:**
> "We have one core knob, alpha. We score each candidate as **alpha times image similarity plus one-minus-alpha times question-text similarity**. At alpha equals zero, only text similarity matters. At alpha equals one, only image similarity matters. The big finding is that **alpha equals one wins** — pure image-only retrieval, achieving **59.33%** before we add anything else on top."

**Talking points:**
- "There's also an optional gate, tau. If the top-1 retrieval score is below tau, we skip retrieval and fall back to baseline. We use this in the live demo for out-of-distribution images."
- "We'll come back to **why** image-only beats text-only in slide 7 — there's a clean reason."

**Hand-off cue:** "*With the system in place, let me hand off to Anushree to walk through the headline result.*"

---

# 🟩 PART 2 — ANUSHREE (slides 6, 7, 9, 10 · ~4 minutes)

*(Note: slide 8 belongs to Nandita — Anushree skips from 7 to 9.)*

**Role**: Deliver the **wins**. Confident, numbers-forward.
**Energy**: Pace up. The interesting parts are landing.

---

## SLIDE 6 — Main Result (~80 s)

**Open with:**
> "Thanks Jillian. Let me show you the headline. Same eval, same model, same KB across all rows. Baseline frozen BLIP-2 alone scores **42.33%**. Adding image-only retrieval over the 20k diverse KB takes us to **59.33%** — a +17 point absolute gain. Adding the type-conditional router on top — which Nandita will explain in detail — takes us to **60.33%**. That's our final system: **+18 points absolute, +43% relative**, no retraining."

**Walk the table top-to-bottom.** Point at each row:
- Baseline 42.33% — frozen BLIP-2 alone
- α=0.0 text-only → 43.67%, basically baseline
- α=0.5 balanced → 46.67%, mediocre
- α=1.0 image-only → **59.33%** ★ (highlight)
- + Type-router → **60.33%** ★★ (final system)

**Key line to land:**
> "**Image-only retrieval is the most important design choice** in this system. Mixing in question-text similarity actively hurts. We'll explain why in the next slide."

**Hand-off cue:** "*This is counterintuitive — let me show you why text retrieval doesn't work here.*"

---

## SLIDE 7 — Why Image Retrieval Wins (~60 s)

**Open with:**
> "Why does image retrieval beat text retrieval, when text retrieval is the standard thing in NLP RAG? Three reasons."

**Walk through the three reasons:**

1. **VQA answers are visually grounded.** *"What color is the car?"* — the answer depends on the image, not the question wording.
2. **Short VQA questions are generic.** Examples like *"how many people"* or *"what color is this"* appear across thousands of completely different images. Text retrieval finds questions that look the same but are about totally different visual content — the retrieved answers are noise.
3. **CLIP image embeddings cluster by visual semantics**, which correlates well with the answer space — same scene type, same objects, often similar answers.

**Concrete example to land:**
> "Take *what color is the car?*. Text retrieval finds other 'what color' questions — but about cars of different colors. The retrieved answers contradict each other. Image retrieval finds **other images of the same car**, where the answer is consistent."

**Hand-off cue:** "*Now let me show you where this matters most.*"

*(Skip slide 8 — Nandita takes that one.)*

---

## SLIDE 9 — Where RAG Helps (~50 s)

**Open with:**
> "Let me break the gain down by question type. Yes/no questions, baseline scores about 83%. Adding RAG actually drops it slightly to 80% — retrieval introduces noise on questions where BLIP-2 was already strong. Open-ended questions, where BLIP-2 baseline only gets 21%, RAG more than doubles to 48%. So **the gain is concentrated on open-ended questions** — exactly where we'd expect retrieval to help most."

**Key line:**
> "This pattern — yes/no slight drop, open-ended big gain — is what motivated the type-conditional router that Nandita will explain in slide 11."

**Hand-off cue:** "*Before that, let me show you the single biggest lever in this whole system.*"

---

## SLIDE 10 — KB Scaling Win (~70 s)

**Open with:**
> "We tried three different knowledge bases on the same 100-sample eval. Read the table top to bottom:
> - The original 5k validation-built KB — which had image-id leakage with the eval set — scored **51.33%**.
> - A 5k **diverse, image-id-disjoint KB** scored **54.67%** — clean, no leakage.
> - The 20k diverse KB scored **59.33%**.
>
> **KB scale and diversity drive most of the gain in this system.** Going from 5k diverse to 20k diverse alone gave us +5 points. Bigger than any prompt or selector trick we tested afterwards."

**Talking points:**
- "The **'5k unique images'** is the key — VQAv2 has multiple questions per image, so a small KB has very few unique scenes."
- "Image-id-disjoint sampling means we caught and **fixed a leakage bug** that would have inflated our numbers."
- "Future-work direction this points to: a 50k+ KB and a trainable retriever — Nandita will come back to this in the limitation slide."

**Hand-off cue:** "*Nandita is going to take us through the metric question, the final system, and the demo.*"

---

# 🟧 PART 3 — NANDITA (slides 8, 11–15 · ~7-8 minutes)

**Role**: Deliver the **technically hardest** slides — strict-vs-lenient metric, bottleneck decomposition, type-router justification — then close on the **live demo**.
**Energy**: Calm, precise, slightly slower than Anushree. These slides reward careful explanation. The demo is the payoff.

---

## SLIDE 8 — Canonical Answer Compression (the strict vs lenient slide) (~80 s)

**This is the conceptually hardest slide. Take your time.**

**Open with:**
> "I want to be honest about what these numbers mean. Here's the question: BLIP-2 baseline on this skateboarder image answers *'Watching the skateboarder'*. Ground truth is *'watching'*. **Strict VQA scoring marks this WRONG** because it requires exact-string match. But the answer literally contains the canonical word. Shouldn't it count?"

**Walk through the three regimes:**
> "Same 100 predictions, three accuracy regimes, no new inference. **STRICT** — exact match, the existing 60.33% headline. **LENIENT** — the ground truth appears as a whole word in the answer; rewards verbose-but-correct, rejects degenerate outputs like 'blueblue'. **SUBSTR** — most permissive, GT appears anywhere."

**Read the table aloud:**
> "Baseline jumps from 42.33% strict to **65% lenient** — that's plus 23 points. The model already knew the answer to 27 cases that strict scoring marked wrong. The RAG advantage shrinks from +17 strict to **just +1 lenient**."

**Land the key insight (slow, deliberate):**
> "**About 16 of the 17 RAG gain points were format compression — the model already knew most of those answers, RAG just nudged it into the canonical short form. About 1 point is genuinely new information from retrieval.** Both are useful — VQA leaderboards reward the canonical form, and so do real users — but I want to be honest about what each is doing."

**End with the robustness point:**
> "The relative ordering of all our design choices is the same under all three metrics. Image-only retrieval, the type-router, KB scaling — they all win regardless of which metric you pick. Strict VQA is a harsh form-judge but it doesn't change which design choices are right."

**Anticipated question to be ready for:**
> *"So is your project less impressive than you said?"*
> Answer: *"Most published RAG-VQA papers report ~+1 point under similar lenient scoring. We're in line with the literature — we're just being more transparent about which part is format and which part is content."*

**Transition:** "*This +1 of new information was further amplified by a small heuristic — the type-conditional router.*"

---

## SLIDE 11 — Final System: Type-Conditional Router (~60 s)

**Open with:**
> "Anushree showed you that yes/no questions lose ~3 points to RAG, and open-ended gains ~28. The fix is one line of code. **Detect if the question starts with 'is', 'are', 'do' — yes/no — and if so, return the baseline answer instead of the RAG answer. For open-ended questions, keep RAG.** That's the whole router."

**Read the breakdown:**
> "Baseline yes/no = 82.86%. RAG yes/no = 80%. Baseline open-ended = 21%. RAG open-ended = 48%. The router gets 82.86% on yes/no and 48% on open-ended — net **+1.00 point overall**. Final system: **60.33%**."

**Honest framing:**
> "We tried much fancier things — cross-encoder reranking, an evidence-aware selector that overrides RAG with retrieved answers, self-consistency voting across alphas. **The simple type-router is the only thing that helped.** Sometimes the simple heuristic wins."

**Transition:** "*OK — so what about the cases where RAG doesn't help? Where's the bottleneck?*"

---

## SLIDE 12 — Main Limitation: Clean Oracle + Bottleneck Decomposition (~90 s)

**This is the most technical slide. Take your time. This is the slide that proves you understand your own system.**

**Open with:**
> "I want to show you exactly where the failure cases come from. We did two things on the same 100-sample eval. First, **clean oracle@k** on the 20k KB — for each query, is the ground-truth answer anywhere in the top-k retrieved entries? At k=3, the answer is in retrieval **only 24% of the time**. At k=10, only **46%**. So the real ceiling for any inference-time selector picking among retrieved answers is 46%."

**Pause to let that land. Then continue:**
> "Notice that our RAG system scores **59.33%** — well above the clean oracle@10 of 46%. That's because BLIP-2 also uses **image grounding and world knowledge**, not just the retrieved text. RAG isn't copying retrieved answers — it's combining them with what BLIP-2 sees."

**Now the decomposition table — read it carefully:**
> "We classified all 100 cases into four buckets based on (a) is the GT in retrieval, and (b) did RAG pick the right answer.
> - **18 cases** — GT in retrieval, model picked it. RAG working as designed.
> - **37 cases** — GT not in retrieval, model still got it right. BLIP-2 image-grounded baseline strength.
> - **6 cases** — GT in retrieval, but the model ignored it. Generator failure.
> - **39 cases** — GT not in retrieval at all. Retrieval failure."

**Land the key insight:**
> "Of the 45 wrong cases, **87% are retrieval-bound — the answer just isn't in our KB** — and only **13% are generator-bound**. This **reverses an earlier intuition**: I kept showing 'the generator ignores evidence' as the failure mode, but on the clean 20k KB, that's only 6 cases out of 100. The dominant bottleneck is **retrieval coverage**."

**Future-work priority order (read explicitly):**
> "By the data, the right priority for future work is: **first**, scale the KB to 50k+ unique images — directly targets 87% of failures. **Second**, train a retriever specifically for VQA-relevant features. **Third**, fine-tune the generator — that targets only 13% of failures, and our QLoRA experiment confirmed lightweight adapters alone aren't enough."

**Transition:** "*Let me close on the headline.*"

---

## SLIDE 13 — Conclusion (~40 s)

**Open with:**
> "To wrap up — our final system is a frozen BLIP-2 model plus a 20k image-id-disjoint visual memory plus image-only CLIP retrieval plus a one-line yes/no router. It improves strict VQA accuracy from **42.33% to 60.33%** — that's +18 absolute points, +43% relative — with no retraining."

**Touch each pillar:**
- "**KB scale** drives most of the gain — going from 5k to 20k unique images contributed about 5 of the 17 RAG points."
- "**Image-only retrieval** beats text and balanced retrieval, because short VQA questions don't discriminate."
- "**Honest scoring**: under semantic-correctness scoring the gain is smaller — about 1 point of new information, 16 points of format compression — but the design choices are robust."
- "**The bottleneck is retrieval coverage**, which points clearly to where to invest next."

**Transition:** "*And now — let me show you the system actually running.*"

---

## SLIDE 14 — Demo Preview (~25 s)

**Open with:**
> "Quick orientation before I switch to the browser. The demo shows four curated cases that each illustrate one finding from the eval. Each case shows the **mode comparison** — baseline, 5k RAG, 20k RAG, final system — plus the **retrieved evidence** and a one-sentence *why this case matters*. I'll walk you through them."

**Hand-off cue:** "*Switching to the browser now.*"

---

## SLIDE 15 — Live Demo (~3 minutes in browser)

**Switch to browser tab. Demo URL: `http://localhost:8081/demo.html`.**

Follow `presentation/demo_walkthrough.md` for exact wording. The four cases in order:

### Case 1 — RAG Success: Visual Memory (*Are the lights on?*)
> *"Vanilla BLIP-2 confidently says 'yes'. The retrieved memory pulls back two on/off questions from similar bedroom scenes — 'Is the TV on? — no' and 'Is the lamp turned on? — no'. The candidate-priors panel shows 'no' as the majority. The model picks up the pattern and correctly answers 'no'. Open-book working as designed."*

### Case 2 — Canonical Answer Compression (*People watching the skateboarder*)
> *"BLIP-2 says 'Watching the skateboarder' — semantically right but VQA scoring requires the canonical short form, so it scores zero. RAG compresses to 'watching' — full credit. This is exactly the format-compression effect from slide 8."*

### Case 3 — KB Scaling Win (*Folding chair?*)
> *"Same question, three modes side by side. Baseline says yes — wrong. The 5k KB also says yes — too small to surface enough scene context. The 20k KB says no — correct. KB scale matters, concretely."*

### Case 4 — Honest Weakness (*Is it daylight?*)
> *"Baseline correctly says yes — clearly daytime. RAG retrieves cow-related questions whose answers cluster on 'no'. The noise misleads BLIP-2 into flipping a correct answer. **9 out of 100 cases look like this** — concentrated on yes/no, which is exactly why the type-router exists."*

### Close
> *"That's the system. Thank you — happy to take questions."*

---

# Q&A backup answers (any presenter can take these)

**"How is this different from regular RAG for text?"**
> Visual retrieval is the difference. Most RAG papers retrieve text. We retrieve visually-similar images and use their associated Q&A as context. The key finding is that for short VQA questions, image similarity is a stronger signal than question-text similarity.

**"Why didn't fine-tuning work?"**
> We trained a QLoRA adapter on the q_proj and v_proj of OPT-2.7B for 5 epochs. Best variant scored 13% — versus 60% with the frozen + RAG approach. The adapter learned the right answer tokens but lost generation discipline (mode collapse: outputs like "blueblue" or "yesyesyes"). The adapter scope was too narrow — q/v projections only — and the FFN and embeddings stayed frozen.

**"What's the absolute hardest improvement to make from here?"**
> Trainable visual retriever. Off-the-shelf CLIP isn't optimized for VQA-relevant features. A retriever fine-tuned on (query, GT-similar-example) pairs could push oracle@3 from 24% to maybe 40%, and that ceiling lift translates directly into more usable retrieval contexts.

**"Why 100 samples?"**
> A controlled set used consistently across every ablation. Not a full benchmark generalization claim — it's a same-eval-set comparison setup so the deltas between configurations are fair. Full benchmark validation is future work.

**"Did you compare to other RAG-VQA systems?"**
> Published RAG-VQA papers (PICa, KAT, REVIVE) report 1-3 point strict gains over baseline. Our +18 strict / +1 lenient is in line with what those papers report — we're more explicit about decomposing the gain into format compression vs new information, which most papers don't do.

---

# Tone reminders for everyone

✅ Say: **"controlled 100-sample eval"**, not *"benchmark generalization"*
✅ Say: **"format compression"**, not *"the metric is broken"*
✅ Say: **"retrieval-bound failure"**, not *"the model is broken"*
✅ Say: **"image-only beats text retrieval"**, not *"text retrieval failed"*

❌ Don't apologise for the lenient number being smaller — own it as honesty
❌ Don't oversell the +18 — always pair it with what the metric measures
❌ Don't pretend BLIP-2 is something we built — it's pretrained, we built the retrieval system around it

---

# Quick logistics checklist

Before the talk:
1. Open the presentation: `presentation/rag_vqa_open_book.html`
2. Open the demo in another tab: `http://localhost:8081/demo.html`
3. Run `python api.py` in a terminal — wait for *"Application startup complete"*
4. **Warm up the API**: open the demo's Curated tab, click any "Run live" button — first call takes ~60s to load BLIP-2; this preloads it
5. Verify the four curated cases load: cow / watermark or watching / chair / daylight
6. Have `presentation/demo_walkthrough.md` open on a phone or second screen for backup

If the API breaks during the demo:
- The Curated tab works fully offline from cached JSON
- Skip the live elements; walk through the four cases verbally — the data is all visible

---

*End of speaker notes.*
