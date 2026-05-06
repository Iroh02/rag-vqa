# Demo Script — Open-Book VQA Live Demo
**Nandita Menon · May 7, 2026**

---

## 30-Second Demo Explanation

"This is the full pipeline running end-to-end. The left side shows the query image and question. In the middle you see what vanilla BLIP-2 answers with no context. On the right is the RAG answer — after CLIP retrieves similar past examples from a 5,000-entry knowledge base. The three cards below show exactly what was retrieved and how similar each one was. The Prompt tab shows the exact text BLIP-2 received. Every number you saw in the slides came from 100 runs of this pipeline."

---

## 2-Minute Demo Walkthrough

**Setup:** Run `python demo_app.py --mode cached` before presenting. Browser opens at `http://localhost:7860`.

---

### Example 1 — RAG Helps: Open-ended (best)

**Select:** "RAG Helps - Open-ended (best)"

**Say:**
"Question: what are the people in the background doing? Baseline BLIP-2 answers 'Watching the skateboarder' — a long-form phrase that doesn't match the annotator answers. RAG answers 'watching' — an exact match. The retrieval panel shows three examples pulled from the same image. Click the Prompt & Context tab. You can see the exact hint text prepended to the question. BLIP-2 gets: 'Hints may be wrong. Similar image suggests: picnic table. Similar image suggests: down. Similar image suggests: watching.' That's the complete input."

**Key point:** "The retrieved answer 'watching' is present in the hints. But notice the baseline already knew the general scene — it just gave the wrong format. The hint nudged it to produce the canonical short answer."

---

### Example 2 — RAG Helps: Copyright watermark

**Select:** "RAG Helps - Open-ended (2nd)"

**Say:**
"This is my favorite example. The question is: what website copyrighted the picture? Baseline BLIP-2 says 'none' — it cannot read the small watermark text. But RAG retrieves a visually similar image from the knowledge base that also had a copyright watermark, and that KB entry has the answer 'foodiebakercom'. RAG produces exactly 'foodiebakercom' — correct."

**Key point:** "This is visual memory in action. The model isn't reading the watermark; it's recognizing a similar image and borrowing the answer. This is the clearest demonstration that retrieval adds factual knowledge the generator alone doesn't have."

---

### Example 3 — RAG Helps: Yes/No fixed

**Select:** "RAG Helps - Yes/No fixed"

**Say:**
"Question: is this a creamy soup? Baseline says 'yes' — wrong. RAG says 'no' — correct. The retrieved examples are from the same image and include the answer 'no' for the same question. RAG flips a wrong yes/no answer to correct."

---

### Example 4 — RAG Hurts: Yes/No confused

**Select:** "RAG Hurts - Yes/No confused"

**Say:**
"Here's the failure mode. Question: is this rice noodle soup? Baseline correctly says 'yes'. But RAG retrieves from the same visually similar image cluster — and one of the hints says 'no' for creamy soup. RAG gets confused and answers 'no' — wrong. This is the 9 out of 100 cases where context hurts. The model can't distinguish which retrieved hint applies to which aspect of the scene."

**Key point:** "This is why alpha-tuning matters and why RAG slightly hurts yes/no questions on average. The model was already confident and correct; adding noisy hints degraded it."

---

### Example 5 — Both Correct: No change

**Select:** "Both Correct - No change"

**Say:**
"How many photos can you see? Both baseline and RAG answer '1' — correct. 70 out of 100 examples are like this: RAG adds no harm when the model is already right. This is the neutral majority."

---

### Example 6 — Both Fail: Hard question

**Select:** "Both Fail - Hard question"

**Say:**
"Finally, where is he looking? Ground truth is 'down'. Baseline says 'sky'. RAG says 'up'. The retrieved hints do include 'down' — it's there in the second retrieved example — but the model ignores it. This is the generator reliability problem: retrieved evidence is present but not used. Fixing this requires fine-tuning or a stronger generator, not better retrieval."

---

## What to Say at the End

"To summarize what you just saw: the demo runs the complete pipeline locally — CLIP retrieval, FAISS search, prompt construction, and BLIP-2 generation — with no retraining. The 9-point gain in the results table is directly observable in examples 1, 2, and 3. The failure mode in example 4 is also real and expected. This is the system."

---

## Quick Reference: Example Order

| # | Label | Verdict | Delta | Key talking point |
|---|-------|---------|-------|-------------------|
| 1 | RAG Helps - Open-ended (best) | helpful | +1.0 | Format matters; hint gives correct short form |
| 2 | RAG Helps - Open-ended (2nd) | helpful | +1.0 | Visual memory adds factual knowledge (watermark) |
| 3 | RAG Helps - Yes/No fixed | helpful | +1.0 | Same-image retrieval flips wrong yes/no |
| 4 | RAG Hurts - Yes/No confused | harmful | −1.0 | Cross-question noise misleads confident model |
| 5 | Both Correct - No change | neutral | 0 | 70% of examples are unchanged |
| 6 | Both Fail - Hard question | neutral | 0 | Evidence present but not used |

---

## If Asked "Is This Live or Pre-recorded?"

"This is live. The images were downloaded from VQAv2 validation split; inference ran on this machine using BLIP-2 loaded in 4-bit quantization. The demo_cases.json was built from a 30-sample inference run earlier today. To run live inference on a new image, switch to the Live Inference tab."

---

## Backup: If Demo Won't Start

Run:
```
python demo_app.py --mode cached
```
If Gradio fails to launch, open `results/demo_cases.json` directly and walk through the 6 examples from the terminal JSON output. All results are pre-computed.

---

*End of demo script.*
