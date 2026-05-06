# Speaker Script — Open-Book VQA
**Nandita Menon · May 7, 2026**

---

## 30-Second Opening

"Standard VQA is normally closed-book: the model sees one image and must answer from its internal knowledge. In this project, I convert BLIP-2 into an open-book VQA system by adding a CLIP + FAISS visual memory. The model retrieves similar past VQA examples and uses their Q&A pairs as hints before answering. This improves accuracy from 42.33% to 51.33% — a +9 point absolute gain — and the main finding is that visual similarity is a stronger retrieval signal than question-text similarity."

---

## 2-Minute Presentation Script

**[Slide 1 — Title]**
Today I'm presenting Open-Book VQA — a system that improves a frozen vision-language model at inference time using retrieval, no retraining required.

**[Slide 2 — Motivation]**
Standard VQA is a closed-book exam. The model sees one image and one question, and must answer purely from its internal weights. My question is: can it do better if it's allowed to look up similar past examples first?

**[Slide 5 — Architecture]**
The pipeline works like this. A query image is encoded by CLIP into a 512-dimensional vector. FAISS searches a 5,000-entry knowledge base of past VQA examples and returns the top-k most visually similar ones. Those Q&A pairs are prepended as hints to the BLIP-2 prompt. BLIP-2 generates the final answer.

**[Slide 8 — Main Results]**
The headline result: baseline BLIP-2 scores 42.33%. With image-only retrieval at alpha=1.0, we reach 51.33% — that's +9 percentage points absolute and 21% relative improvement. One key finding is that text-only retrieval slightly hurts, while image-only retrieval gives the full gain.

**[Slide 11 — Oracle]**
The most interesting finding: at k=1, only 24% of retrieved answers exactly match the ground truth, yet RAG accuracy at k=1 is 53.33%. The model is not copying retrieved answers. It's using them as visual and semantic priors to improve its own generation.

**[Slide 12 — Answer Type]**
Breaking down by question type: RAG strongly helps open-ended questions, +14 points. It slightly hurts yes/no questions, where BLIP-2 was already strong.

**[Slide 16 — Conclusion]**
To summarize: we built an open-book VQA system that improves a frozen 3B model by 9 points at inference time. Visual retrieval beats text retrieval. The model genuinely reasons over context rather than copying. Future work includes a larger eval set and more aggressive fine-tuning.

---

## 5-Minute Presentation Script

**[Slide 1 — Title]** (30 sec)
Today I'm presenting Open-Book VQA — improving visual question answering by giving a frozen BLIP-2 model access to a visual memory at inference time.

**[Slide 2 — Motivation]** (30 sec)
The core analogy is the closed-book exam. A student who can consult notes often does better than one who can't, especially on hard questions about rare topics. VQA models face the same problem — they must answer entirely from memorized weights, with no ability to look up related examples.

**[Slide 3 — Problem Statement]** (20 sec)
The task is standard VQAv2: given an image and a natural-language question, output a short answer. Evaluation uses VQA soft accuracy — which accounts for annotator disagreement by counting matches out of 10 annotators, capped at 3.

**[Slide 4 — Core Idea]** (20 sec)
The idea is simple. Before the model answers, retrieve the most visually similar past examples from a knowledge base. Prepend their Q&A pairs as hints. The model now has context to draw from.

**[Slide 5 — Architecture]** (45 sec)
The pipeline: CLIP ViT-B/32 encodes the query image as a 512-dimensional embedding. We maintain two FAISS indices — one for images, one for question text. We fuse the two similarity scores as: score equals alpha times image similarity plus 1 minus alpha times text similarity. The top-k results are formatted into a hints prompt: "Hints may be wrong. Similar image suggests: [answer]. Question: [q]. Answer:" BLIP-2 then generates the response.

**[Slide 6 — Scoring]** (20 sec)
The key parameter is alpha. Alpha=0 means text-only retrieval. Alpha=1 means image-only retrieval. We also have an optional gate tau — if the top-1 score falls below tau, we skip retrieval and fall back to baseline.

**[Slide 7 — Setup]** (15 sec)
We evaluate on 100 VQAv2 validation samples, held fixed across all runs for fair comparison. The knowledge base has 5,000 entries built from the training split.

**[Slide 8 — Main Results]** (45 sec)
The main result table. Baseline is 42.33%. Text-only retrieval at alpha=0 actually slightly hurts — 41.33%. Balanced at alpha=0.5 gives 46.33%. The best result is image-only retrieval at alpha=1.0: 51.33%. That's a +9 point absolute gain and 21% relative improvement. The tau gate adds a small further improvement in some configurations but isn't critical.

**[Slide 9 — Why Visual Wins]** (30 sec)
Why does image retrieval win? VQA answers are grounded in visual content. Similar images — same scene, same object type, same color palette — tend to have similar answers. Short VQA questions like "what color is this?" are nearly identical across completely different images, so text retrieval doesn't discriminate. CLIP image embeddings cluster by visual semantics in a way that correlates with the answer space.

**[Slide 10 — Top-k]** (30 sec)
The top-k ablation shows that more retrieved examples give better accuracy: k=1 gives 53%, k=3 gives 68%, k=5 gives 75%, k=10 gives 91%. The k=10 result is essentially a 10-shot prompt and should be interpreted as a controlled upper-bound exploration. I'd cite k=3 as the practically useful operating point.

**[Slide 11 — Oracle]** (30 sec)
I measured retrieved-answer overlap — how often the exact ground-truth answer appears among the retrieved examples. At k=1, only 24% of the time. Yet RAG accuracy at k=1 is 53%. The model outperforms answer overlap at k=1 and k=3. This shows the model is not just copying. It's combining BLIP-2 visual reasoning with retrieved examples as contextual priors.

**[Slide 12 — Answer Type]** (20 sec)
Breaking down by question type: yes/no questions drop slightly from 85% to 82% — RAG adds noise where the model was already strong. Open-ended questions jump from 27% to 40%, a +14 point gain. Open-ended questions are the main driver of the overall improvement.

**[Slide 13 — Helpful/Harmful]** (15 sec)
RAG corrects 21 examples and breaks 9. A 2.3-to-1 helpful-to-harmful ratio. 70% of examples are unchanged. This is a healthy signal.

**[Slide 15 — Fine-Tuning]** (20 sec)
I also attempted QLoRA fine-tuning on RAG-format prompts. Training loss decreased, showing the model learned the prompt format. But inference accuracy did not improve enough to displace inference-time RAG as the main result. The OPT-2.7B backbone is largely frozen, and lightweight adapters were insufficient to significantly shift generation behavior. This is future work.

**[Slide 16 — Conclusion]** (15 sec)
To summarize: open-book VQA improves frozen BLIP-2 by 9 points at inference time. Visual retrieval beats text retrieval. The model reasons over context rather than copying. Main limitation is that context can hurt high-confidence questions, and the generator doesn't always use retrieved evidence reliably.

---

## Q&A Preparation

---

**Q: Why does image retrieval beat text retrieval?**

VQA answers are grounded in visual content, not in question syntax. Similar images — same scene type, same objects, same colors — tend to have similar answer distributions. Short VQA questions like "what color is this?" or "how many people?" are nearly identical across completely different visual scenes. Text retrieval finds questions that look alike syntactically but reference totally different images, so the retrieved answers are noise. CLIP image embeddings cluster by visual semantics in a way that directly correlates with the answer space. That's why alpha=1.0 wins.

---

**Q: Is the model just copying retrieved answers?**

No. At k=1, the retrieved-answer overlap — how often the exact ground-truth answer appears in the top-1 retrieved result — is only 24%. Yet RAG accuracy at k=1 is 53.33%. If the model were copying, you'd expect RAG accuracy to be bounded by answer overlap. Instead, RAG accuracy at k=1 exceeds answer overlap by 29 percentage points. The retrieved examples are acting as semantic and visual context — they help the model understand the answer space and format, even when the exact answer isn't present.

---

**Q: Why does text-only retrieval hurt?**

At alpha=0.0, accuracy drops to 41.33% — below the 42.33% baseline. The reason is that short VQA questions are generic. "What color is it?" and "How many are there?" appear across thousands of completely different images. Text retrieval finds questions with similar surface form but totally different visual referents. The retrieved answers become random noise that misleads BLIP-2 away from its correct baseline answer.

---

**Q: Why does RAG hurt yes/no questions?**

BLIP-2 is already very accurate on yes/no questions — 85% at baseline. The model has strong internal calibration for binary visual questions. When we prepend retrieved hints, those hints introduce open-ended answers and related-but-different scene descriptions that can shift the model away from its correct yes/no answer. The more the model is uncertain about a question, the more it benefits from hints. When it's already confident and correct, hints add noise.

---

**Q: Why are top-k results so high? Is there leakage?**

The k=10 result (90.67%) should be interpreted carefully. At k=10, we're giving BLIP-2 a 10-shot context — 10 Q&A pairs from visually similar images. This is a large amount of context for a 2.7B model. Oracle@10 is 92.33% — meaning 92% of the time, the correct answer is somewhere in the top-10 retrieved set. The k=10 RAG result is slightly below Oracle@10, which is the expected pattern. There is no ground-truth injection. The retrieval is purely CLIP cosine similarity. However, I'd frame this as a controlled upper-bound exploration rather than a practical operating point. k=3 is the main result.

Regarding leakage: the knowledge base and evaluation samples are from different parts of the dataset pipeline. A stricter future benchmark would enforce image_id-disjoint KB construction explicitly.

---

**Q: Is 100 samples enough?**

This is a controlled evaluation set used consistently across all ablations, so the comparisons between configurations are internally fair. The relative ordering of methods — image retrieval beating text retrieval, k=3 being better than k=1 — is robust because every comparison is on the same 100 samples. However, 100 samples is not enough to claim full VQAv2 benchmark-level generalization. A larger-scale evaluation is explicitly listed as future work.

---

**Q: Is there data leakage between the KB and eval set?**

The knowledge base is built from the VQAv2 training split, and evaluation is on the validation split. These are different splits. For an even stricter guarantee, the build_kb function in the codebase also supports explicit image_id-based exclusion — any eval image can be blacklisted from the KB. This was implemented but the primary result uses the train/validation split separation as the main guard. A fully rigorous benchmark should enforce image_id disjointness and report it explicitly.

---

**Q: Why did fine-tuning not help?**

BLIP-2 uses a frozen ViT encoder and a frozen OPT-2.7B language model. Only the Q-Former bridge module is trainable during standard instruction tuning. QLoRA adds LoRA adapters with rank 16 on top of the frozen LM, but the number of trainable parameters is small relative to the full 2.7B backbone. The model learned to follow the hints-format prompt — training loss did decrease — but it couldn't significantly shift how OPT-2.7B generates answers from that format. This is a known limitation of lightweight adaptation of large frozen language models. More compute, larger rank, or partial unfreezing of the LM would be needed.

---

**Q: What would you improve next?**

Five things: First, scale the knowledge base from 5k to 50k+ entries and verify the improvement holds. Second, enforce strict image_id-disjoint KB construction and report leakage explicitly. Third, evaluate on a larger set — at least 1,000 samples — to confirm the alpha sweep findings. Fourth, try a trainable retriever that learns which visual features matter for answer prediction, rather than off-the-shelf CLIP. Fifth, revisit fine-tuning with a partially unfrozen LM layer and more training data.

---

**Q: Why use CLIP and FAISS specifically?**

CLIP is the natural choice because it's trained on image-text pairs and produces semantically rich image embeddings that cluster by visual content — which is what we need for visual retrieval. It's also pre-trained, fast, and works without any domain-specific fine-tuning. FAISS IndexFlatIP does exact inner-product search over L2-normalized embeddings, which is equivalent to cosine similarity. It's exact (no approximation error), fast enough for a 5k-entry KB, and straightforward to implement. For larger KBs (100k+), you'd swap to FAISS IVF or HNSW for approximate search.

---

*End of speaker script.*
