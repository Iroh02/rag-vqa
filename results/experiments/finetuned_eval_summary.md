# Fine-Tuned (QLoRA) Evaluation — RAG-VQA

Eval set: 100 validation samples (offset=5000), 20k diverse KB, top_k=3, α=1.0.
Same configuration as the 59.33% headline. Only change: load LoRA adapter on top of frozen BLIP-2.

QLoRA config (from `results/checkpoints/rag_checkpoint_epoch_*/adapter_config.json`):
- target modules: `q_proj`, `v_proj` of OPT-2.7B language head only
- rank `r = 16`, `lora_alpha = 32`, dropout 0.05
- 4-bit base + 16-bit LoRA adapter weights

## Results

| Mode | Frozen BLIP-2 | LoRA epoch 1 | LoRA epoch 5 |
|---|---:|---:|---:|
| Baseline (no retrieval) | 42.33% | **0.00%** | **2.00%** |
| RAG, hints prompt format (training format) | — | 2.33% | 3.33% |
| RAG, `current_prompt` Q/A few-shot | **59.33%** | 2.67% | 13.33% |

**Δ vs frozen + RAG headline (59.33%):** −46.00 (best fine-tuned variant: epoch-5 with `current_prompt`).

## Why fine-tuning fails here — mode collapse

Sampled outputs from epoch-5 LoRA show the failure mode is **degenerate repetition**:

| Question | Ground truth | Fine-tuned output |
|---|---|---|
| Is there snow on the ground? | no | `nononono` |
| What color is the sky? | blue | `blueblue` |
| Is it sunny or cloudy? | sunny | `cloudyyesyesyesyesyesyes` |
| How cold is it? | cold / 15 degrees / freezing | `verycold0f0f0f0f` |
| Is the skier going downhill? | no / no / yes | `yesyesyesyesyesyesyes` |

The adapter learned the answer (`blue`, `cloudy`, `cold` are all there) but cannot stop generating. Token repetition / off-distribution suffixes appear in nearly every prediction. There's a `_clean_hints_output()` post-processor in `src/model.py` for exactly this pattern, and even with it, the rag-hints variant only reaches 3.33%.

## Diagnosis

- **Adapter scope is too narrow** — `q_proj`+`v_proj` only on OPT-2.7B leaves the FFN and embedding layers frozen. The model can shift attention but can't shift the output distribution shape. The result: it tries to "follow" the new format but loses generation discipline.
- **Training data is too small / hints prompt is too narrow** — fine-tuning on the hints format apparently caused mode collapse onto short answers without teaching when to stop.
- **Quantization × LoRA × frozen Q-Former** — three lossy abstractions stacked. Each is fine alone; together they don't leave enough capacity for stable adaptation on this dataset.

This is the expected outcome of an under-resourced fine-tune: training loss does decrease (the model learns the prompt format), but inference quality collapses because the regularisation isn't enough to maintain coherent text generation.

## Bottom line for the report

> Fine-tuning was attempted (5-epoch QLoRA, rank 16, on q/v projections of OPT-2.7B). It **does not displace inference-time RAG** as the main result. Best fine-tuned variant scores 13.33% vs the frozen + RAG headline at 59.33%. The adapter learns the right answer tokens but loses generation discipline (mode collapse / repetition). Closing the gap would require partial unfreezing of the language model, more training data, or a fundamentally different adaptation regime — listed as future work.

## How to reproduce

```
# stop the API to free the GPU
python -m scripts.eval_finetuned --epoch 5    # ~5 min
python -m scripts.eval_finetuned --epoch 1    # ~3 min
```

Outputs:
- `results/experiments/finetuned_eval_epoch{1,5}.json`
- `results/experiments/finetuned_eval_epoch{1,5}_detail.csv`
- this file: `results/experiments/finetuned_eval_summary.md`
