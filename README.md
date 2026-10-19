# LoRA fine-tuning and inference optimization of Llama 3.2

Parameter-efficient supervised fine-tuning (SFT) of `meta-llama/Llama-3.2-1B-Instruct`
on an instruction-following dataset, with a held-out test evaluation, a LoRA rank
sweep, a gradient-checkpointing memory/time trade-off, and an inference benchmark
of merged vs unmerged adapters. Everything runs on Apple Silicon (MPS) or CUDA
with no quantization library required.

## What's in here

| file | purpose |
|---|---|
| `data.py` | Loads `databricks/databricks-dolly-15k`, renders Llama 3 chat template, masks prompt tokens out of the loss, dynamic-padding collator |
| `train.py` | LoRA SFT with HF `Trainer`; bf16 frozen base + fp32 adapters; optional gradient checkpointing; logs peak memory, throughput, val loss |
| `evaluate.py` | Response-token perplexity on the **held-out test split** for base vs tuned, plus side-by-side generations |
| `benchmark.py` | Latency / tok-per-s / memory for base, base+LoRA, and `merge_and_unload()` merged weights |
| `run_experiments.sh` | Rank sweep (4/8/16), checkpointing on/off, eval, benchmark |
| `make_table.py` | Builds the results tables below from `results/*.json` |

## Method

**Data.** 3,000 train / 300 val / 500 test examples, fixed-seed disjoint splits.
Each record is rendered through the model's own chat template (system + user +
assistant). Labels for prompt tokens are set to `-100`, so the loss is computed
only on the assistant's response — the model learns to answer, not to repeat the
question.

**Fine-tuning.** LoRA on the attention projections (`q_proj, k_proj, v_proj, o_proj`),
`alpha = 2 × rank`, dropout 0.05, LR 2e-4 with cosine decay and 5% warmup,
effective batch 16 (2 per step × 8 gradient accumulation), 2 epochs, max 512 tokens.
Checkpoint selection uses validation loss only; the test split is read exactly
once, in `evaluate.py`.

**Optimizations.**
- *Mixed precision without AMP*: frozen base weights live in bf16, trainable
  adapter weights in fp32 (PEFT's default adapter dtype). Halves base-model
  memory vs fp32 and works on MPS, where `bitsandbytes` 4-bit does not.
- *Dynamic padding*: batches pad to their own longest sequence instead of
  `max_length`, so short Dolly answers don't cost 512 tokens of compute.
- *Gradient checkpointing*: measured on/off to show the memory-vs-time trade-off.
- *Gradient accumulation*: keeps an effective batch of 16 within a small
  per-step footprint.
- *Adapter merging for inference*: `merge_and_unload()` folds the low-rank
  update into the base weights, removing the per-layer adapter matmuls from
  every forward pass; benchmarked against the unmerged adapter.

## Results

<!-- Run `python make_table.py` and paste its output here. -->

_Results pending — see `run_experiments.sh`._

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Llama 3.2 is gated: accept the license on its Hugging Face page, then
huggingface-cli login
```

Quick single run (~20–40 min on an M-series Mac for 1B):

```bash
python train.py --run_name r8 --rank 8
python evaluate.py --run_name r8
python benchmark.py --run_name r8
python make_table.py
```

Full matrix: `bash run_experiments.sh`.

Other models work unchanged, e.g. `--model_id meta-llama/Llama-3.2-3B-Instruct`
(fits in 32 GB unified memory with `--grad_ckpt`) or an ungated alternative
such as `--model_id Qwen/Qwen2.5-1.5B-Instruct`.

## Notes

- Set `DATASET_JSONL=/path/to/file.jsonl` (fields: `instruction`, `context`, `response`)
  to train on your own data or run offline.
- Base dtype is chosen per device: bf16 on MPS and Ampere+ CUDA, fp16 on older
  CUDA such as a T4.
- MPS has no peak-memory counter, so on Mac `peak_train_memory_gb` is the
  driver-allocated memory (including the allocator cache) sampled at the end
  of each optimizer step. The MPS cache is released after each step to stop
  fragmentation from variable-length batches.
- `use_cache` is disabled during training (required for gradient checkpointing)
  and re-enabled for generation.
