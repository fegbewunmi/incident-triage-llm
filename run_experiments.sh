#!/usr/bin/env bash
# Full experiment matrix. On an M-series Mac with 32 GB, the 1B runs take
# roughly 20-40 min each for 3000 examples x 2 epochs; start with ONE run
# (e.g. just r8) to confirm everything works before launching the sweep.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

# 1. Rank sweep (quality vs trainable params)
for R in 4 8 16; do
  python train.py --run_name "r${R}" --rank "${R}" --model_id "${MODEL}"
done

# 2. Gradient checkpointing on vs off at r=8 (memory vs time trade-off)
python train.py --run_name r8_ckpt --rank 8 --grad_ckpt --model_id "${MODEL}"

# 3. Held-out test evaluation for every run
for RUN in r4 r8 r16 r8_ckpt; do
  python evaluate.py --run_name "${RUN}"
done

# 4. Inference benchmark on the chosen run
python benchmark.py --run_name r8

# 5. Build README tables
python make_table.py
