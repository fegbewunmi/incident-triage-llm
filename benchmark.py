"""Inference benchmark: base model vs base+LoRA (unmerged) vs merged weights.

Merging folds the low-rank update into the base matrices (W' = W + BA*scale),
so the merged model has the same parameter count and latency as the base model
while keeping the fine-tuned behaviour. Reports median per-request latency,
decode throughput, and device memory.

Example:
  python benchmark.py --run_name r8
"""

import argparse
import os
import statistics

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_json, peak_memory_gb, pick_device, pick_dtype, reset_peak_memory, save_json, sync, timer
from data import build_messages

PROMPTS = [
    {"instruction": "Explain the difference between a process and a thread.", "context": ""},
    {"instruction": "Give three tips for writing a clear commit message.", "context": ""},
    {"instruction": "What causes the seasons on Earth?", "context": ""},
    {"instruction": "Summarize the plot of a heist story in two sentences.", "context": ""},
]


@torch.no_grad()
def bench(model, tokenizer, device, max_new_tokens, warmup=1, repeats=3) -> dict:
    model.eval()
    model.config.use_cache = True
    latencies, tok_per_s = [], []
    for i, ex in enumerate(PROMPTS * repeats):
        ids = tokenizer.apply_chat_template(
            build_messages(ex), tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(device)
        sync(device)
        with timer() as t:
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,  # fixed length so throughput is comparable
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            sync(device)
        if i < warmup:
            continue
        new_tokens = out.shape[1] - ids.shape[1]
        latencies.append(t["s"])
        tok_per_s.append(new_tokens / t["s"])
    return {
        "median_latency_s": round(statistics.median(latencies), 3),
        "median_decode_tok_per_s": round(statistics.median(tok_per_s), 1),
        "memory_gb": round(peak_memory_gb(device), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    train_result = load_json(os.path.join(args.results_dir, f"train_{args.run_name}.json"))
    model_id, adapter_dir = train_result["model_id"], train_result["adapter_dir"]
    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"run_name": args.run_name, "device": str(device), "dtype": str(dtype), "max_new_tokens": args.max_new_tokens}

    # 1. Base model
    reset_peak_memory(device)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    results["base"] = bench(base, tokenizer, device, args.max_new_tokens)
    print("base:", results["base"])

    # 2. Base + unmerged LoRA (extra matmuls per adapted layer on every forward)
    tuned = PeftModel.from_pretrained(base, adapter_dir).to(device)
    results["lora_unmerged"] = bench(tuned, tokenizer, device, args.max_new_tokens)
    print("lora_unmerged:", results["lora_unmerged"])

    # 3. Merged: fold the adapter into the base weights, drop the adapter modules
    merged = tuned.merge_and_unload()
    results["lora_merged"] = bench(merged, tokenizer, device, args.max_new_tokens)
    print("lora_merged:", results["lora_merged"])

    merged_dir = os.path.join("merged", args.run_name)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    results["merged_dir"] = merged_dir

    u, m = results["lora_unmerged"], results["lora_merged"]
    results["merge_speedup_pct"] = round(100 * (u["median_latency_s"] - m["median_latency_s"]) / u["median_latency_s"], 1)
    save_json(os.path.join(args.results_dir, f"bench_{args.run_name}.json"), results)
    print(results)


if __name__ == "__main__":
    main()
