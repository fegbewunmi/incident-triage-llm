"""Inference benchmark on real triage prompts: base model vs base + unmerged
LoRA adapter vs adapter merged into the base weights.

Merging folds the low-rank update into the base matrices (W' = W + BA * scale),
removing the per-layer adapter matmuls from every forward pass. Generation
length is fixed so tokens/s is comparable across variants.

Example:
  python benchmark.py --run_name triage_r8
"""

import argparse
import os
import statistics

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_json, peak_memory_gb, pick_device, pick_dtype, reset_peak_memory, save_json, sync, timer
from data import chat_ids, load_jsonl, prompt_messages


@torch.no_grad()
def bench(model, tokenizer, prompts, device, new_tokens, warmup=2) -> dict:
    model.eval()
    model.config.use_cache = True
    lat, tps = [], []
    for i, ids in enumerate(prompts):
        x = torch.tensor([ids], device=device)
        sync(device)
        with timer() as t:
            out = model.generate(x, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            sync(device)
        if i < warmup:
            continue
        lat.append(t["s"])
        tps.append((out.shape[1] - x.shape[1]) / t["s"])
    return {"median_latency_s": round(statistics.median(lat), 3),
            "median_decode_tok_per_s": round(statistics.median(tps), 1),
            "memory_gb": round(peak_memory_gb(device), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--n_prompts", type=int, default=22, help="first 2 are warmup")
    p.add_argument("--new_tokens", type=int, default=96, help="about the length of one triage JSON")
    args = p.parse_args()

    tr = load_json(os.path.join(args.results_dir, f"train_{args.run_name}.json"))
    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(tr["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    alerts = load_jsonl(os.path.join(tr["data_dir"], "test.jsonl"))[: args.n_prompts]
    prompts = [chat_ids(tokenizer, prompt_messages(r["alert"]), True) for r in alerts]

    res = {"run_name": args.run_name, "device": str(device), "dtype": str(dtype), "new_tokens": args.new_tokens}
    reset_peak_memory(device)
    base = AutoModelForCausalLM.from_pretrained(tr["model_id"], dtype=dtype).to(device)
    res["base"] = bench(base, tokenizer, prompts, device, args.new_tokens)
    print("base:", res["base"])

    tuned = PeftModel.from_pretrained(base, tr["adapter_dir"]).to(device)
    res["lora_unmerged"] = bench(tuned, tokenizer, prompts, device, args.new_tokens)
    print("lora_unmerged:", res["lora_unmerged"])

    merged = tuned.merge_and_unload()
    res["lora_merged"] = bench(merged, tokenizer, prompts, device, args.new_tokens)
    print("lora_merged:", res["lora_merged"])

    merged_dir = os.path.join("merged", args.run_name)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    res["merged_dir"] = merged_dir
    u, m = res["lora_unmerged"]["median_latency_s"], res["lora_merged"]["median_latency_s"]
    res["merge_speedup_pct"] = round(100 * (u - m) / u, 1)
    save_json(os.path.join(args.results_dir, f"bench_{args.run_name}.json"), res)
    print(res)


if __name__ == "__main__":
    main()
