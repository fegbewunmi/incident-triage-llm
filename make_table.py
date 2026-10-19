"""Collect results/*.json into Markdown tables and print them (paste into README)."""

import glob
import os

from common import load_json


def main(results_dir="results"):
    trains = {os.path.basename(p)[6:-5]: load_json(p) for p in sorted(glob.glob(f"{results_dir}/train_*.json"))}
    evals = {os.path.basename(p)[5:-5]: load_json(p) for p in sorted(glob.glob(f"{results_dir}/eval_*.json"))}
    benches = {os.path.basename(p)[6:-5]: load_json(p) for p in sorted(glob.glob(f"{results_dir}/bench_*.json"))}

    if trains:
        model = next(iter(trains.values()))["model_id"]
        device = next(iter(trains.values()))["device"]
        print(f"### Fine-tuning runs — `{model}` on `{device}`\n")
        print("| run | rank | grad ckpt | trainable params | % of total | train time | samples/s | peak mem (GB) | best val loss | base test PPL | tuned test PPL | PPL ↓ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, t in trains.items():
            e = evals.get(name, {})
            print(
                f"| {name} | {t['rank']} | {'yes' if t['grad_ckpt'] else 'no'} | {t['trainable_params']:,} "
                f"| {t['trainable_pct']:.3f}% | {t['train_runtime_s'] / 60:.1f} min | {t['train_samples_per_s']} "
                f"| {t['peak_train_memory_gb']} | {t['best_val_loss']} "
                f"| {e.get('base_test_ppl', '—')} | {e.get('tuned_test_ppl', '—')} | {e.get('ppl_reduction_pct', '—')}% |"
            )
        print()

    for name, b in benches.items():
        print(f"### Inference benchmark — run `{name}`, {b['max_new_tokens']} new tokens, `{b['device']}` {b['dtype']}\n")
        print("| variant | median latency (s) | decode tok/s | memory (GB) |")
        print("|---|---|---|---|")
        for k, label in [("base", "base model"), ("lora_unmerged", "base + LoRA (unmerged)"), ("lora_merged", "LoRA merged into base")]:
            r = b[k]
            print(f"| {label} | {r['median_latency_s']} | {r['median_decode_tok_per_s']} | {r['memory_gb']} |")
        print(f"\nMerging removed the adapter overhead: **{b['merge_speedup_pct']}%** lower latency than unmerged.\n")


if __name__ == "__main__":
    main()
