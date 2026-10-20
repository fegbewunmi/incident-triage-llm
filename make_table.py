"""Collect results/*.json into the Markdown tables for the README."""

import glob
import os

from common import load_json
from eval_triage import table


def main(results_dir="results"):
    for path in sorted(glob.glob(f"{results_dir}/eval_triage_*.json")):
        e = load_json(path)
        print(f"### Triage quality — run `{e['run_name']}`, {e['n']} alerts from `{os.path.basename(e['eval_file'])}`\n")
        print(table(e["results"]) + "\n")

    trains = [load_json(p) for p in sorted(glob.glob(f"{results_dir}/train_*.json"))]
    if trains:
        print("### Fine-tuning runs\n")
        print("| run | rank | grad ckpt | trainable params | % of total | train examples | train time | peak mem (GB) | best val loss |")
        print("|---|---|---|---|---|---|---|---|---|")
        for t in trains:
            print(f"| {t['run_name']} | {t['rank']} | {'yes' if t['grad_ckpt'] else 'no'} | {t['trainable_params']:,} "
                  f"| {t['trainable_pct']:.3f}% | {t.get('train_examples', '—')} | {t['train_runtime_s'] / 60:.1f} min "
                  f"| {t['peak_train_memory_gb']} | {t['best_val_loss']} |")
        print()

    for path in sorted(glob.glob(f"{results_dir}/bench_*.json")):
        b = load_json(path)
        print(f"### Inference — run `{b['run_name']}`, {b['new_tokens']} new tokens, `{b['device']}` {b['dtype']}\n")
        print("| variant | median latency (s) | decode tok/s | memory (GB) |")
        print("|---|---|---|---|")
        for k, label in [("base", "base model"), ("lora_unmerged", "base + LoRA (unmerged)"), ("lora_merged", "LoRA merged into base")]:
            r = b[k]
            print(f"| {label} | {r['median_latency_s']} | {r['median_decode_tok_per_s']} | {r['memory_gb']} |")
        print(f"\nMerging cut latency **{b['merge_speedup_pct']}%** vs the unmerged adapter.\n")


if __name__ == "__main__":
    main()
