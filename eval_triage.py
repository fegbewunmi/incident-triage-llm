"""Evaluate triage quality, latency and cost on the human-reviewed gold set.

Variants:
  base_zero        untuned Llama, full rules prompt
  base_few         untuned Llama, full rules prompt + 3 worked examples
  tuned            fine-tuned Llama (LoRA merged), one-line prompt
  gemini           Gemini 2.5 Flash, full rules prompt, JSON mode, thinking off
  gemini_thinking  same, with dynamic thinking
  claude           Claude (Anthropic API), full rules prompt

Outputs that are not valid triage JSON count as wrong on every field and are
treated as investigate=true (the safe fallback: run the full investigation).

Example:
  python eval_triage.py --run_name triage_r8
  python eval_triage.py --run_name triage_r8 --variants base_zero,base_few,tuned
"""

import argparse
import os
import statistics
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_json, pick_device, pick_dtype, save_json, sync, timer
from data import chat_ids, load_jsonl, prompt_messages
from schema import FAMILIES, FULL_SYSTEM, SHORT_SYSTEM, parse_output, render_alert
from teachers import make_client

SCORED = ["failure_family", "severity", "origin_service"]


def pick_shots(train: list[dict], families=("failed_deployment", "dependency_failure", "noise")) -> list[dict]:
    shots = []
    for fam in families:
        shots.append(next(r for r in train if r["label"]["failure_family"] == fam))
    return shots


@torch.no_grad()
def run_local(model, tokenizer, records, device, system, shots=None, max_new_tokens=200):
    model.eval()
    model.config.use_cache = True
    preds = []
    for i, r in enumerate(records):
        ids = torch.tensor([chat_ids(tokenizer, prompt_messages(r["alert"], system, shots), True)], device=device)
        sync(device)
        with timer() as t:
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            sync(device)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        preds.append({"text": text, "latency_s": t["s"], "input_tokens": ids.shape[1],
                      "output_tokens": out.shape[1] - ids.shape[1], "thinking_tokens": 0})
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(records)}")
    return preds


def run_api(client, records, json_mode=True):
    preds = []
    for i, r in enumerate(records):
        text, usage, latency = client.complete(FULL_SYSTEM, render_alert(r["alert"]), json_mode=json_mode, max_tokens=2048)
        preds.append({"text": text, "latency_s": latency, **usage})
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(records)}")
    return preds


def macro_f1(golds, preds, labels):
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def score(records, preds, in_price=0.0, out_price=0.0) -> dict:
    n = len(records)
    status = Counter()
    correct = Counter()
    exact = 0
    gold_fam, pred_fam = [], []
    noise_total = noise_caught = incidents_total = incidents_missed = 0
    confusion = defaultdict(Counter)
    for r, p in zip(records, preds):
        parsed, st = parse_output(p["text"])
        p["parsed"], p["status"] = parsed, st
        status[st] += 1
        g = r["label"]
        pv = parsed or {}
        for k in SCORED:
            correct[k] += pv.get(k) == g[k]
        exact += all(pv.get(k) == g[k] for k in SCORED)
        gold_fam.append(g["failure_family"])
        pred_fam.append(pv.get("failure_family", "invalid"))
        confusion[g["failure_family"]][pv.get("failure_family", "invalid")] += 1
        pred_investigate = pv.get("investigate", True)
        if g["investigate"]:
            incidents_total += 1
            incidents_missed += not pred_investigate
        else:
            noise_total += 1
            noise_caught += not pred_investigate

    lat = sorted(p["latency_s"] for p in preds)
    in_tok = statistics.mean(p["input_tokens"] for p in preds)
    out_tok = statistics.mean(p["output_tokens"] + p.get("thinking_tokens", 0) for p in preds)
    cost_per_1k = 1000 * (in_tok * in_price + out_tok * out_price) / 1e6
    pct = lambda x: round(100 * x / n, 1)
    return {
        "n": n,
        "valid_json_pct": pct(status["ok"] + status["bad_schema"]),
        "valid_schema_pct": pct(status["ok"]),
        "family_acc_pct": pct(correct["failure_family"]),
        "family_macro_f1": round(macro_f1(gold_fam, pred_fam, FAMILIES), 3),
        "severity_acc_pct": pct(correct["severity"]),
        "origin_acc_pct": pct(correct["origin_service"]),
        "exact_match_pct": pct(exact),
        "noise_filtered_pct": round(100 * noise_caught / max(noise_total, 1), 1),
        "incidents_missed": incidents_missed,
        "incidents_total": incidents_total,
        "median_latency_s": round(lat[len(lat) // 2], 3),
        "p95_latency_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
        "mean_input_tokens": round(in_tok, 1),
        "mean_output_tokens": round(out_tok, 1),
        "cost_per_1k_alerts_usd": round(cost_per_1k, 4),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


LABELS = {
    "base_zero": "Llama 3.2 1B, untuned, rules prompt",
    "base_few": "Llama 3.2 1B, untuned, rules + 3 examples",
    "tuned": "**Llama 3.2 1B, fine-tuned (LoRA)**",
    "gemini": "Gemini 2.5 Flash, rules prompt",
    "gemini_thinking": "Gemini 2.5 Flash + thinking, rules prompt",
    "claude": "Claude, rules prompt",
}


def table(results: dict) -> str:
    rows = ["| model | valid JSON | family acc | family F1 | severity acc | origin acc | all 3 right | noise filtered | incidents missed | median latency | prompt tokens | $ / 1k alerts |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        rows.append(
            f"| {LABELS.get(name, name)} | {m['valid_schema_pct']}% | {m['family_acc_pct']}% | {m['family_macro_f1']} "
            f"| {m['severity_acc_pct']}% | {m['origin_acc_pct']}% | {m['exact_match_pct']}% | {m['noise_filtered_pct']}% "
            f"| {m['incidents_missed']}/{m['incidents_total']} | {m['median_latency_s']}s | {m['mean_input_tokens']:.0f} "
            f"| ${m['cost_per_1k_alerts_usd']} |"
        )
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True, help="training run whose adapter to evaluate")
    p.add_argument("--variants", default="base_zero,base_few,tuned,gemini,gemini_thinking")
    p.add_argument("--eval_file", default=None, help="defaults to data/gold.jsonl, else data/test.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--gemini_model", default="gemini-2.5-flash")
    p.add_argument("--claude_model", default="claude-haiku-5-5")
    # USD per 1M tokens. Check current pricing before quoting numbers.
    p.add_argument("--gemini_in_price", type=float, default=0.30)
    p.add_argument("--gemini_out_price", type=float, default=2.50)
    p.add_argument("--claude_in_price", type=float, default=1.00)
    p.add_argument("--claude_out_price", type=float, default=5.00)
    args = p.parse_args()

    tr = load_json(os.path.join(args.results_dir, f"train_{args.run_name}.json"))
    data_dir = tr["data_dir"]
    eval_file = args.eval_file or (os.path.join(data_dir, "gold.jsonl") if os.path.exists(os.path.join(data_dir, "gold.jsonl"))
                                   else os.path.join(data_dir, "test.jsonl"))
    records = load_jsonl(eval_file)[: args.limit]
    print(f"evaluating on {eval_file}: {len(records)} alerts")
    variants = args.variants.split(",")
    results, predictions = {}, {}

    local = [v for v in variants if v in ("base_zero", "base_few", "tuned")]
    if local:
        device = pick_device()
        dtype = pick_dtype(device)
        tokenizer = AutoTokenizer.from_pretrained(tr["model_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(tr["model_id"], dtype=dtype).to(device)
        shots = pick_shots(load_jsonl(os.path.join(data_dir, "train.jsonl")))
        for v in ("base_zero", "base_few"):
            if v in local:
                print(f"[{v}]")
                predictions[v] = run_local(model, tokenizer, records, device, FULL_SYSTEM, shots if v == "base_few" else None)
                results[v] = score(records, predictions[v])
        if "tuned" in local:
            print("[tuned]")
            model = PeftModel.from_pretrained(model, tr["adapter_dir"]).merge_and_unload().to(device)
            predictions["tuned"] = run_local(model, tokenizer, records, device, SHORT_SYSTEM)
            results["tuned"] = score(records, predictions["tuned"])
        del model

    for v in variants:
        if v == "gemini" or v == "gemini_thinking":
            print(f"[{v}]")
            client = make_client("gemini", args.gemini_model, thinking_budget=0 if v == "gemini" else -1)
            predictions[v] = run_api(client, records)
            results[v] = score(records, predictions[v], args.gemini_in_price, args.gemini_out_price)
        elif v == "claude":
            print(f"[{v}]")
            client = make_client("anthropic", args.claude_model)
            predictions[v] = run_api(client, records, json_mode=False)
            results[v] = score(records, predictions[v], args.claude_in_price, args.claude_out_price)

    # Merge into any earlier results for this run, so local and API variants can be run separately.
    out_path = os.path.join(args.results_dir, f"eval_triage_{args.run_name}.json")
    prev = load_json(out_path) if os.path.exists(out_path) else {}
    if prev.get("eval_file") != eval_file or prev.get("n") != len(records):
        prev = {}
    merged = {**prev.get("results", {}), **results}
    order = list(LABELS)
    ordered = {v: merged[v] for v in sorted(merged, key=lambda v: order.index(v) if v in order else 99)}
    md = table(ordered)
    print("\n" + md)
    out = {"run_name": args.run_name, "eval_file": eval_file, "n": len(records), "results": ordered,
           "predictions": {**prev.get("predictions", {}),
                           **{v: [{"id": r["id"], "gold": r["label"], **pr} for r, pr in zip(records, ps)]
                              for v, ps in predictions.items()}}}
    save_json(out_path, out)
    with open(os.path.join(args.results_dir, f"eval_triage_{args.run_name}.md"), "w") as f:
        f.write(md + "\n")


if __name__ == "__main__":
    main()
