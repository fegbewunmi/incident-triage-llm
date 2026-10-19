"""Held-out test evaluation: response-token perplexity for the base model vs the
fine-tuned adapter, plus side-by-side generations on a few test prompts.

The test split is constructed with the same seed/sizes as train.py, so it is
disjoint from the data used for training and checkpoint selection.

Example:
  python evaluate.py --run_name r8
"""

import argparse
import math
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_json, pick_device, pick_dtype, save_json
from data import PadCollator, build_messages, prepare


@torch.no_grad()
def response_perplexity(model, dataset, collator, device, batch_size=4) -> float:
    """exp(mean cross-entropy over response tokens only)."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(dataset), batch_size):
        batch = collator([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        # shift so token t predicts token t+1
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += (shift_labels != -100).sum().item()
    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def generate(model, tokenizer, example, device, max_new_tokens=160) -> str:
    model.eval()
    ids = tokenizer.apply_chat_template(
        build_messages(example), tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--n_samples", type=int, default=5, help="test prompts to show generations for")
    args = p.parse_args()

    train_result = load_json(os.path.join(args.results_dir, f"train_{args.run_name}.json"))
    model_id = train_result["model_id"]
    device = pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw, tok = prepare(
        tokenizer,
        train_result["max_len"],
        train_result["train_size"],
        train_result["val_size"],
        train_result["test_size"],
        train_result["seed"],
    )
    test_raw, test_tok = raw["test"], tok["test"]
    collator = PadCollator(tokenizer.pad_token_id)

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    base_ppl = response_perplexity(base, test_tok, collator, device)
    print(f"base test perplexity: {base_ppl:.3f}")

    samples = [test_raw[i] for i in range(args.n_samples)]
    base_gen = [generate(base, tokenizer, ex, device) for ex in samples]

    tuned = PeftModel.from_pretrained(base, train_result["adapter_dir"]).to(device)
    tuned_ppl = response_perplexity(tuned, test_tok, collator, device)
    print(f"tuned test perplexity: {tuned_ppl:.3f}")
    tuned_gen = [generate(tuned, tokenizer, ex, device) for ex in samples]

    out = {
        "run_name": args.run_name,
        "test_size": len(test_tok),
        "base_test_ppl": round(base_ppl, 3),
        "tuned_test_ppl": round(tuned_ppl, 3),
        "ppl_reduction_pct": round(100 * (base_ppl - tuned_ppl) / base_ppl, 1),
        "samples": [
            {
                "instruction": ex["instruction"],
                "reference": ex["response"],
                "base": b,
                "tuned": t,
            }
            for ex, b, t in zip(samples, base_gen, tuned_gen)
        ],
    }
    save_json(os.path.join(args.results_dir, f"eval_{args.run_name}.json"), out)

    for s in out["samples"]:
        print("\n" + "=" * 80)
        print("PROMPT:   ", s["instruction"])
        print("-" * 80)
        print("BASE:     ", s["base"][:400])
        print("-" * 80)
        print("TUNED:    ", s["tuned"][:400])
        print("-" * 80)
        print("REFERENCE:", s["reference"][:400])


if __name__ == "__main__":
    main()
