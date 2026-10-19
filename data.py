"""Dataset loading, chat-template formatting, and prompt-token loss masking.

We use databricks/databricks-dolly-15k (CC BY-SA 3.0, ungated). Records with
a context field are rendered as "context + instruction"; the rest as plain
instructions. Train/val/test are disjoint fixed-seed splits, and checkpoint
selection uses val only — test is read exactly once in evaluate.py.
"""

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and accurately."


def build_messages(example: dict) -> list[dict]:
    instruction = example["instruction"].strip()
    context = (example.get("context") or "").strip()
    user = f"{context}\n\n{instruction}" if context else instruction
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def load_splits(train_size: int, val_size: int, test_size: int, seed: int = 42):
    # Set DATASET_JSONL to a local .jsonl with instruction/context/response fields
    # to run offline or on your own data.
    local = os.environ.get("DATASET_JSONL")
    if local:
        ds = load_dataset("json", data_files=local, split="train")
    else:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    # Drop very long responses so a 512-token window keeps most of the answer.
    ds = ds.filter(lambda ex: len(ex["response"]) < 1200)
    ds = ds.shuffle(seed=seed)
    n = train_size + val_size + test_size
    assert n <= len(ds), f"asked for {n} rows, dataset has {len(ds)}"
    ds = ds.select(range(n))
    return {
        "train": ds.select(range(0, train_size)),
        "val": ds.select(range(train_size, train_size + val_size)),
        "test": ds.select(range(train_size + val_size, n)),
    }


def tokenize_with_mask(example: dict, tokenizer, max_len: int) -> dict:
    """Return input_ids/attention_mask/labels where prompt tokens have label -100
    so the loss is computed only on the assistant's response."""
    messages = build_messages(example)
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["response"].strip()}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    # Sanity: the full sequence must start with the prompt tokens.
    assert full_ids[: len(prompt_ids)] == prompt_ids, "chat template prefix mismatch"

    full_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[:max_len]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class PadCollator:
    """Dynamic padding: each batch is padded to its own longest row, not to
    max_len. This is the single biggest throughput win over padding='max_length'."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            pad = longest - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
        }


def prepare(tokenizer, max_len: int, train_size: int, val_size: int, test_size: int, seed: int = 42):
    raw = load_splits(train_size, val_size, test_size, seed)
    cols = raw["train"].column_names
    tok = {
        k: v.map(lambda ex: tokenize_with_mask(ex, tokenizer, max_len), remove_columns=cols)
        for k, v in raw.items()
    }
    return raw, tok
