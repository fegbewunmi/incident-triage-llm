"""Training data: chat formatting of triage records and prompt-token loss masking.

Each record becomes [system, user=alert JSON, assistant=triage JSON]. Prompt
tokens are labeled -100 so the loss covers only the triage JSON the model must
produce. The test split is never touched here.
"""

import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset

from schema import SHORT_SYSTEM, render_alert, render_label

# Llama 3.x chat templates stamp "Today Date: <today>" into the system header.
# Pin it so training and evaluation prompts are identical regardless of the day.
FIXED_DATE = "01 Jan 2026"


def chat_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_dict=False, date_string=FIXED_DATE,
    )


def prompt_messages(alert: dict, system: str = SHORT_SYSTEM, shots: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for s in shots or []:
        msgs += [{"role": "user", "content": render_alert(s["alert"])},
                 {"role": "assistant", "content": render_label(s["label"])}]
    msgs.append({"role": "user", "content": render_alert(alert)})
    return msgs


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def tokenize_with_mask(record: dict, tokenizer, max_len: int) -> dict | None:
    msgs = prompt_messages(record["alert"])
    prompt_ids = chat_ids(tokenizer, msgs, add_generation_prompt=True)
    full_ids = chat_ids(tokenizer, msgs + [{"role": "assistant", "content": render_label(record["label"])}], False)
    assert full_ids[: len(prompt_ids)] == prompt_ids, "chat template prefix mismatch"
    if len(full_ids) > max_len:
        return None  # never truncate: a cut-off target teaches broken JSON
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
    }


@dataclass
class PadCollator:
    """Dynamic padding: each batch pads to its own longest row."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            pad = longest - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask), "labels": torch.tensor(labels)}


def prepare(tokenizer, data_dir: str, max_len: int, train_limit: int | None = None) -> dict:
    out = {}
    for split in ("train", "val"):
        rows = load_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
        if split == "train" and train_limit:
            rows = rows[:train_limit]
        toks = [tokenize_with_mask(r, tokenizer, max_len) for r in rows]
        kept = [t for t in toks if t is not None]
        lengths = [len(t["input_ids"]) for t in kept]
        print(f"{split}: {len(kept)}/{len(rows)} kept (dropped {len(rows) - len(kept)} over {max_len} tokens), "
              f"tokens median={sorted(lengths)[len(lengths) // 2]} max={max(lengths)}")
        out[split] = Dataset.from_list(kept)
    return out
