"""LoRA fine-tuning of Llama 3.2 as the Orion Commerce alert-triage model.

Runs on Apple Silicon (MPS), CUDA, or CPU with no bitsandbytes dependency:
frozen base weights are held in bf16 (fp16 on pre-Ampere CUDA), LoRA adapter
weights train in fp32. Checkpoint selection uses the validation split only.

Example:
  python train.py --run_name triage_r8 --rank 8
  python train.py --run_name triage_r8_ckpt --rank 8 --grad_ckpt
  python train.py --run_name triage_r16 --rank 16
"""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from common import peak_memory_gb, pick_device, pick_dtype, reset_peak_memory, save_json, timer
from data import PadCollator, prepare


class MemoryCallback(TrainerCallback):
    """Records peak device memory seen during training steps."""

    def __init__(self, device):
        self.device = device
        self.peak_gb = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        self.peak_gb = max(self.peak_gb, peak_memory_gb(self.device))
        # Dynamic padding gives every batch a different shape, which fragments
        # the MPS allocator cache until it runs out. Release the cache each step.
        if self.device.type == "mps":
            torch.mps.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--run_name", required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=None, help="defaults to 2*rank")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--train_limit", type=int, default=None, help="use only the first N training records")
    p.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="adapters")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    dtype = pick_dtype(device)
    alpha = args.alpha or 2 * args.rank
    print(f"device={device} base_dtype={dtype} rank={args.rank} alpha={alpha} grad_ckpt={args.grad_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tok = prepare(tokenizer, args.data_dir, args.max_len, args.train_limit)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    model.config.use_cache = False  # incompatible with gradient checkpointing; re-enabled at inference
    model.to(device)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    # PEFT keeps adapter weights in fp32 by default when the base is bf16/fp16.
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    targs = TrainingArguments(
        output_dir=os.path.join("checkpoints", args.run_name),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=0.05,  # float in [0,1) = ratio of total steps
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    mem_cb = MemoryCallback(device)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok["train"],
        eval_dataset=tok["val"],  # validation only; test is never seen here
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[mem_cb],
    )

    reset_peak_memory(device)
    with timer() as t:
        train_out = trainer.train()
    val_metrics = trainer.evaluate()

    adapter_dir = os.path.join(args.out_dir, args.run_name)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    result = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "device": str(device),
        "base_dtype": str(dtype),
        "rank": args.rank,
        "alpha": alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
        "grad_ckpt": args.grad_ckpt,
        "lr": args.lr,
        "epochs": args.epochs,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_len": args.max_len,
        "data_dir": args.data_dir,
        "train_examples": len(tok["train"]),
        "val_examples": len(tok["val"]),
        "seed": args.seed,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "train_runtime_s": round(t["s"], 1),
        "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0), 3),
        "final_train_loss": round(train_out.training_loss, 4),
        "best_val_loss": round(val_metrics["eval_loss"], 4),
        "peak_train_memory_gb": round(mem_cb.peak_gb, 2),
        "adapter_dir": adapter_dir,
    }
    save_json(os.path.join(args.results_dir, f"train_{args.run_name}.json"), result)
    print(result)


if __name__ == "__main__":
    main()
