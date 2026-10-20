"""Shared helpers: device selection, dtype, memory tracking, timing."""

import json
import os
import time
from contextlib import contextmanager

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # bf16 on Apple Silicon (M2+) and on Ampere+ CUDA. fp16 on older CUDA (e.g. T4).
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_gb(device: torch.device) -> float:
    """Peak device memory since the last reset (GB). MPS has no peak counter, so
    there we report driver-allocated memory (tensors + allocator cache); sample
    it at the point of interest, e.g. at the end of each training step."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timer():
    """Usage: with timer() as t: ...; t['s'] holds elapsed seconds."""
    out = {}
    start = time.perf_counter()
    yield out
    out["s"] = time.perf_counter() - start


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
