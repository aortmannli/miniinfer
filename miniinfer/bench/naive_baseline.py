"""Rung zero: tokens/sec for the no-cache greedy loop.   python -m miniinfer.bench.naive_baseline"""
from __future__ import annotations

import time

import torch
from transformers import AutoConfig, AutoTokenizer

from miniinfer.engine.generate import naive_generate
from miniinfer.model import Qwen3Config, Qwen3ForCausalLM, load_weights
from miniinfer.bench.results import log_result

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Explain how a transformer language model generates text, step by step."
NEW_TOKENS = 128


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = Qwen3Config.from_hf(AutoConfig.from_pretrained(MODEL_ID))
    model = load_weights(Qwen3ForCausalLM(cfg), MODEL_ID).float().to(device).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)

    naive_generate(model, ids, 8)  # warmup
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = naive_generate(model, ids, NEW_TOKENS)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n_new = out.shape[1] - ids.shape[1]
    log_result(
        "naive_no_cache",
        metrics={"tokens_per_sec": n_new / dt, "new_tokens": n_new, "seconds": dt},
        settings={"model": MODEL_ID, "dtype": "fp32", "batch_size": 1, "prompt_tokens": ids.shape[1]},
    )


if __name__ == "__main__":
    main()
