from __future__ import annotations

import torch


@torch.no_grad()
def naive_generate(model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None):
    """Greedy decoding with NO KV cache: re-runs the full sequence every step.

    This is rung zero of the optimization ladder. Keep it simple and correct;
    it's the reference every later optimization is checked against.
    """
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    return ids
