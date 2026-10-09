"""Parity tests against Hugging Face. Run these after EVERY change to the model code."""
import torch

from miniinfer.engine.generate import naive_generate
from tests.conftest import DEVICE

PROMPT = "The capital of France is Paris. The capital of Japan is"
TOL = 1e-4  # relative max error in fp32; tune if needed, but investigate before loosening


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-8)).item()


def _ids(tokenizer):
    return tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)


@torch.no_grad()
def test_layerwise_hidden_states(hf_model, mini_model, tokenizer):
    ids = _ids(tokenizer)
    ref = hf_model(ids, output_hidden_states=True).hidden_states
    mine = mini_model(ids, return_hidden_states=True).hidden_states

    assert len(mine) == len(ref), f"expected {len(ref)} hidden states, got {len(mine)}"
    errs = [rel_err(m, r) for m, r in zip(mine, ref)]
    bad = next((i for i, e in enumerate(errs) if e > TOL), None)
    assert bad is None, (
        f"first divergence at hidden_states[{bad}] (0 = embeddings, "
        f"{len(ref) - 1} = post-final-norm): rel err {errs[bad]:.2e}\n"
        f"all errors: {[f'{e:.1e}' for e in errs]}"
    )


@torch.no_grad()
def test_logits(hf_model, mini_model, tokenizer):
    ids = _ids(tokenizer)
    ref = hf_model(ids).logits
    mine = mini_model(ids).logits
    assert rel_err(mine, ref) < TOL


@torch.no_grad()
def test_greedy_matches_hf(hf_model, mini_model, tokenizer):
    ids = _ids(tokenizer)
    n = 32
    ref = hf_model.generate(ids, max_new_tokens=n, do_sample=False, min_new_tokens=n)
    mine = naive_generate(mini_model, ids, n)
    assert mine.tolist() == ref.tolist(), (
        f"HF:   {tokenizer.decode(ref[0])!r}\nMine: {tokenizer.decode(mine[0])!r}"
    )
