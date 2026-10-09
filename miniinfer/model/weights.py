from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .qwen3 import Qwen3ForCausalLM


def load_weights(model: Qwen3ForCausalLM, repo_id: str) -> Qwen3ForCausalLM:
    path = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json"]))
    state = {}
    for f in sorted(path.glob("*.safetensors")):
        state.update(load_file(str(f)))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if model.cfg.tie_word_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]  # shared with embeddings
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"
    return model
