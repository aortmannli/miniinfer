"""Qwen3 forward pass, written from scratch.

Parameter names deliberately match Hugging Face's, so the checkpoint loads with
strict key matching. Only the forward() bodies marked TODO are left to write.

Things worth checking in the HF config rather than assuming:
  * head_dim is set explicitly and is NOT hidden_size // num_attention_heads
  * num_key_value_heads < num_attention_heads (grouped-query attention)
  * q_norm / k_norm: RMSNorm over head_dim applied to q and k BEFORE RoPE
  * tie_word_embeddings: lm_head shares the embedding matrix
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int

    @classmethod
    def from_hf(cls, hf_config) -> "Qwen3Config":
        vals = {}
        for f in fields(cls):
            if f.name == "rope_theta":
                # newer transformers versions nest this under rope_parameters
                theta = getattr(hf_config, "rope_theta", None)
                if theta is None:
                    theta = hf_config.rope_parameters["rope_theta"]
                vals[f.name] = theta
            else:
                vals[f.name] = getattr(hf_config, f.name)
        return cls(**vals)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    # Same convention as HF: [embeddings, after layer 0, ..., after last layer],
    # where the LAST entry has the final norm applied.
    hidden_states: list[torch.Tensor] | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean square 
        mean_square = (x*x).mean(dim=-1, keepdim=True)
        mean_square += self.eps
        inv_rms = mean_square.rsqrt()
        result = (x*inv_rms)*self.weight
        return result


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim); positions: (T,)."""
    # TODO: rotary embedding. Check which pairing convention HF uses for
    # Qwen3 (rotate_half: first half vs second half of head_dim, not
    # interleaved pairs). Getting this wrong gives plausible-looking garbage.
    raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: SwiGLU
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        q_out = cfg.num_attention_heads * cfg.head_dim
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # TODO: project -> reshape to heads -> q_norm/k_norm -> RoPE ->
        #       repeat KV heads for GQA -> causal softmax attention -> o_proj
        # Write the attention math by hand first (matmul + mask + softmax);
        # swap in F.scaled_dot_product_attention later and re-run parity.
        raise NotImplementedError


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # TODO: pre-norm residual blocks (attention, then MLP)
        raise NotImplementedError


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, return_hidden_states: bool = False) -> ModelOutput:
        _, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        n = len(self.model.layers)

        h = self.model.embed_tokens(input_ids)
        hs = [h]
        for i, layer in enumerate(self.model.layers):
            h = layer(h, positions)
            if i < n - 1:
                hs.append(h)
        h = self.model.norm(h)
        hs.append(h)  # last entry is post-final-norm, matching HF

        logits = self.lm_head(h)
        return ModelOutput(logits, hs if return_hidden_states else None)
