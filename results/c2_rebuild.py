"""AI-assisted early reconstruction of the Phase 2 transformer.

This file is deliberately separate from ``crucible.model`` so the original
Phase 2 implementation remains untouched.  It is not a no-AI G1 attempt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 2_049
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    context: int = 512
    d_ff: int = 2_048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight.to(dtype=x.dtype)


def apply_rope(x: Tensor, positions: Tensor, theta: float = 10_000.0) -> Tensor:
    """Apply split-half rotary embeddings to ``(batch, heads, steps, dim)``."""
    half = x.shape[-1] // 2
    if x.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    frequencies = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cos = angles.cos().unsqueeze(1).to(dtype=x.dtype)
    sin = angles.sin().unsqueeze(1).to(dtype=x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class LayerCache:
    def __init__(self) -> None:
        self.key: Tensor | None = None
        self.value: Tensor | None = None

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        self.key = key if self.key is None else torch.cat((self.key, key), dim=2)
        self.value = value if self.value is None else torch.cat((self.value, value), dim=2)
        return self.key, self.value

    def reset(self) -> None:
        self.key = None
        self.value = None


class KVCache:
    def __init__(self, config: ModelConfig, batch_size: int) -> None:
        self.config = config
        self.batch_size = batch_size
        self.layers = [LayerCache() for _ in range(config.n_layers)]
        self.length = 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.length = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, x: Tensor, positions: Tensor, cache: LayerCache | None = None
    ) -> Tensor:
        batch, steps, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, steps, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        q, k, v = (apply_rope(heads(t), positions, self.config.rope_theta) for t in (q, k, v))
        if cache is None:
            keys, values = k, v
            key_positions = positions
        else:
            keys, values = cache.append(k, v)
            key_positions = torch.arange(keys.shape[2], device=x.device).expand(batch, -1)
        mask = key_positions[:, None, :] <= positions[:, :, None]
        attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None], dropout_p=0.0)
        return self.o_proj(attended.transpose(1, 2).contiguous().view(batch, steps, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, positions: Tensor, cache: LayerCache | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), positions, cache)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def n_params(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        return total - (self.token_embedding.weight.numel() if non_embedding else 0)

    def new_cache(self, batch_size: int) -> KVCache:
        return KVCache(self.config, batch_size)

    def forward(self, ids: Tensor, cache: KVCache | None = None) -> Tensor:
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, steps)")
        batch, steps = ids.shape
        start = 0 if cache is None else cache.length
        if steps + start > self.config.context:
            raise ValueError("sequence exceeds context length")
        if cache is not None and batch != cache.batch_size:
            raise ValueError("cache batch size does not match ids")
        positions = torch.arange(start, start + steps, device=ids.device).expand(batch, -1)
        x = self.token_embedding(ids)
        for block, layer_cache in zip(self.blocks, cache.layers if cache else [None] * len(self.blocks), strict=True):
            x = block(x, positions, layer_cache)
        logits = self.lm_head(self.norm(x))
        if cache is not None:
            cache.length += steps
        return logits
