"""Phase 2 — decoder-only transformer: RMSNorm, RoPE, attention, MLP, KV cache.

Written by Claude (see CLAUDE.md and the git log), against the contracts and
tests committed in 3f465b1. Constants are recorded in ADR-0007.

Shape names used throughout:
    B  batch            T  time steps (tokens)      V  vocab size
    D  d_model          H  n_heads                  Dh D // H (head dim)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Shape of one model. Values from ADR-0006; constants from ADR-0007."""

    vocab_size: int = 2049  # ADR-0001 (2,048) + <|endoftext|>
    d_model: int = 512  # ADR-0006, S25
    n_layers: int = 8  # ADR-0006, S25
    n_heads: int = 8  # ADR-0006, S25 (head_dim 64)
    context: int = 512  # ADR-0006
    d_ff: int = 2048  # ADR-0006: 4 x d_model
    rope_theta: float = 10_000.0  # ADR-0007
    norm_eps: float = 1e-6  # ADR-0007
    init_std: float = 0.02  # ADR-0007
    tie_embeddings: bool = True  # ADR-0006 counts parameters with tying

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
            )


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    CONTRACT
    - `forward(x)` returns a tensor with x's shape, dtype and device.
    - Normalises over the last dimension only; positions and batch entries do
      not interact.
    - Scale-invariant: for c > 0, forward(c * x) == forward(x) within floating
      point tolerance. (That is what distinguishes it from LayerNorm, which
      also removes the mean.)
    - Holds exactly one learnable parameter vector of size `dim`, initialised
      so the module starts as an identity-preserving scale.
    - Finite for an all-zero input; that is what `eps` is for.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def apply_rope(x: Tensor, positions: Tensor, *, theta: float) -> Tensor:
    """Apply rotary position embedding to a (B, H, T, Dh) tensor.

    CONTRACT
    - Returns the same shape, dtype and device as `x`.
    - `positions` is an integer tensor broadcastable to (B, T): the absolute
      position of each step. Steps are independent, and position p always
      produces the same transform for the same vector.
    - Norm-preserving: ||out[..., t, :]|| == ||x[..., t, :]|| within tolerance.
    - Position 0 is the identity transform.
    - RELATIVE: for a query q at position m and a key k at position n, the dot
      product <rope(q, m), rope(k, n)> depends only on (m - n), not on m and n
      separately.
    - Requires an even head dimension.
    """
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError(f"head dimension must be even, got {head_dim}")

    half = head_dim // 2
    inv_freq = theta ** (
        -torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    steps = positions.shape[-1]
    angles = positions.to(torch.float32).reshape(-1, steps, 1) * inv_freq
    angles = angles.unsqueeze(1)  # (B, 1, T, half), broadcast over heads
    cos, sin = angles.cos(), angles.sin()

    left, right = x.float().split(half, dim=-1)
    rotated = torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)
    return rotated.type_as(x)


class KVCache:
    """Per-layer key/value store for incremental decoding.

    CONTRACT
    - Holds keys and values for at most `capacity` steps per layer.
    - `append(layer, k, v)` adds the steps in k/v (shape (B, H, T_new, Dh))
      and returns everything stored for that layer so far, oldest first, as a
      (keys, values) pair of (B, H, T_total, Dh) tensors.
    - Appending past `capacity` raises ValueError rather than silently
      dropping or wrapping.
    - `length` is the number of steps currently stored, the same for every
      layer once a full forward has run.
    - `reset()` empties it; a reset cache behaves exactly like a new one.
    - Storage matches the dtype and device of the tensors appended.
    """

    def __init__(self, n_layers: int, capacity: int) -> None:
        self.n_layers = n_layers
        self.capacity = capacity
        self._keys: list[Tensor | None] = [None] * n_layers
        self._values: list[Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        first = self._keys[0]
        return 0 if first is None else first.shape[-2]

    def append(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        stored_k, stored_v = self._keys[layer], self._values[layer]
        held = 0 if stored_k is None else stored_k.shape[-2]
        if held + k.shape[-2] > self.capacity:
            raise ValueError(
                f"cache capacity {self.capacity} exceeded: {held} stored, "
                f"{k.shape[-2]} more"
            )
        keys = k if stored_k is None else torch.cat([stored_k, k], dim=-2)
        values = v if stored_v is None else torch.cat([stored_v, v], dim=-2)
        self._keys[layer], self._values[layer] = keys, values
        return keys, values

    def reset(self) -> None:
        self._keys = [None] * self.n_layers
        self._values = [None] * self.n_layers


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention, causal.

    CONTRACT
    - `forward(x, positions, cache=None, layer=0)` maps (B, T, D) -> (B, T, D).
    - CAUSAL: the output at step t is a function of steps <= t only.
    - `positions` gives each step's absolute position, so a cached decode step
      is positioned correctly.
    - With a cache, the result for the new steps equals what an uncached
      forward over the whole prefix would produce for those steps.
    - Parameters: the four projections (queries, keys, values, output). No
      biases, no dropout, and no positional parameters (RoPE has none).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        batch, steps, _ = x.shape
        heads, head_dim = self.config.n_heads, self.config.head_dim

        def split(projected: Tensor) -> Tensor:
            return projected.view(batch, steps, heads, head_dim).transpose(1, 2)

        theta = self.config.rope_theta
        q = apply_rope(split(self.q_proj(x)), positions, theta=theta)
        k = apply_rope(split(self.k_proj(x)), positions, theta=theta)
        v = split(self.v_proj(x))

        if cache is not None:
            k, v = cache.append(layer, k, v)

        # Query i sits at absolute position (total - steps + i) and may attend
        # to every key at or before it.
        total = k.shape[-2]
        query_pos = torch.arange(total - steps, total, device=x.device).unsqueeze(1)
        key_pos = torch.arange(total, device=x.device).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_pos <= query_pos
        )

        merged = attended.transpose(1, 2).reshape(batch, steps, self.config.d_model)
        return self.o_proj(merged)


class MLP(nn.Module):
    """Position-wise feed-forward network.

    CONTRACT
    - (B, T, D) -> (B, T, D). Every position is transformed independently, so
      permuting positions permutes outputs identically.
    - Widens to `config.d_ff` in between.
    - Two weight matrices, no biases.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    """One transformer block: attention and MLP, each on a residual path.

    CONTRACT
    - (B, T, D) -> (B, T, D), causal (inherited from attention).
    - Pre-norm: each sub-layer normalises its own input, and the residual path
      from block input to block output carries the unmodified signal. A block
      whose sub-layer output projections are zeroed is therefore exactly the
      identity function.
    - Holds two RMSNorms, one attention, one MLP.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), positions, cache, layer)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    """Decoder-only transformer over token ids.

    CONTRACT
    - `forward(ids, cache=None)` maps (B, T) int64 ids -> (B, T, V) float
      logits.
    - CAUSAL end to end: logits[:, t] depend on ids[:, :t+1] only.
    - T may be 1 up to `config.context`; longer raises ValueError rather than
      silently truncating.
    - Ids outside range(vocab_size) raise rather than indexing garbage.
    - With a cache, `forward(next_ids, cache)` continues the sequence: the
      logits equal those an uncached forward over the whole sequence gives for
      those steps, within tolerance.
    - Deterministic: same input and parameters, same output. No dropout.
    - `n_params(non_embedding=False)` returns the parameter count, for
      checking against ADR-0006.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, ids: Tensor, cache: KVCache | None = None) -> Tensor:
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) ids, got shape {tuple(ids.shape)}")
        steps = ids.shape[1]
        if steps == 0:
            raise ValueError("ids must contain at least one step")
        start = cache.length if cache is not None else 0
        if start + steps > self.config.context:
            raise ValueError(
                f"sequence of {start + steps} steps exceeds context "
                f"{self.config.context}"
            )
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= self.config.vocab_size:
            raise ValueError(
                f"token ids must lie in [0, {self.config.vocab_size}); "
                f"got [{low}, {high}]"
            )

        positions = torch.arange(start, start + steps, device=ids.device)
        positions = positions.unsqueeze(0).expand(ids.shape[0], steps)

        x = self.embedding(ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, positions, cache, layer)
        return self.lm_head(self.final_norm(x))

    def n_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embedding.weight.numel()
        return total

    @torch.no_grad()
    def new_cache(self, batch_size: int) -> KVCache:
        """A cache sized for this model and `batch_size`, capacity = context."""
        return KVCache(self.config.n_layers, self.config.context)
