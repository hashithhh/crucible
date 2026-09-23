"""Phase 2 — transformer interface and contracts. Implementation is Hashith's.

Every method below raises NotImplementedError on purpose. The docstrings state
what must be TRUE of the result (shapes, invariants, behaviour), never how to
compute it: Phase 2 ends with G1, rebuilding this from memory, and a docstring
that spelled out the algorithm would make that gate meaningless.

Sizes follow ADR-0006 (S25). Constants marked PENDING belong in ADR-0007.

Shape names used throughout:
    B  batch            T  time steps (tokens)      V  vocab size
    D  d_model          H  n_heads                  Dh D // H (head dim)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Shape of one model. Values come from ADR-0006 unless marked PENDING."""

    vocab_size: int = 2049  # ADR-0001 (2,048) + <|endoftext|>
    d_model: int = 512  # ADR-0006, S25
    n_layers: int = 8  # ADR-0006, S25
    n_heads: int = 8  # ADR-0006, S25 (head_dim 64)
    context: int = 512  # ADR-0006
    d_ff: int = 2048  # ADR-0006: 4 x d_model
    rope_theta: float = 10_000.0  # PENDING ADR-0007
    norm_eps: float = 1e-6  # PENDING ADR-0007
    init_std: float = 0.02  # PENDING ADR-0007
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
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


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
      separately. That property is the reason the mechanism exists, and
      test_rope_dot_product_depends_only_on_offset pins it.
    - Requires an even head dimension.
    """
    raise NotImplementedError


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
        raise NotImplementedError

    @property
    def length(self) -> int:
        raise NotImplementedError

    def append(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention, causal.

    CONTRACT
    - `forward(x, positions, cache=None, layer=0)` maps (B, T, D) -> (B, T, D).
    - CAUSAL: the output at step t is a function of steps <= t only. Changing
      x[:, t+1:, :] must not change output[:, :t+1, :].
    - `positions` gives each step's absolute position, so a cached decode step
      is positioned correctly.
    - With a cache, the result for the new steps equals what an uncached
      forward over the whole prefix would produce for those steps.
    - Parameters: the four projections (queries, keys, values, output). No
      biases, no dropout, and no positional parameters (RoPE has none).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Block(nn.Module):
    """One transformer block: attention and MLP, each on a residual path.

    CONTRACT
    - (B, T, D) -> (B, T, D), causal (inherited from attention).
    - Pre-norm: each sub-layer normalises its own input, and the residual path
      from block input to block output carries the unmodified signal. A block
      whose sub-layer output projections are zeroed is therefore exactly the
      identity function, which test_block_with_zeroed_outputs_is_identity
      pins.
    - Holds two RMSNorms, one attention, one MLP.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        raise NotImplementedError


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
      those steps, within tolerance. That equivalence is the KV cache's whole
      correctness condition (test_kv_cache_matches_full_forward).
    - Deterministic: same input and parameters, same output. No dropout.
    - `n_params(non_embedding=False)` returns the parameter count, for
      checking against ADR-0006.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        raise NotImplementedError

    def forward(self, ids: Tensor, cache: KVCache | None = None) -> Tensor:
        raise NotImplementedError

    def n_params(self, non_embedding: bool = False) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def new_cache(self, batch_size: int) -> KVCache:
        """A cache sized for this model and `batch_size`, capacity = context."""
        raise NotImplementedError
