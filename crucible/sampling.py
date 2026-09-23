"""Phase 2.4 — sampling interface and contracts. Implementation is Hashith's.

Contracts state what must hold of the output, not how to produce it (see
crucible/model.py for why).

Convention: each function takes the logits for ONE step, shape (B, V), and
returns either modified logits of the same shape or chosen ids, (B,).
Filtering functions mark rejected entries with -inf so they compose: the
result is still logits, and softmax of it is the distribution drawn from.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crucible.model import Transformer


def greedy(logits: Tensor) -> Tensor:
    """Pick the highest-scoring id per row.

    CONTRACT
    - (B, V) -> (B,) int64 on the same device.
    - Deterministic, and unchanged by any strictly increasing rescaling of a
      row's logits.
    - Ties resolve to the lowest id, so the result never depends on iteration
      order.
    """
    raise NotImplementedError


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by a temperature.

    CONTRACT
    - (B, V) -> (B, V), same dtype and device.
    - temperature == 1.0 leaves the distribution unchanged.
    - Lower temperature concentrates probability mass on the argmax; higher
      flattens it. As temperature -> 0 the sampled id tends to greedy's.
    - temperature <= 0 raises ValueError. Zero is not "greedy" here: it would
      divide by zero, and callers wanting greedy call greedy().
    - -inf entries stay -inf.
    """
    raise NotImplementedError


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the k highest-scoring entries per row and reject the rest.

    CONTRACT
    - (B, V) -> (B, V), same dtype and device.
    - Exactly k entries per row stay finite (fewer only if the row had fewer
      finite entries to begin with); every other entry is -inf.
    - Kept entries keep their original values, so top-k then temperature and
      temperature then top-k agree on which ids survive.
    - k <= 0 raises ValueError; k >= V is a no-op.
    - Ties at the boundary resolve deterministically.
    """
    raise NotImplementedError


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filter: keep the smallest top set whose probability mass reaches p.

    CONTRACT
    - (B, V) -> (B, V), same dtype and device.
    - The kept set per row is the smallest prefix of the descending-probability
      order whose cumulative probability is >= p, so it always keeps at least
      the argmax, even when one entry alone already exceeds p.
    - Kept entries keep their original values; the rest become -inf.
    - p <= 0 or p > 1 raises ValueError; p == 1.0 keeps every finite entry.
    - Probabilities come from the logits as given, so filtering after a
      temperature change may keep a different number of entries. That is
      expected, and why sample_next fixes the order.
    """
    raise NotImplementedError


def sample_next(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Choose the next id per row.

    CONTRACT
    - (B, V) -> (B,) int64 on the logits' device.
    - Order: temperature, then top-k, then top-p, then draw.
    - Every returned id is one the filters kept: an id whose filtered logit is
      -inf is never returned.
    - With `generator`, the draw is reproducible: same generator state, same
      logits, same ids.
    - Validation is delegated to the functions above, so the same bad inputs
      raise the same errors.
    """
    raise NotImplementedError


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> Tensor:
    """Continue `prompt` by up to `max_new_tokens` ids, using a KV cache.

    CONTRACT
    - prompt is (B, T_prompt) int64; returns (B, T_prompt + n) with the prompt
      unchanged at the front and n <= max_new_tokens.
    - Uses the model's KV cache: each new step processes one position rather
      than the whole prefix again. The result must equal what the same
      sampling decisions produce without a cache
      (test_generate_matches_uncached).
    - Stops early only once every row has produced `eos_id`, when one is given.
    - Never exceeds the model's context; asking for more raises ValueError.
    - Leaves the model in eval mode and allocates no gradients.
    """
    raise NotImplementedError
