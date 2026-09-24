"""Phase 2.4 — greedy, temperature, top-k, top-p, and cached generation.

Written by Claude (see CLAUDE.md and the git log), against the contracts and
tests committed in 3f465b1.

Convention: each function takes the logits for ONE step, shape (B, V), and
returns either modified logits of the same shape or chosen ids, (B,).
Filtering functions mark rejected entries with -inf so they compose: the
result is still logits, and softmax of it is the distribution drawn from.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crucible.model import Transformer

REJECTED = float("-inf")


def greedy(logits: Tensor) -> Tensor:
    """Pick the highest-scoring id per row.

    CONTRACT
    - (B, V) -> (B,) int64 on the same device.
    - Deterministic, and unchanged by any strictly increasing rescaling of a
      row's logits.
    - Ties resolve to the lowest id, so the result never depends on iteration
      order.
    """
    best = logits.amax(dim=-1, keepdim=True)
    ids = torch.arange(logits.shape[-1], device=logits.device)
    # Where a row ties, take the smallest id rather than trusting argmax.
    return torch.where(logits == best, ids, logits.shape[-1]).amin(dim=-1)


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
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return logits
    return logits / temperature


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
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k >= logits.shape[-1]:
        return logits
    keep = logits.topk(k, dim=-1).indices
    out = torch.full_like(logits, REJECTED)
    return out.scatter(-1, keep, logits.gather(-1, keep))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filter: keep the smallest top set whose mass reaches p.

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
    if p <= 0 or p > 1:
        raise ValueError(f"p must lie in (0, 1], got {p}")
    ordered, order = logits.sort(dim=-1, descending=True)
    probs = ordered.softmax(dim=-1)
    # Mass strictly before this entry; below p means the entry is still needed.
    preceding = probs.cumsum(dim=-1) - probs
    keep_ordered = preceding < p
    keep = torch.zeros_like(keep_ordered).scatter(-1, order, keep_ordered)
    return logits.masked_fill(~keep, REJECTED)


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
    scores = apply_temperature(logits, temperature)
    if top_k is not None:
        scores = top_k_filter(scores, top_k)
    if top_p is not None:
        scores = top_p_filter(scores, top_p)
    probs = scores.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


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
      than the whole prefix again. The result equals what the same sampling
      decisions produce without a cache.
    - Stops early only once every row has produced `eos_id`, when one is given.
    - Never exceeds the model's context; asking for more raises ValueError.
    - Leaves the model in eval mode and allocates no gradients.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    total = prompt.shape[1] + max_new_tokens
    if total > model.config.context:
        raise ValueError(
            f"{prompt.shape[1]} prompt + {max_new_tokens} new steps exceeds "
            f"context {model.config.context}"
        )

    model.eval()
    cache = model.new_cache(prompt.shape[0])
    out = prompt
    logits = model(prompt, cache)[:, -1]
    finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)

    for _ in range(max_new_tokens):
        nxt = sample_next(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        out = torch.cat([out, nxt.unsqueeze(-1)], dim=-1)
        if eos_id is not None:
            finished |= nxt == eos_id
            if bool(finished.all()):
                break
        logits = model(nxt.unsqueeze(-1), cache)[:, -1]
    return out
