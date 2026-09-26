# ADR-0008 — Training hyperparameters

- **Status:** accepted
- **Date:** 2026-09-26
- **Applies to:** `crucible/training.py`, `scripts/train.py`,
  `scripts/encode_corpus.py` (Phase 3). Model shape comes from ADR-0006 and
  model constants from ADR-0007; this ADR covers the constants the *run*
  needs.
- **Corpus:** the measured one — 432,175,881 tokens, 429,998,810 of them in
  the train split (ADR-0006 amendment 2026-09-26, `data/tokens/meta.json`).

## Context

Phase 3 trains S25 (26,223,616 parameters) for one epoch of TinyStories on a
6 GB RTX 4050 locally and a 16 GB T4 on Kaggle. CLAUDE.md requires an ADR for
every constant, and a training loop is mostly constants. Most of these are
borrowed; this ADR's job is to say **from where**, and to be honest about
which were reasoned from this setup and which are conventions carried over
because nothing measured here contradicts them.

**Borrowed means borrowed.** Where a value comes from GPT-3, Chinchilla or
nanoGPT, it is cited and not re-derived. Where this setup differs from the
source in a way that matters, the difference is stated rather than ignored.

## Decisions

### Data shape

| Constant | Value | Where it comes from |
|---|---:|---|
| `context` | 512 | ADR-0006 |
| `batch_tokens` | 65,536 | reasoned below |
| `batch_size` | 128 sequences | `batch_tokens / context` |
| `micro_batch` | 16 (local), 64 (T4) | measured / projected, below |
| `steps` | 6,561 | `429,998,810 / 65,536`, one epoch |
| `TOKENS_PER_SHARD` | 67,108,864 | reasoned below |
| `VAL_FRACTION` | 0.005 | reasoned below |

**`batch_tokens` = 65,536.** GPT-3 Small (125M) used 0.5M tokens per step,
and nanoGPT's 124M OpenWebText config matches it. Copying 0.5M here would
give 860 optimizer steps for the whole epoch, too few for a cosine schedule
to be a schedule at all. 65,536 gives **6,561 steps** — enough for warmup,
decay and a readable loss curve — and the gradient-noise argument for large
batches is at its weakest exactly here, at 26M parameters on a single narrow
domain. It is a power of two, so the accumulation count divides exactly.

**`micro_batch`.** ADR-0006's amendment measured S25 on the local card:
micro-batch 16 peaks at 2.19 GB allocated and runs at 41,500 tokens/s, the
fastest of the three sizes tried. 128 / 16 = **8 gradient-accumulation
steps**. The T4 has 16 GB, so 64 should fit with the same arithmetic at 2
accumulation steps; that number is projected, not measured, and stays so
until the MFU run Phase 3 owes.

**`TOKENS_PER_SHARD` = 64M (128 MB).** Large enough that per-file overhead is
irrelevant at ~432M tokens (7 shards), small enough that one shard memmaps
comfortably on a machine with 2-3 GB free.

**`VAL_FRACTION` = 0.005.** 0.5% of the corpus is 2,177,071 held-out tokens
across 9,028 stories — enough that held-out loss is stable between
checkpoints rather than noise, and small enough that training gives up
nothing that matters. The split is by story digest, not by position, so it is
deterministic and independent of corpus order.

### Optimizer

| Constant | Value | Where it comes from |
|---|---:|---|
| `peak_lr` | 6e-4 | GPT-3 Small; argued below |
| `min_lr` | 6e-5 | `peak_lr / 10`, Chinchilla convention |
| `warmup_steps` | 200 | reasoned below |
| `betas` | (0.9, 0.95) | GPT-3 / Chinchilla, not Adam's default |
| `eps` | 1e-8 | PyTorch default; nothing here argues against it |
| `weight_decay` | 0.1 | GPT-3 / Chinchilla |
| `grad_clip` | 1.0 | GPT-3 / Chinchilla / nanoGPT |

**`peak_lr` = 6e-4, and why that is not simply a copy.** 6e-4 is GPT-3
Small's value at 125M parameters with a 0.5M-token batch. Two differences
pull in opposite directions: S25 is **5x smaller**, and smaller models
tolerate larger learning rates; but the batch is **8x smaller**, and smaller
batches want smaller ones. Square-root scaling on the batch alone would
suggest ~2e-4. Taking 6e-4 is the judgement that the model-size effect
roughly offsets the batch effect, and it is the less conservative of the two
readings.

**This is the value most likely to be wrong, so it has a stated trigger.** If
the loss diverges, or the pre-clip gradient norm sits above `grad_clip` for
most of the first 500 steps, the remedy is to halve `peak_lr` and restart,
recording that as an amendment here. Logging grad norm (Phase 3.5) exists
partly to make this observable rather than guessed at.

**`weight_decay` = 0.1, applied to tensors of rank 2 or higher only.** Weight
decay on a bias or a normalisation gain shrinks a shift, not a direction,
which is not what the regulariser is for. S25 has no biases, so in practice
this exempts the RMSNorm gain vectors. The embedding matrix **is** decayed:
it is rank 2, it is tied to the output head under ADR-0006, and as the output
head it is doing the work of a weight matrix.

**`grad_clip` = 1.0** on the global norm, applied after unscaling and before
the optimizer step. Universal at this scale; the value is not tuned here.

### Schedule

Linear warmup from 0 to `peak_lr` over `warmup_steps`, then cosine decay to
`min_lr` at the final step, then flat at `min_lr` if the run is ever extended
past it. The cosine is the Chinchilla shape, and Chinchilla's own finding is
that the decay should end **at** the end of the run: a cosine cut short, or
stretched past its horizon, is measurably worse than one that lands. That is
why `steps` is fixed from the corpus up front rather than left open.

Holding at `min_lr` past the horizon is a safety property, not a plan. It
means an accidental over-run degrades gently instead of turning the cosine
back upward.

### Precision

| | Local RTX 4050 | Kaggle T4 |
|---|---|---|
| Autocast dtype | bf16 | fp16 |
| Loss scaler | none | **required** |

Turing has no bfloat16, so the T4 must use fp16, and fp16 needs a loss
scaler. **Why the scaler exists** (Phase 3.3 asks for this in one paragraph,
and if it cannot be written down it was not understood): fp16 has 5 exponent
bits, so values below roughly 6e-8 flush to zero. Activations sit comfortably
inside that range but *gradients* do not — small gradients underflow to zero
during the backward pass, and the update they should have produced is
silently lost. The scaler multiplies the loss by a large factor before
`backward()`, which by the chain rule multiplies every gradient by that same
factor and lifts them back into representable range; the optimizer divides
them out again before stepping, so the arithmetic is unchanged. If the factor
is too large, gradients overflow to inf instead, so the scaler watches for
non-finite gradients, **skips that step entirely** and halves the factor —
which is why an fp16 run legitimately takes slightly fewer optimizer steps
than iterations, and why a skipped step is not a bug to chase. bf16 has
fp32's 8 exponent bits and therefore its dynamic range, so none of this
applies to it; that is the whole reason bf16 is preferred where the hardware
has it.

Master weights, the optimizer state and the loss reduction stay fp32 under
both. Only the forward pass and the backward through it are reduced.

### Run control

| Constant | Value | Why |
|---|---:|---|
| `seed` | 1337 | arbitrary, and fixed so it can be quoted |
| `eval_every` | 250 steps | ~26 held-out evaluations across the run |
| `checkpoint_every` | 500 steps | reasoned below |

**The seed is genuinely arbitrary.** It is recorded not because 1337 is a
good number but because an unrecorded seed makes a run unreproducible, and
this project's rule is that constants are written down.

**`checkpoint_every` = 500 steps** (~14 minutes locally). Kaggle sessions run
12 hours and can end without warning, so the real question is how much work a
crash may destroy. 14 minutes is a cheap premium against a 3-hour run, and a
checkpoint of a 26M-parameter model plus its Adam state is ~300 MB, which
Kaggle's 20 GB working directory absorbs even if several are kept.

## Consequences

- **One epoch is 6,561 steps and ~3.0 hours locally** at the measured 41.5k
  tokens/s, down from the ~3.5 hours ADR-0006 projected, because the corpus
  turned out 15.3% smaller after dedup.
- **`peak_lr` carries the most risk**, and is the only constant here with a
  pre-stated revision trigger. Any change to it is an amendment with the
  grad-norm evidence that prompted it.
- **The T4 column is unverified.** `micro_batch` 64 and the fp16 path are
  reasoned, not measured. The MFU measurement ADR-0006 says Phase 3 owes is
  where they get checked; until then only the local column is evidence.
- **A second epoch is not scheduled.** ADR-0006's amendment records it as the
  lever if loss is still falling steeply at the horizon. Taking it means
  extending the cosine and re-registering the run, not quietly continuing
  past `min_lr`.

## Sources

- Brown et al., *Language Models are Few-Shot Learners* (GPT-3), Table 2.1
  and Appendix B: learning rate by model size, betas, weight decay, gradient
  clipping, cosine decay to 10% after linear warmup.
- Hoffmann et al., *Training Compute-Optimal Language Models* (Chinchilla):
  the ~20 tokens/param ratio, and the finding that the cosine schedule should
  end at the end of training.
- Karpathy, `nanoGPT`: the accumulation structure, the rank-2 weight-decay
  split, and the 124M configuration used above as the comparison point.
- `results/s25_smoke.json`: the local micro-batch and throughput measurements.
- `data/tokens/meta.json`, `results/shard_verification.json`: the corpus.
