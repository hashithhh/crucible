# ADR-0006 — Model size and training data

- **Status:** accepted
- **Date:** 2026-09-22
- **Numbers:** `python scripts/model_budget.py` → `results/model_budget.json`.
  Every figure below comes from that script, whose inputs are the measured
  constants listed under Sources.

## Amendment, 2026-09-25 — phase numbering, and first measured throughput

Two corrections to what this ADR assumed, with the original text below left
as written.

**Phase numbering.** This ADR and ADR-0001 both say "Phase 2 trains the
tokenizer on TinyStories-train". The project's phase list puts the
*transformer* at Phase 2 (built 2026-09-24) and training at Phase 3, so the
list wins: **retraining the tokenizer on the full 1.9 GB corpus, and
re-measuring bytes/token on it, belongs to Phase 3 data preparation**, before
any training run. Nothing about the decisions changes; only where the task
sits. It is still not done.

**First measured throughput** (`results/s25_smoke.json`), S25 on the local
RTX 4050 Laptop, bf16 autocast, AdamW, context 512:

| Micro-batch | Tokens/s | Peak allocated |
|---:|---:|---:|
| 8 | 38,900 | 1.32 GB |
| 16 | 41,500 | 2.19 GB |
| 32 | 40,100 | 3.92 GB |

The memory estimate above said 2.2 GB at micro-batch 16 including 0.5 GB of
CUDA overhead; measured allocation alone is 2.19 GB, so the model fits the
6 GB card with room to spare but the estimate was optimistic by roughly that
overhead. 40k tokens/s is ~7.3 TFLOP/s against this card, i.e. a low-to-mid
MFU, which suggests the 25-40% band assumed for the T4 is optimistic for a
512-wide model. One epoch of 510M tokens takes ~3.5 hours locally. The T4
measurement Phase 3 owes is still owed.

## Context
The project's working target has been a ~100M-parameter model. ADR-0001 fixed
the tokenizer at 2,048 tokens, which turns TinyStories-train (1,924,281,556
bytes) into **510.5M unique tokens**. At Chinchilla's ~20 training tokens per
parameter, that supports **~25M parameters**, not 100M: a 100M model needs
~2.0B tokens.

Two machines are in play, and they bound different things:

- **Local:** RTX 4050 Laptop, **6,141 MiB** (nvidia-smi). Used for development
  and short runs.
- **Kaggle:** NVIDIA T4, **16 GB, 65 TFLOPS mixed-precision peak** (NVIDIA
  datasheet), offered as 1 or 2 T4s. **~30 GPU-hours per week** (T4 x2 draws
  on the same quota), **12-hour sessions**, **20 GB** of persistent
  `/kaggle/working`. Turing has no bf16, so training is FP16 with loss scaling.

## Assumptions stated up front
- **Throughput is not measured yet.** Hours below use T4 peak x an assumed
  **25–40% MFU**, and **1.8x** for T4 x2 data parallelism. Small models
  (d_model 512) may land below that band on a T4. Phase 3's first job is to
  measure tokens/s and re-run `model_budget.py` with the real MFU.
- **Architecture for counting:** pre-LN GPT, no biases, RoPE (no positional
  parameters), tied input/output embedding, MLP width 4 x d_model, context 512,
  vocab 2,049 (2,048 + `<|endoftext|>`), head_dim 64.
- **Memory:** AdamW with FP32 master weights = 16 bytes/param; activations per
  Korthikanti et al. 2022 with fused attention (34 bytes x seq x batch x d per
  layer; top-2 MoE adds the MLP share, 19, once more); 0.5 GB CUDA overhead
  assumed.
- **FLOPs/token:** 6 x active params + 12 x layers x context x d_model
  (PaLM, Chowdhery et al. 2022).

## 1. Phase 3 target: which option

| | (a) +FineWeb-Edu, 100M | **(b) 25M, TinyStories** | (c) 100M, TinyStories, 1 epoch | (c) 100M, 4 epochs |
|---|---:|---:|---:|---:|
| Config | M100 (100.7M) | **S25 (26.2M)** | M100 | M100 |
| Unique tokens | 2.00B | **0.51B** | 0.51B | 0.51B |
| Tokens seen / param | 19.9 | **19.5** | 5.1 | 20.3 |
| Download | 6.23 GB | **1.92 GB** | 1.92 GB | 1.92 GB |
| Disk, peak / steady | 10.2 / 4.0 GB | **2.95 / 1.02 GB** | 2.95 / 1.02 GB | 2.95 / 1.02 GB |
| Train, 1x T4 | 14–23 h | **1.0–1.6 h** | 3.7–5.8 h | 15–23 h |
| Train, T4 x2 | 8–13 h | **0.6–0.9 h** | 2.0–3.2 h | 8–13 h |
| Share of weekly quota (1x T4) | 48–76% | **3–5%** | 12–19% | 49–78% |

Disk counts token ids as uint16. "Peak" is download plus tokenized data before
the raw download is deleted.

### Option (a) in detail — FineWeb-Edu sizes, verified
From the dataset card and HF datasets-server at revision `87f0914`:
- `sample-10BT` is "around 10B gpt2 tokens": **9,672,101 documents in 14
  parquet shards, 28.52 GB** (13 shards of 2.15 GB, one of 0.54 GB).
- 1,000 random documents: **5,030 bytes and 4.577 bytes per GPT-2 token** on
  average (the dataset's own `token_count`), i.e. ~48.6 GB of text in the
  sample.
- Our 2,048-token vocabulary compresses FineWeb-Edu far less well than
  TinyStories: **2.51** bytes/token when trained on TinyStories, **2.89** when
  retrained on a 50/50 mix, against 3.73 on TinyStories itself.

Reaching 2.0B tokens with the mixed tokenizer takes 537.6M TinyStories tokens
plus **1.46B FineWeb-Edu tokens (73% of the data)**: 4.23 GB of text, 2.48 GB
of parquet, so **2 shards** (4.31 GB) to download.

### Decision: **(b) — lower the Phase 3 target to ~25M (S25, 26.2M params).**

1. **It matches the data.** 510.5M unique tokens for 26.2M params is 19.5
   tokens/param, Chinchilla-optimal in one epoch, with no new data and no
   repetition.
2. **It keeps the project in one domain, and keeps ADR-0001 valid.** Under (a),
   73% of training tokens would be web text. The tokenizer would have to be
   retrained on the mix (web text at 2,048 vocab: 2.89 bytes/token), ADR-0001's
   sweep redone on that mix, and the model would stop being a TinyStories
   model. That is a different project, not a bigger version of this one.
3. **The domain supports the size.** The TinyStories paper reports fluent,
   coherent stories from models "below 10 million total parameters" (Eldan &
   Li, 2023). 25M is comfortably inside the regime the dataset was built for.
4. **It leaves compute for Phases 4 and 5.** One S25 run is 3–5% of a week's
   Kaggle quota and fits in one session. Under (a) or (c)-4-epochs a single run
   takes half to three quarters of a week and exceeds the 12-hour session on
   one T4, forcing checkpoint/resume before any science happens.
5. **It fits the local GPU,** including its MoE variant (§2), so development
   does not depend on Kaggle.

**Rejected:**
- **(a)** buys the 100M headline at the cost of the domain, the tokenizer
  decision, and ~15x the compute per run.
- **(c), 1 epoch** trains a 100M model at 5 tokens/param: undertrained by
  design. **(c), 4 epochs** reaches ~20 tokens/param through repetition
  (Muennighoff et al. 2023 find ~4 epochs close to unique data), at the same
  cost as (a) per run. Data-constrained scaling is a legitimate Phase 5
  question, but at 100M each point costs 15–23 T4-hours. Phase 5 can study it
  far more cheaply at small scale (§3).

**What this gives up:** the "~100M-parameter LLM" framing in CLAUDE.md and the
README. It becomes ~25M, with 100M recorded here as the size the data does not
support.

## 2. Phase 4: ablation scale

**All Phase 4 ablations run at the S25 base (26.2M).** Arithmetic from the
configs in `scripts/model_budget.py`, 8 experts, top-2 routing, each expert a
full-width MLP:

| Config | d / L / H | Total params | Active | AdamW state | Est. peak, micro-batch 16 |
|---|---|---:|---:|---:|---:|
| G86 (GPT-2-small shape) | 768 / 12 / 12 | 86.5M | 86.5M | 1.38 GB | — |
| **G86-MoE8** | 768 / 12 / 12 | **483.0M** | 143.2M | **7.73 GB** | 12.3 GB |
| M100 (true ~100M) | 768 / 14 / 12 | 100.7M | 100.7M | 1.61 GB | 5.2 GB |
| M100-MoE8 | 768 / 14 / 12 | 563.2M | 166.8M | 9.01 GB | 14.3 GB |
| **S25** | 512 / 8 / 8 | **26.2M** | 26.2M | 0.42 GB | 2.2 GB |
| **S25-MoE8** | 512 / 8 / 8 | **143.7M** | 43.0M | **2.30 GB** | **4.7 GB** |

- **The ~485M / ~7.8 GB figures are confirmed** (483.0M, 7.73 GB) for the
  GPT-2-small shape, which with our vocabulary is 86.5M dense rather than
  100M. At a true 100M it is worse: 563.2M and 9.01 GB.
- **"Exceeds 6 GB" holds for the local GPU:** the optimizer state alone is
  larger than the RTX 4050's 6,141 MiB. **It does not hold for a T4:** the
  estimates put 100M-MoE at 12–14 GB, within 16 GB. So memory is not what rules
  out 100M ablations on Kaggle. Compute and data are: each 100M-MoE run costs
  as much as option (a) or more, for a base model the data does not support.
- **S25-MoE8 fits everywhere:** ~4.7 GB estimated at micro-batch 16 on the
  local GPU, 1.5–2.5 T4-hours per run on the full 510M tokens.
- **Design note for Phase 4's own ADR:** top-2 over 8 experts makes 43.0M
  params active against the dense base's 26.2M, so dense vs MoE is not
  FLOP-matched. Top-1 routing (26.2M active) or a wider dense baseline are the
  two ways to match it; that choice belongs to Phase 4.

## 3. Sizes used

All share head_dim 64, MLP 4 x d_model, context 512, vocab 2,049 (tied), and
the architecture above.

| Name | Role | d_model | n_layers | n_heads | Params | Tokens/param (510.5M) |
|---|---|---:|---:|---:|---:|---:|
| **S25** | Phase 3 model; Phase 4 dense base | **512** | **8** | **8** | 26.2M | 19.5 |
| **S25-MoE8** | Phase 4 MoE (8 experts, top-2) | 512 | 8 | 8 | 143.7M total / 43.0M active | 11.9 (active) |
| S13 | Phase 5 ladder | 384 | 8 | 6 | 14.9M | 34 |
| S7 | Phase 5 ladder | 320 | 6 | 5 | 8.0M | 64 |
| S3 | Phase 5 ladder | 256 | 4 | 4 | 3.7M | 139 |

The Phase 5 ladder spans 139 → 19.5 tokens/param on the same corpus: from
heavily data-rich down to Chinchilla-optimal. The whole ladder, one epoch per
size, costs about 2.1–3.3 T4-hours. Extending it above S25, into the
data-constrained regime option (c) would have targeted, is cheap one size at a
time and needs no decision now.

M100 and G86 are recorded for comparison only; neither is trained.

## Sources
- TinyStories-train size: pinned in `crucible/data.py` (SHA-verified).
- Separator share and rate: measured on TinyStories-valid (ADR-0001).
- Bytes/token at 2,048: ADR-0001 (TinyStories). The FineWeb-Edu figures are
  from 1,000 random `sample-10BT` documents via HF datasets-server (random
  seed 0), using 500 to fit and 500 to measure.
- FineWeb-Edu: dataset card and HF API, `HuggingFaceFW/fineweb-edu` at
  revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- T4: NVIDIA T4 datasheet (Mar 2019): 65 TFLOPS FP16/FP32 mixed, 8.1 TFLOPS
  FP32, 16 GB GDDR6, 300 GB/s.
- Kaggle limits: Kaggle's own announcements ([T4 x2
  launch](https://www.kaggle.com/product-feedback/361104), [floating
  quota](https://www.kaggle.com/product-feedback/173129)) and a secondary
  summary for the 12-hour session and 20 GB limits
  ([AIMultiple](https://aimultiple.com/free-cloud-gpu)). Kaggle's docs pages
  render client-side and could not be read directly; confirm in the Kaggle UI
  before Phase 3.
- Local GPU: `nvidia-smi`, RTX 4050 Laptop, 6,141 MiB.
- Eldan & Li 2023, *TinyStories* (arXiv 2305.07759); Hoffmann et al. 2022
  (Chinchilla); Muennighoff et al. 2023 (data-constrained scaling); Korthikanti
  et al. 2022 (activation memory); Chowdhery et al. 2022 (PaLM FLOPs).

## Consequences
- **ADR-0001's rationale shifts, and its decision holds.** ADR-0001 broke the
  2,048 vs 4,096 tie with a token budget framed against 100M params (~2.0B
  tokens). At 25M the Chinchilla target is ~0.52B: 2,048 supplies 97% of it,
  4,096 (466.6M tokens) 89%. The token argument is weaker. The budget argument
  is stronger: at S25 the 2,048 embedding is 4.0% of parameters, and 4,096
  would make it 7.7%. Same answer; ADR-0001's text still cites the 100M
  budget.
- ~~**CLAUDE.md and the README still describe a ~100M model.**~~ **Done
  2026-09-23/25:** the README, CLAUDE.md, the Phase 1 writeup and
  `scripts/sweep_vocab.py` (with its regenerated figure and CSV) all state
  ~25M. Remaining 100M mentions in the repo are historical by design: this
  ADR's rejected options, ADR-0001's original text under its amendment, and
  `model_budget.py`'s M100 config, which exists to be compared against.
- **Phase 3 opens by measuring MFU** on a T4 and re-running
  `model_budget.py`. If S25 lands well below 25% MFU, hours scale up
  proportionally. The decision does not change until they exceed a session.
- **No FineWeb-Edu download.** Data stays at 1.92 GB, one domain, one
  tokenizer.
- **Reversible:** if Phase 5 shows the ladder still improving steeply at S25,
  option (a) is the documented path up, with its costs already computed.
