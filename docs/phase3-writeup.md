# Crucible Phase 3 — training a 26M-parameter model, and the numbers that moved

**Status:** Phase 3 trained and scored. C3 passed; C4's ten samples are recorded and **awaiting my
score**; the T4 MFU measurement is still owed.
Repo: `github.com/hashithhh/crucible`
**Date:** 2026-09-28

Phase 3 is the training run: take the Phase 1 tokenizer and the Phase 2 transformer, build the
pipeline between them, and train S25 — 26.2M parameters — on TinyStories until it writes
coherent text. This is what was built, what was measured, and where the first number was wrong.

**On how this was built:** as in Phases 1 and 2, the code was written by Claude, not by hand, and
every commit says so. Two further things were Claude's in this phase that are easy to lose track
of, so I'm stating them here rather than at the end: **the pass marks for both Phase 3 checks were
set by Claude**, without my review, after I'd left three requests for them unanswered and told it
to finish the phase. They were written down before the run started, which is the part that
matters, and the ledger records whose they are. **The judgement on the text samples was not
delegated** — that one is mine. Details in [The honest part](#the-honest-part).

---

## What it is

| | |
|---|---|
| Model | S25: 8 layers, d_model 512, 8 heads, context 512, 26,223,616 parameters |
| Data | TinyStories, deduplicated: 432,175,881 tokens, 0.5% held out |
| Run | one epoch, 6,561 steps × 65,536 tokens, AdamW, cosine to 10% after 200 warmup steps |
| Hardware | RTX 4050 Laptop (6 GB), bf16 autocast |
| Held-out loss | **1.3383 nats/token — 0.517 bits/byte** |
| C3 (loss) | **pass** (bar 2.60); the non-gating stretch of 2.00 was also met |
| C4 (text) | samples recorded; **score pending** — see [Text](#text) |
| MFU | **32.3%** on the local card, against a measured matmul ceiling |

---

## The data was 15% smaller than the plan assumed

ADR-0006 sized the model from an estimate: 1.92 GB of text at the tokenizer's 3.73 bytes/token
gives 510.5M tokens, and 510.5M tokens at Chinchilla's ~20 tokens per parameter supports ~25M
parameters. That is where S25 came from.

Encoding the corpus replaced the estimate with a count:

| | Estimated | Measured |
|---|---:|---:|
| Stories | not counted | 2,119,489 |
| Exact duplicates | not modelled | **320,241 (15.1%)** |
| Tokens | 510,500,000 | **432,175,881 (−15.3%)** |
| Tokens per S25 parameter | 19.5 | **16.48** |

The gap is entirely duplicates. Dedup removed 15.1% of stories and 15.3% of tokens, a close enough
match to say nothing else went missing — and the verification script confirms it exactly: the
separator token appears 1,799,248 times across the shards, once per unique story. So 19.5 was never
a real number. It counted the corpus with its repeats included.

**S25 was kept anyway.** The Chinchilla ratio is a compute-optimal argument, for when model size and
data are both free to move. Here the data is fixed, and shrinking to the 21.6M parameters that
would restore 20 tokens/param would reach a *worse* loss on the same corpus. Shrinking a model to
make a ratio come out is optimising the justification rather than the model. ADR-0006 was amended
rather than rewritten, and the cost is stated in it plainly: S25 sees ~82% of the tokens Chinchilla
would spend on a model its size.

Encoding itself went from ~97 minutes single-threaded to **7.2 minutes** across 16 worker
processes, at 1.0M tokens/s. Dedup and the train/val split both key off one 8-byte hash per story,
so the split is deterministic and independent of the order the corpus arrives in.

---

## The training constants, and which one was a guess

ADR-0008 carries every constant in the run with its source. Most are borrowed from GPT-3,
Chinchilla and nanoGPT and say so. Two are worth talking about.

**The batch is 65,536 tokens, not GPT-3's 0.5M.** Copying 0.5M would give 860 optimizer steps for
the whole epoch — too few for a cosine schedule to be a schedule at all. 65,536 gives 6,561.

**The learning rate was the constant most likely to be wrong, and the ADR said so before the
run.** 6e-4 is GPT-3 Small's value, at five times S25's size and eight times its batch. Those pull
in opposite directions — smaller models tolerate larger learning rates, smaller batches want smaller
ones — and square-root scaling on the batch alone argues for ~2e-4. 6e-4 was the less conservative
reading, so it got a revision trigger written down in advance: halve it if the pre-clip gradient
norm sits above the clip threshold for most of the first 500 steps.

It didn't fire. The norm exceeded 1.0 on **38%** of early logged steps, settled to ~0.4 by step
1,000, and ended the run at ~0.33. That is now an observation rather than a hope.

---

## Six things that went wrong, or nearly did

### 1. The resume test failed by 0.001, and the bug was in the test

Kaggle sessions end after 12 hours, so the run has to resume from a checkpoint *exactly*. The test
for that trains 8 steps straight, then 4 steps plus a resume, and compares the losses. It failed —
by about 1e-3.

The loop was fine. The test had stopped the first run by passing `steps=4`, which also moved the
cosine schedule's end point to step 4. The two runs had different learning rates before the break
ever happened. The fix is a `--stop-at` flag that pauses a run *without moving its horizon*, which
is also what a timed-out Kaggle session actually does: the session ends, the schedule doesn't.

### 2. A passing resume test can pass for the wrong reason

Weights and optimizer state are the obvious half of a checkpoint. The half that's easy to forget is
the **data order**: a run that resumes with a fresh sampler re-sees batches it already trained on,
skips others, and prints a perfectly believable loss curve.

The sampler's RNG state goes in the checkpoint. But a test that checks resume works could still
pass if the run were too short for two different continuations to disagree — so a second test
deliberately rewinds *only* the sampler and asserts that the losses then **diverge**. The first
test is only evidence because the second one can fail.

### 3. A smoke-test checkpoint was sitting exactly where the sampler looks

The first smoke run of the training script wrote a 20-step model to `checkpoints/final.pt` — before
the script grew a flag to put smoke output elsewhere. `final.pt` is also the default path of the
script that generates the text samples for C4.

It was caught by listing the checkpoint directory before ending a session, not by any test. Had
it not been, C4 would have sampled a 20-step model and recorded it as the trained one.

### 4. The run was killed three times, and resuming was the design working

The run lived through four processes. Three times the machine ran critically low on memory while
the session was idle and the tooling stopped the job, and each time it resumed from the latest
checkpoint:

| Process | Started from | Stopped at | Steps lost |
|---|---:|---:|---:|
| 1 | step 0 | 1,220 (memory) | 220 |
| 2 | 1,000 | 3,140 (memory) | 140 |
| 3 | 3,000 | 4,500 (memory) | 0 — it had just checkpointed |
| 4 | 4,500 | 6,561, done | — |

Resuming cost 360 steps in total and changed nothing else — same schedule, same data order — which
is the whole reason item 3.4 insisted on *exact* resume rather than "close enough". At the two
resumes checked by hand, the first logged loss matched the loss at the stop: 1.474 against a
held-out 1.4751 at step 3,000, and 1.324 exactly at step 4,500. This was the only kind of failure
in the phase that was genuinely external, and the one the design had already paid for.

### 5. A 285-minute sleep exposed a metric that lied

Mid-run, the laptop slept for 285 minutes between steps 1,860 and 1,870. The process froze, woke,
and carried on at full speed; nothing was lost. But afterwards the run log's `tokens_per_s` field
read ~3,200 while the GPU was doing ~42,000.

That field is a cumulative average since the process started, so one stall poisons every record
after it. The MFU script took its median from that field, and by the end of the run it would have
reported MFU several times too low. It now computes rates between consecutive log records, never
across a resume, and has a regression test built from the stall and the resumes this run actually
had. The training script's own field was left alone rather than changed mid-run, so its console
figure is still misleading; that is cosmetic and recorded.

### 6. The thresholds were set conservatively

C3's pass mark was 2.60 and its non-gating stretch 2.00. Held-out loss crossed the pass mark before
step 500 and the stretch by step 750 — 11% into the run, with the cosine still near its peak — and
finished at **1.3383, about half the pass mark**.

The bars were anchored to the bigram baseline rather than to an estimate of what S25 could reach,
and that choice made them too easy. They were not moved once numbers existed: moving a bar mid-run
is what the ledger exists to prevent. The honest reading is that C3 confirms the pipeline works and
says little about how good the model is. A tighter bar needs a reference this project didn't have
before this run — and now does.

---

## The results

### Loss

Held-out loss over the run, full validation split (2,177,071 tokens) at each eval:

| Step | Held-out loss | Step | Held-out loss |
|---:|---:|---:|---:|
| 250 | 2.6894 | 3,000 | 1.4751 |
| 500 | 2.0538 | 4,000 | 1.4144 |
| 750 | 1.8600 | 5,000 | 1.3706 |
| 1,000 | 1.7536 | 6,000 | 1.3444 |
| 2,000 | 1.5663 | **6,561 (final)** | **1.3383** |

The curve was still falling at the horizon, but slowly — 0.006 over the last 561 steps, with the
learning rate already at its 6e-5 floor. ADR-0006 recorded a second partial epoch as the lever if
the curve were still falling *steeply*; it isn't, so that lever isn't pulled.

C3 is scored against baselines measured on the same held-out split **before the model existed**:

| Predictor | Held-out loss (nats/token) | bits/byte |
|---|---:|---:|
| Uniform over 2,049 tokens | 7.6251 | 2.947 |
| Unigram frequencies | 6.0734 | 2.347 |
| Bigram table, smoothed | 3.6190 | 1.399 |
| **S25, final checkpoint** | **1.3383** | **0.517** |

The registered bar: **hard fail at 3.6190** (a transformer that can't beat a table of pair counts
has failed to work), **pass at 2.60** (≈1 bit per byte), and a non-gating **stretch at 2.00**.

**C3: pass.** 1.3383 is 63% below the bigram baseline, and the stretch is met. The scorer reads the
three bounds out of the ledger rather than holding its own copy, and refuses any checkpoint short
of the 6,561-step horizon, so this is the registered check on the registered checkpoint and not the
best number seen along the way.

### Text

The ten C4 prompts were fixed in the ledger before the model was trained. One sample each at
temperature 0.8, top-p 0.95, up to 200 new tokens, seed 1337 — no re-rolls. All ten are in
`results/c4_samples.md`, verbatim. The first one, unedited:

> **Once upon a time, there was a little girl named Lily.** She loved to eat snacks, especially
> crackers and cheese. One day, she went to the kitchen and saw her mom washing dishes. She asked
> her mom if she could help, but her mom said no because the floor was very dirty. Lily was sad and
> didn't understand why she couldn't help. She started to cry and said, "I want to help, Mommy.
> Please let me help." ...

It keeps its characters straight across the passage, carries dialogue, and follows a small plot
through a conflict towards a resolution. Not every sample is that tidy — in one the *wind* is the
character who says "I am scared", in another a cake is praised for being "so flexible" — and
deciding whether those count is what the C4 criteria are for.

**C4: [pending: my count out of 10]**

### Throughput

| | |
|---|---|
| Throughput | 43,662 tokens/s (median of 692 step intervals from the real run) |
| FLOPs per token | 182.5M (6N plus the attention term) |
| Achieved | 7.97 TFLOP/s |
| Matmul ceiling, **measured** on this card | 24.66 TFLOP/s |
| **MFU** | **32.3%** |

That is inside the 25–40% band ADR-0006 assumed when it budgeted T4 hours, even though the Phase 2
smoke measurement had been read as pointing below it. The difference is the denominator. The
ceiling here was *measured*, with large dense bf16 matmuls on this device, rather than taken from a
spec sheet. A laptop part's published peak is higher than what it sustains in practice, so an MFU
computed against the spec sheet would read lower than 32.3% for the same run. I've used the measured
ceiling because it makes the figure a statement about the training loop rather than about the
laptop's power and thermal limits — and I'm saying so, so nobody has to guess which one they're
reading.

---

## The honest part

**The code is Claude's.** As in Phases 1 and 2. "Trained a small language model with AI
assistance" is true. "Wrote a training pipeline by hand" is not true of this repo.

**The pass marks are Claude's too.** C3's and C4's thresholds were set by Claude, without my
review, after I'd left three requests for them unanswered. The ledger says so in both entries
rather than presenting them as my bar. What made them checks rather than decoration is that they
were committed to the repo before the run started, and neither was moved once numbers existed —
including when C3 turned out to be too easy (see above).

**C4 is the one judgement I didn't delegate.** The ten text samples were generated with prompts
and settings fixed in the ledger before the model was trained, one sample each, no re-rolls, and
recorded verbatim. [pending: C4 scoring note — only say "I scored them" once Hashith has]

**Some of the pipeline is written but unexercised.** The fp16 path with a loss scaler exists for
Kaggle's T4, which has no bfloat16. There's no Turing GPU here, so it has never run. MFU was
measured on the local card only; the T4 figure that ADR-0006 said Phase 3 owes is still owed.

**This is the third phase with a dropped or delegated check**, after the minbpe comparison in
Phase 1 and the from-memory rebuild in Phase 2. This one is different in kind — the thresholds were
registered and met, not skipped — but the pattern of the checks drifting towards whoever wrote the
code is worth saying out loud.

---

## What's next

Phase 4: ablations on S25 as the dense base. The Phase 3 run is the reference every ablation is
measured against, which is why its loss curve and its constants are all on the record — and why a
C3-style bar for Phase 4 can be set against 1.3383 rather than against a bigram table.
