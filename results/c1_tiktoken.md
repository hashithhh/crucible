# C1 — Crucible tokenizer vs tiktoken

- **Date:** 2026-09-22
- **Script:** `scripts/compare_tiktoken.py` → `results/c1_tiktoken.json`
- **Slice:** the ADR-0001 held-out slice, 574 stories / 499,260 bytes from the
  end of TinyStories-valid, `<|endoftext|>` removed. Our 2,048 row reproduces
  ADR-0001's 3.7258 bytes/token exactly (the script refuses to run otherwise).
- **tiktoken:** 0.14.0, `encode_ordinary`. Hardware as in ADR-0001 (AMD Ryzen 7
  7735HS, single-threaded encode, Python 3.13.5).

## This is not a like-for-like comparison

Ours is a **2,048-token** vocabulary trained on 5 MB of TinyStories. cl100k_base
(**100,277**) and o200k_base (**200,019**) are pretrained by OpenAI on far larger
and broader corpora: **49x and 98x** our vocabulary. The gap below is mostly
that difference, not the implementation. Read every number with that in mind.

## Results

| Tokenizer | Vocab | Tokens | Bytes/token | Extra tokens vs cl100k | Encode MB/s |
|---|---:|---:|---:|---:|---:|
| **Ours, ADR-0001** | **2,048** | 134,000 | **3.7258** | **+13.6%** | 0.33 |
| Ours, sample ceiling *(context)* | 14,143 | 116,355 | 4.2908 | −1.4% | 0.35 |
| cl100k_base | 100,277 | 117,971 | 4.2321 | — | 6.69 |
| o200k_base | 200,019 | 117,113 | 4.2631 | −0.7% | 11.23 |

Encode throughput is the median of 5 timed runs after an untimed warm-up.

## C1 against the stated target

The pre-registered record in the vault was **not found**: the only Obsidian
vault on this machine (`D:\PROJECTS\Nexus\nexus`) has no Crucible entry. The
target below is as stated in the Phase 1.4 request: **within 10%, kill
threshold >30% worse**, reference cl100k_base, "worse" read as extra tokens
for the same text.

| Measure | Value | Target ≤10% | Kill >30% |
|---|---:|---|---|
| Extra tokens vs cl100k_base | +13.6% | miss | not reached |
| Bytes/token below cl100k_base | −12.0% | miss | not reached |
| Extra tokens vs o200k_base | +14.4% | miss | not reached |

**Verdict as specified: misses the 10% target; well inside the kill threshold.**
The verdict is the same under either reading of "worse", and against either
tiktoken encoding.

## The target does not measure what it seems to

C1 as specified cannot tell a weak implementation from a small vocabulary, so it
should not gate anything on its own:

- **The gap is the vocabulary size.** The same code at the largest vocabulary
  this 5 MB sample supports (14,143, still 7x smaller than cl100k) compresses
  this slice **1.4% better** than cl100k and 0.6% better than o200k. Nothing in
  the implementation is costing 13.6%. The 2,048 choice is.
- **ADR-0001 already priced this, and prefers it.** ADR-0001 chose 2,048 over
  4,096 *because* it yields more tokens from a token-limited corpus. Under that
  rationale, "more tokens than cl100k" is the intended effect, not a defect. A
  C1 kill would overrule a recorded decision, not catch a bug.
- **The ceiling row flatters us, too.** It is trained on the same distribution
  it is tested on; tiktoken is general-purpose. So neither row is fair, in
  opposite directions.

A meaningful version of C1 would compare at matched vocabulary size and matched
training data, which tiktoken's pretrained encodings do not allow. The closest
available check, the ceiling row, says the implementation is competitive. If a
gate is wanted, re-register it explicitly (for example "ours at the sample
ceiling within 10% of cl100k") rather than read this result as a pass or fail.

## Throughput (not part of the target)

tiktoken encodes **20x (cl100k) to 34x (o200k) faster**: a compiled Rust core
against our pure-Python, cache-less encoder. At 0.33 MB/s one core would take
~97 minutes to encode the 1.9 GB training file. That matters for Phase 2 (data
preparation), not for tokenization quality; it is embarrassingly parallel
across stories.
