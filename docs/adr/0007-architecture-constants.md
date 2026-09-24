# ADR-0007 — Transformer architecture constants

- **Status:** accepted
- **Date:** 2026-09-24
- **Applies to:** `crucible/model.py` (Phase 2). Shapes come from ADR-0006;
  this ADR covers the constants ADR-0006 left open.

## Context
ADR-0006 fixed the shape of S25: d_model 512, 8 layers, 8 heads (head_dim 64),
d_ff 2048, context 512, vocab 2,049, tied embeddings. Building the model
needed several more constants, which the Phase 2 scaffolding carried as
PENDING placeholders. CLAUDE.md requires an ADR for each.

None of these is a free parameter in a meaningful sense yet: nothing has been
trained, so there is no measurement to choose between alternatives. They are
recorded as *defaults with reasons*, to be revisited with evidence in Phase 3.

## Decisions

| Constant | Value | Why |
|---|---:|---|
| `rope_theta` | 10,000.0 | The original RoPE base (Su et al., 2021), and what GPT-NeoX, LLaMA 1/2 and most small models use at short context. Larger bases exist to extend context beyond training length; at context 512 there is nothing to extend. |
| `norm_eps` | 1e-6 | Standard for RMSNorm (LLaMA uses 1e-6, some models 1e-5). It only has to keep an all-zero row finite, and sits well below any real activation scale in fp32. |
| `init_std` | 0.02 | GPT-2's initialisation, carried by nanoGPT and most reimplementations. At d_model 512 it is about half of 1/sqrt(d_model) = 0.044: conservative rather than aggressive. |
| Activation | GELU | GPT-2's choice, and what the d_ff = 4 x d_model width assumes. SwiGLU is the modern alternative but changes the parameter count: it needs three matrices at 2/3 the width to stay equal, which would break ADR-0006's arithmetic. |
| Biases | none | No bias on any projection, following GPT-2-without-bias practice (nanoGPT, LLaMA). Keeps ADR-0006's count exact and drops a term that contributes little at this scale. |
| Normalisation placement | pre-norm | Each sub-layer normalises its input; the residual stream runs unmodified from block input to output. Post-norm needs warmup tricks to stay stable with depth; pre-norm trains without them. Pinned by `test_block_with_zeroed_outputs_is_identity`. |
| Norm compute dtype | fp32 | RMSNorm computes the reciprocal square root in fp32 and casts back, so mixed-precision training cannot lose the scale. Costs nothing measurable at this size. |
| Attention masking | explicit boolean mask | A query at absolute position p attends to keys 0..p, built from positions rather than `is_causal`. `is_causal` assumes queries and keys share an origin, which is false during cached decoding; one mask covers both paths, so cached and uncached results agree (`test_kv_cache_matches_full_forward`). |
| Attention kernel | `F.scaled_dot_product_attention` | PyTorch picks the fused kernel. Writing the softmax by hand would read better for G1 but runs slower and is less numerically careful; the explicit mask keeps the causal structure visible anyway. |
| RoPE layout | split-half | The head dimension splits into two halves that rotate against each other, as in the original implementation, rather than interleaved pairs. The two differ by a permutation of head dimensions and are equivalent as long as one convention is used throughout. |

## Consequences
- **Nothing here is measured.** Each value is a defensible default, not a
  result. Phase 3 should re-examine `init_std` first (it shows up immediately
  in a loss curve), and the activation choice if throughput disappoints.
- **Parameter count is unchanged:** 26,223,616 for S25, exactly as ADR-0006
  computed, pinned by `test_s25_parameter_count_matches_adr_0006`.
- **Switching to SwiGLU later means re-running `scripts/model_budget.py`,**
  because d_ff would have to change to keep the count.
- **The mask costs a little memory** at long context (a T x T boolean), which
  is irrelevant at 512 and worth revisiting if context grows.
