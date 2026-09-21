"""ADR-0006 arithmetic: parameters, memory, data and T4 hours from explicit configs.

    python scripts/model_budget.py    # prints tables; writes results/model_budget.json

Arithmetic only. There is no model code here: parameter counts follow from the
architecture stated in CONFIGS, and every measured input is a named constant
with its source in docs/adr/0006-model-size-and-data.md. Change an input and
the ADR's numbers regenerate.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# --- measured inputs ------------------------------------------------------------

VOCAB = 2048 + 1  # ADR-0001 learned vocab + <|endoftext|>
CONTEXT = 512  # ADR-0001: stories average ~220 tokens
TS_TRAIN_BYTES = 1_924_281_556  # TinyStories-train, pinned size (crucible/data.py)
SEPARATOR_BYTE_SHARE = 0.0158  # <|endoftext|> lines, measured on TinyStories-valid
SEPARATORS_PER_BYTE = 21_989 / 19_447_282  # separators / bytes, TinyStories-valid
TS_BPT_TS_TOKENIZER = 3.7258  # ADR-0001, held-out
TS_BPT_MIX_TOKENIZER = 3.537  # 2048 vocab trained on a 50/50 TS/FWE mix
FWE_BPT_MIX_TOKENIZER = 2.890  # same tokenizer, on held-out FineWeb-Edu docs
FWE_PARQUET_BYTES = 28_518_193_415  # sample-10BT, 14 shards (HF datasets-server)
FWE_ROWS = 9_672_101
FWE_TEXT_BYTES_PER_DOC = 5_030.2  # mean over 1,000 random docs
FWE_SHARD_BYTES = 2_153_000_000  # typical shard; 13 of 14 are 2.152-2.153 GB

T4_PEAK_FLOPS = 65e12  # NVIDIA T4 datasheet, mixed precision FP16/FP32
MFU_BAND = (0.25, 0.40)  # ASSUMED achieved fraction of peak; measure in Phase 3
DDP_2XT4_SPEEDUP = 1.8  # ASSUMED scaling for Kaggle "T4 x2"
KAGGLE_WEEKLY_GPU_HOURS = 30
KAGGLE_SESSION_HOURS = 12

LOCAL_GPU_BYTES = 6141 * 2**20  # RTX 4050 Laptop, nvidia-smi
T4_GPU_BYTES = 16e9  # datasheet
ADAMW_BYTES_PER_PARAM = 16  # fp32 weights + grads + Adam m + v (AMP, fp32 master)
ACT_BYTES_PER_SBD_DENSE = 34  # Korthikanti et al. 2022, per layer, fused attention
ACT_BYTES_PER_SBD_MLP = 19  # the MLP share of the 34
CUDA_CONTEXT_BYTES = 0.5e9  # ASSUMED runtime + allocator overhead

PHASE3_TARGET_TOKENS = 2.0e9  # ~20 tokens/param (Chinchilla) x 100M
EPOCHS_DATA_CONSTRAINED = 4  # Muennighoff et al. 2023: ~4 epochs ~ unique data
UINT16_BYTES = 2  # stored token id; vocab 2,049 fits


@dataclass(frozen=True)
class Config:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_experts: int = 1  # 1 = dense
    top_k: int = 1

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model

    def params(self) -> dict[str, int]:
        """Pre-LN GPT, no biases, RoPE (no position params), tied embeddings.

        Per layer: attention 4 d^2 (Q, K, V, O); MLP 2 d d_ff = 8 d^2 per
        expert; router d x E if MoE; two LayerNorm weight vectors. Plus the
        token embedding (tied with the output head) and a final LayerNorm.
        """
        d, n_layers, n_exp = self.d_model, self.n_layers, self.n_experts
        attn = 4 * d * d * n_layers
        mlp_one = 2 * d * self.d_ff * n_layers
        router = d * n_exp * n_layers if n_exp > 1 else 0
        norms = 2 * d * n_layers + d
        emb = VOCAB * d
        total = attn + n_exp * mlp_one + router + norms + emb
        active = attn + min(self.top_k, n_exp) * mlp_one + router + norms + emb
        return {
            "total": total,
            "active": active,
            "embedding": emb,
            "non_embedding": total - emb,
        }

    def train_flops_per_token(self) -> float:
        """6 x active params + 12 L T d for attention (PaLM, Chowdhery et al. 2022)."""
        attn_scores = 12 * self.n_layers * CONTEXT * self.d_model
        return 6 * self.params()["active"] + attn_scores

    def memory(self, micro_batch: int) -> dict[str, float]:
        p = self.params()["total"]
        sbd = CONTEXT * micro_batch * self.d_model
        extra_mlp = (self.top_k - 1) * ACT_BYTES_PER_SBD_MLP
        acts = sbd * (ACT_BYTES_PER_SBD_DENSE + extra_mlp) * self.n_layers
        logits = micro_batch * CONTEXT * VOCAB * 6  # fp16 logits + fp32 for loss
        state = p * ADAMW_BYTES_PER_PARAM
        return {
            "adamw_state_gb": state / 1e9,
            "activations_gb": acts / 1e9,
            "logits_gb": logits / 1e9,
            "total_gb": (state + acts + logits + CUDA_CONTEXT_BYTES) / 1e9,
        }


CONFIGS = {
    # Phase 3 option (b) base, and the Phase 4 ablation base.
    "S25": Config("S25", d_model=512, n_layers=8, n_heads=8),
    "S25-MoE8": Config("S25-MoE8", 512, 8, 8, n_experts=8, top_k=2),
    # Phase 3 options (a)/(c): ~100M dense.
    "M100": Config("M100", d_model=768, n_layers=14, n_heads=12),
    "M100-MoE8": Config("M100-MoE8", 768, 14, 12, n_experts=8, top_k=2),
    # GPT-2-small shape; the ~485M MoE figure in the request corresponds to it.
    "G86": Config("G86", d_model=768, n_layers=12, n_heads=12),
    "G86-MoE8": Config("G86-MoE8", 768, 12, 12, n_experts=8, top_k=2),
    # Phase 5 ladder below the base (head_dim 64 throughout).
    "S3": Config("S3", d_model=256, n_layers=4, n_heads=4),
    "S7": Config("S7", d_model=320, n_layers=6, n_heads=5),
    "S13": Config("S13", d_model=384, n_layers=8, n_heads=6),
}


def ts_tokens(bpt: float) -> float:
    text = TS_TRAIN_BYTES * (1 - SEPARATOR_BYTE_SHARE)
    return text / bpt + TS_TRAIN_BYTES * SEPARATORS_PER_BYTE


def t4_hours(flops: float) -> dict[str, list[float]]:
    lo, hi = MFU_BAND
    fast = flops / (T4_PEAK_FLOPS * hi) / 3600
    slow = flops / (T4_PEAK_FLOPS * lo) / 3600
    return {
        "one_t4_hours": [round(fast, 1), round(slow, 1)],
        "t4x2_hours": [
            round(fast / DDP_2XT4_SPEEDUP, 1),
            round(slow / DDP_2XT4_SPEEDUP, 1),
        ],
    }


def option(cfg: Config, unique: float, seen: float, download: float) -> dict:
    return {
        "config": cfg.name,
        "params": cfg.params()["total"],
        "unique_tokens": unique,
        "tokens_seen": seen,
        "download_gb": download / 1e9,
        "disk_peak_gb": (download + unique * UINT16_BYTES) / 1e9,
        "disk_steady_gb": unique * UINT16_BYTES / 1e9,
        "flops": cfg.train_flops_per_token() * seen,
    }


def main() -> int:
    out: dict = {"configs": {}, "options": {}}
    print("config        d   L   H        total       active  emb%  AdamW GB")
    for key, c in CONFIGS.items():
        p = c.params()
        mem = {b: c.memory(b) for b in (8, 16, 32)}
        out["configs"][key] = {
            **asdict(c),
            "d_ff": c.d_ff,
            "params": p,
            "flops_per_token": c.train_flops_per_token(),
            "memory_by_micro_batch": mem,
        }
        print(
            f"{key:<10} {c.d_model:>4} {c.n_layers:>3} {c.n_heads:>3} "
            f"{p['total']:>12,} {p['active']:>12,} "
            f"{100 * p['embedding'] / p['total']:>5.1f} "
            f"{mem[16]['adamw_state_gb']:>9.2f}"
        )

    ts_ts = ts_tokens(TS_BPT_TS_TOKENIZER)
    ts_mix = ts_tokens(TS_BPT_MIX_TOKENIZER)
    fwe_tokens = PHASE3_TARGET_TOKENS - ts_mix
    fwe_text = fwe_tokens * FWE_BPT_MIX_TOKENIZER
    text_per_parquet = FWE_ROWS * FWE_TEXT_BYTES_PER_DOC / FWE_PARQUET_BYTES
    fwe_parquet = fwe_text / text_per_parquet
    shards = math.ceil(fwe_parquet / FWE_SHARD_BYTES)

    m100, s25 = CONFIGS["M100"], CONFIGS["S25"]
    a_download = TS_TRAIN_BYTES + shards * FWE_SHARD_BYTES
    opts = {
        "a_fineweb_edu_100M": option(
            m100, PHASE3_TARGET_TOKENS, PHASE3_TARGET_TOKENS, a_download
        ),
        "b_25M_tinystories": option(s25, ts_ts, ts_ts, TS_TRAIN_BYTES),
        "c_100M_1epoch": option(m100, ts_ts, ts_ts, TS_TRAIN_BYTES),
        "c_100M_4epochs": option(
            m100, ts_ts, EPOCHS_DATA_CONSTRAINED * ts_ts, TS_TRAIN_BYTES
        ),
    }
    opts["a_fineweb_edu_100M"].update(
        ts_tokens_mix_tokenizer=ts_mix,
        fwe_tokens=fwe_tokens,
        fwe_share=fwe_tokens / PHASE3_TARGET_TOKENS,
        fwe_text_gb=fwe_text / 1e9,
        fwe_text_per_parquet_byte=text_per_parquet,
        fwe_parquet_needed_gb=fwe_parquet / 1e9,
        fwe_shards_to_download=shards,
    )
    print()
    for key, o in opts.items():
        o.update(t4_hours(o["flops"]))
        o["tokens_per_param_unique"] = o["unique_tokens"] / o["params"]
        o["tokens_per_param_seen"] = o["tokens_seen"] / o["params"]
        o["weeks_of_kaggle_quota_1xT4"] = [
            round(h / KAGGLE_WEEKLY_GPU_HOURS, 2) for h in o["one_t4_hours"]
        ]
        print(
            f"{key:<20} N={o['params'] / 1e6:6.1f}M "
            f"unique={o['unique_tokens'] / 1e9:5.3f}B "
            f"seen/param={o['tokens_per_param_seen']:5.1f} "
            f"dl={o['download_gb']:5.2f}GB peak={o['disk_peak_gb']:5.2f}GB "
            f"steady={o['disk_steady_gb']:4.2f}GB "
            f"1xT4={o['one_t4_hours']}h 2xT4={o['t4x2_hours']}h"
        )
    a = opts["a_fineweb_edu_100M"]
    print(
        f"(a) detail: TS tokens, mix tokenizer "
        f"{a['ts_tokens_mix_tokenizer'] / 1e6:.1f}M; "
        f"FWE tokens {a['fwe_tokens'] / 1e9:.3f}B ({a['fwe_share']:.0%} of data); "
        f"FWE text {a['fwe_text_gb']:.2f}GB; "
        f"parquet {a['fwe_parquet_needed_gb']:.2f}GB "
        f"-> {a['fwe_shards_to_download']} shards"
    )
    moe = option(CONFIGS["S25-MoE8"], ts_ts, ts_ts, TS_TRAIN_BYTES)
    moe.update(t4_hours(moe["flops"]))
    opts["phase4_S25_MoE8_run"] = moe
    print(f"Phase 4 S25-MoE8 run on TinyStories: 1xT4={moe['one_t4_hours']}h")
    out["options"] = opts

    print("\nmemory at micro-batch 16, context 512 (GB):")
    for key in ("S25", "S25-MoE8", "M100", "G86-MoE8", "M100-MoE8"):
        m = CONFIGS[key].memory(16)
        caps = (("RTX4050-6GB", LOCAL_GPU_BYTES), ("T4-16GB", T4_GPU_BYTES))
        fits = [n for n, cap in caps if m["total_gb"] * 1e9 <= cap] or ["none"]
        print(
            f"  {key:<10} state={m['adamw_state_gb']:5.2f} "
            f"acts={m['activations_gb']:5.2f} logits={m['logits_gb']:4.2f} "
            f"total={m['total_gb']:5.2f}  fits: {', '.join(fits)}"
        )

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "model_budget.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
