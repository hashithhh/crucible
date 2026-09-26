"""Phase 3.7 / C4: sample from the trained model and record the result.

    python scripts/sample.py                             # checkpoints/final.pt
    python scripts/sample.py --checkpoint checkpoints/step_003000.pt

Generates one continuation for each of C4's ten registered prompts and writes
all ten to `results/c4_samples.md` **verbatim, whatever they look like**.

THE PROMPTS ARE READ FROM THE LEDGER, NOT STORED HERE. C4 fixed them before
the model existed precisely so they could not be chosen to flatter it, and a
second copy in this file is a second thing that can quietly drift. If the
ledger's list is missing, or is not exactly ten prompts, this refuses to run
rather than sampling from an unregistered set.

One sample per prompt. No re-rolls and no cherry-picking: C4 says so, and a
script that could quietly re-roll would make the record worthless.

This script scores nothing. C4 is judged by a person reading the ten
passages, so its only job is to produce an honest record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.sampling import generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

LEDGER = ROOT / "results" / "LEDGER.md"
DEFAULT_CKPT = ROOT / "checkpoints" / "final.pt"
DEFAULT_MODEL = ROOT / "models" / "tinystories-2048"

# results/LEDGER.md, C4 -> Procedure. Changing any of these makes the run a
# different check from the one that was registered.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 200
SEED = 1337
EXPECTED_PROMPTS = 10

PROMPT_LINE = re.compile(r"^\d+\.\s+`(.+)`\s*$", re.MULTILINE)


def load_prompts(ledger: Path = LEDGER) -> list[str]:
    """The ten prompts C4 registered, read from the ledger itself."""
    if not ledger.is_file():
        raise FileNotFoundError(f"{ledger} not found; C4's prompts live there")
    text = ledger.read_text(encoding="utf-8")
    start = text.find("# C4 ")
    if start < 0:
        raise ValueError(f"{ledger.name} has no C4 entry")
    prompts = PROMPT_LINE.findall(text[start:])
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError(
            f"C4 registered {EXPECTED_PROMPTS} prompts; {ledger.name} now "
            f"yields {len(prompts)}. Refusing to sample from an unregistered "
            "prompt set."
        )
    return prompts


def load_model(path: Path, device: torch.device) -> tuple[Transformer, dict]:
    """Rebuild the model from the checkpoint's own recorded config."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), blob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    prompts = load_prompts()
    device = torch.device(args.device)
    model, blob = load_model(args.checkpoint, device)
    tok = Tokenizer()
    tok.load(str(args.model))
    eot = tok.vocab_size - 1

    generator = torch.Generator(device=device).manual_seed(SEED)
    samples = []
    for i, prompt in enumerate(prompts, 1):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = generate(
            model,
            ids,
            MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            generator=generator,
            eos_id=eot,
        )
        produced = out[0, ids.shape[1] :].tolist()
        new_ids = [t for t in produced if t != eot]
        continuation = tok.decode(new_ids)
        samples.append(
            {
                "n": i,
                "prompt": prompt,
                "continuation": continuation,
                "new_tokens": len(new_ids),
                "hit_separator": eot in produced,
            }
        )
        print(f"{i:>2}. {prompt}\n    {continuation}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "date": time.strftime("%Y-%m-%d"),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "step": blob.get("step"),
        "val_loss_at_save": blob.get("best_val"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "samples": samples,
    }
    (args.out / "c4_samples.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# C4 - samples from the trained S25",
        "",
        f"- **Checkpoint:** `{record['checkpoint']}`, step {record['step']:,}",
        f"- **Sampling:** temperature {TEMPERATURE}, top-p {TOP_P}, "
        f"up to {MAX_NEW_TOKENS} new tokens, seed {SEED}",
        f"- **Date:** {record['date']}",
        "",
        "All ten prompts are C4's, registered in `LEDGER.md` before this model",
        "existed. One sample each, no re-rolls, recorded verbatim including",
        "whatever is wrong with them. Scoring is C4's three criteria, judged by",
        "Hashith - not by the script that wrote this file.",
        "",
        "| # | Grammatical | On topic | Not looping | Counts |",
        "|---|---|---|---|---|",
        *[f"| {s['n']} |  |  |  |  |" for s in samples],
        "",
        "**Total: __ / 10**  (pass 7, hard fail 3 or below)",
        "",
        "---",
        "",
    ]
    for s in samples:
        lines += [
            f"## {s['n']}",
            "",
            f"**Prompt:** {s['prompt']}",
            "",
            "```",
            s["continuation"].strip() or "(empty)",
            "```",
            "",
            f"*{s['new_tokens']} new tokens; "
            f"{'reached' if s['hit_separator'] else 'did not reach'} the "
            "separator.*",
            "",
        ]
    (args.out / "c4_samples.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {args.out / 'c4_samples.md'} - C4 is scored by reading it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
