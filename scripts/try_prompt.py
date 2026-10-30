"""Ad-hoc, unofficial sampling — NOT part of C4, writes nothing to results/.

Use this to try your own prompts on the trained model. It has zero bearing
on the registered C4 record (results/c4_samples.*) and touches no files.

    python scripts/try_prompt.py --prompt "hi"
    python scripts/try_prompt.py --prompt "The robot looked at the sky" --tokens 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.sampling import generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

DEFAULT_CKPT = ROOT / "checkpoints" / "final.pt"
DEFAULT_MODEL = ROOT / "models" / "tinystories-2048"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model = model.to(device).eval()

    tok = Tokenizer()
    tok.load(str(args.model))
    eot = tok.vocab_size - 1

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(
        model,
        ids,
        args.tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generator=generator,
        eos_id=eot,
    )
    produced = out[0, ids.shape[1]:].tolist()
    continuation = tok.decode([t for t in produced if t != eot])

    print(f"\nPROMPT: {args.prompt}")
    print(f"OUTPUT: {continuation}\n")
    print("[unofficial - not part of C4, nothing written to results/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
