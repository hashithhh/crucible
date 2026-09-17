"""Corpus access for Phase 1 sweeps.

The corpus is TinyStories, fetched by `scripts/get_data.py` into `data/`.
Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

DATASET_REPO = "roneneldan/TinyStories"


@dataclass(frozen=True)
class DataFile:
    """A pinned file from the dataset repo.

    `size` and `sha256` are the values HuggingFace publishes for the LFS
    object (`X-Linked-Size`, `X-Linked-ETag`). Pinning them means a silent
    upstream change fails verification instead of quietly changing every
    measurement downstream.
    """

    name: str
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return DATA_DIR / self.name

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{self.name}"
        )


VALID = DataFile(
    name="TinyStories-valid.txt",
    size=19_447_282,
    sha256="94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
)
TRAIN = DataFile(
    name="TinyStories-train.txt",
    size=1_924_281_556,
    sha256="c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
)


def load_corpus(n_bytes: int | None = None) -> str:
    """Return the TinyStories validation set, or a prefix of it.

    - `n_bytes=None` returns the whole file.
    - Otherwise returns at most `n_bytes` bytes of UTF-8, cut after the last
      newline that fits, so no line is split. If no complete line fits, the
      result is `""`. If the file is shorter than `n_bytes`, it is returned
      whole, including a final line with no trailing newline.
    - Negative `n_bytes` raises ValueError.
    - A missing file raises FileNotFoundError naming the script that fetches it.
    """
    if n_bytes is not None and n_bytes < 0:
        raise ValueError(f"n_bytes must be >= 0 or None, got {n_bytes}")

    path = VALID.path
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/get_data.py")

    size = path.stat().st_size
    with path.open("rb") as f:
        if n_bytes is None or n_bytes >= size:
            # Also caps the read: f.read(n) preallocates n bytes, so a large
            # n_bytes would otherwise raise MemoryError on a small file.
            return f.read().decode("utf-8")
        buf = f.read(n_bytes)

    if len(buf) == n_bytes:
        # b"\n" never occurs inside a multi-byte UTF-8 sequence, so cutting
        # just after it is always a valid decode boundary.
        buf = buf[: buf.rfind(b"\n") + 1]
    return buf.decode("utf-8")
