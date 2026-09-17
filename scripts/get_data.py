"""Download TinyStories into data/ and verify it.

    python scripts/get_data.py          # validation set only (~19 MB)
    python scripts/get_data.py --full   # also the training set (~1.9 GB)

Interrupted downloads resume from `<name>.part`. A file is renamed into place
only after its size, SHA-256 and UTF-8 decoding all check out, so anything in
data/ without a .part suffix has been verified.

Standard library only.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crucible.data import DATA_DIR, TRAIN, VALID, DataFile  # noqa: E402

CHUNK_BYTES = 1 << 20  # 1 MiB: bounds memory on the 1.9 GB file
MAX_ATTEMPTS = 5  # consecutive network failures before giving up
RETRY_BACKOFF_S = 2.0  # doubled after each failed attempt
TIMEOUT_S = 60


class VerificationError(Exception):
    pass


def _progress(done: int, total: int, started: float, resumed_from: int) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = (done - resumed_from) / elapsed / 1e6
    print(
        f"\r  {done / 1e6:9.1f} / {total / 1e6:.1f} MB "
        f"({100 * done / total:5.1f}%)  {rate:6.1f} MB/s",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _fetch_once(df: DataFile, part: Path) -> None:
    """Continue `part` towards `df.size` bytes. Raises on network errors."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > df.size:
        print(f"  {part.name} is larger than expected; restarting", file=sys.stderr)
        part.unlink()
        offset = 0
    if offset == df.size:
        return

    req = urllib.request.Request(df.url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        print(f"  resuming at {offset / 1e6:.1f} MB", file=sys.stderr)

    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        if offset and resp.status == 206:
            match = re.match(r"bytes (\d+)-", resp.headers.get("Content-Range", ""))
            if not match or int(match.group(1)) != offset:
                raise VerificationError(
                    f"server resumed at the wrong offset: "
                    f"{resp.headers.get('Content-Range')!r}, wanted {offset}"
                )
            mode = "ab"
        else:
            # 200: the server ignored the Range header and is sending it all.
            if offset:
                print("  server ignored Range; restarting", file=sys.stderr)
            offset, mode = 0, "wb"

        started, done = time.monotonic(), offset
        with part.open(mode) as out:
            while chunk := resp.read(CHUNK_BYTES):
                out.write(chunk)
                done += len(chunk)
                _progress(done, df.size, started, offset)
        print(file=sys.stderr)


def download(df: DataFile) -> dict[str, int]:
    part = df.path.with_name(df.path.name + ".part")
    backoff = RETRY_BACKOFF_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _fetch_once(df, part)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(
                f"\n  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}", file=sys.stderr
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(backoff)
            backoff *= 2

    got = part.stat().st_size
    if got != df.size:
        raise VerificationError(
            f"{part.name}: {got} bytes, expected {df.size}. Re-run to resume."
        )
    stats = verify(df, part)
    part.replace(df.path)
    return stats


def verify(df: DataFile, path: Path) -> dict[str, int]:
    """One streaming pass: size, SHA-256, strict UTF-8, and corpus stats."""
    size = path.stat().st_size
    if size != df.size:
        raise VerificationError(f"{path.name}: {size} bytes, expected {df.size}")

    sha = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    lines = 0
    chars: set[str] = set()
    offset = 0
    last = b""
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            sha.update(chunk)
            try:
                chars.update(decoder.decode(chunk))
            except UnicodeDecodeError as exc:
                raise VerificationError(
                    f"{path.name}: invalid UTF-8 near byte {offset + exc.start}"
                ) from exc
            lines += chunk.count(b"\n")
            offset += len(chunk)
            last = chunk[-1:]
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{path.name}: truncated UTF-8 at end") from exc

    if sha.hexdigest() != df.sha256:
        raise VerificationError(
            f"{path.name}: SHA-256 {sha.hexdigest()} != pinned {df.sha256}. "
            f"Delete the file and re-run."
        )
    if size and last != b"\n":
        lines += 1  # final line without a trailing newline
    return {"bytes": size, "lines": lines, "unique_chars": len(chars)}


def fetch(df: DataFile) -> None:
    print(f"{df.name}")
    if df.path.exists():
        print("  present; verifying", file=sys.stderr)
        stats = verify(df, df.path)
    else:
        stats = download(df)
    print(f"  bytes:        {stats['bytes']:,}")
    print(f"  lines:        {stats['lines']:,}")
    print(f"  unique chars: {stats['unique_chars']:,}")
    print("  size, sha256, utf-8: ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"also download {TRAIN.name} ({TRAIN.size / 1e9:.1f} GB)",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(exist_ok=True)
    try:
        fetch(VALID)
        if args.full:
            fetch(TRAIN)
    except VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
