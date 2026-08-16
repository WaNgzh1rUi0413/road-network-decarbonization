"""Verify the versioned study inputs against SHA-256 checksums."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "checksums.csv"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    failures = []
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            failures.append(f"missing: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']}")
        elif digest(path) != row["sha256"]:
            failures.append(f"checksum mismatch: {row['path']}")
    if failures:
        raise SystemExit("Input verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(rows)} input files.")


if __name__ == "__main__":
    main()

