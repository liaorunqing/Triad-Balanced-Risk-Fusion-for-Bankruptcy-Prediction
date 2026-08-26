"""Generate the public-release file manifest with SHA-256 checksums."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FILE_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or relative.as_posix() == OUTPUT.name:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    rows = []
    for path in included_files():
        rows.append(
            {
                "Path": path.relative_to(ROOT).as_posix(),
                "Bytes": path.stat().st_size,
                "SHA256": sha256(path),
            }
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Path", "Bytes", "SHA256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(rows)} entries.")


if __name__ == "__main__":
    main()
