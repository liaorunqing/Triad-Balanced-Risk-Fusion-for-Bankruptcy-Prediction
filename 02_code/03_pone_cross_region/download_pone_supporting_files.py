from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "03_data" / "01_raw" / "pone_global_2016"
ARTICLE_FILE_URL = (
    "https://journals.plos.org/plosone/article/file"
    "?id=10.1371/journal.pone.0166693.s{number:03d}&type=supplementary"
)
FILES = {
    1: "S1File.xlsx",
    2: "S2File.xlsx",
    3: "S3File.xlsx",
    4: "S4File.pdf",
    5: "S5File.xlsx",
    6: "S6File.xlsx",
    7: "S7File.xlsx",
    11: "S11File.xlsx",
    12: "S12File.xlsx",
    13: "S13File.xlsx",
    14: "S14File.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download publisher-hosted supporting files for Alaminos et al. "
            "(2016). Review the publisher and COMPUSTAT-related terms first."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--acknowledge-third-party-terms",
        action="store_true",
        help="Confirm that you reviewed the third-party data notice.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files that already exist.",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Triad-Risk-Fusion-Reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    if "text/html" in content_type.lower() or payload.lstrip().startswith(b"<!DOCTYPE"):
        raise RuntimeError(f"Publisher returned HTML instead of a data file: {url}")
    destination.write_bytes(payload)


def main() -> None:
    args = parse_args()
    if not args.acknowledge_third_party_terms:
        raise SystemExit(
            "Read 03_data/PONE_DATA_INSTRUCTIONS.md, then rerun with "
            "--acknowledge-third-party-terms."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, filename in FILES.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.overwrite:
            print(f"[skip] {destination}")
            continue
        url = ARTICLE_FILE_URL.format(number=number)
        print(f"[download] {url} -> {destination}")
        download(url, destination)
    print("Download complete. The files remain subject to their source terms.")


if __name__ == "__main__":
    main()
