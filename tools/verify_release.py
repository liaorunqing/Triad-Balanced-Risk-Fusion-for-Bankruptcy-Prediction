"""Verify completeness and integrity of the public repository release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FILE_MANIFEST.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_metric(path: Path, method: str, metric: str) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") == method and row.get("metric") == metric:
                return float(row["mean"])
    raise RuntimeError(f"Metric not found: {method} / {metric} in {path}")


def close(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def verify_required_files() -> None:
    required = [
        "README.md",
        "CITATION.cff",
        "DATA_AVAILABILITY.md",
        "LICENSE_NOTICE.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RESULTS_INDEX.md",
        "02_code/01_core_polish_model/bankruptcy_baingo_kelm.py",
        "02_code/02_external_validation/run_external_validation.py",
        "02_code/03_pone_cross_region/run_cross_region_transfer.py",
        "02_code/04_manuscript_analysis/analyze_revised_results.py",
        "03_data/01_raw/polish_bankruptcy/1year.arff",
        "04_results/03_manuscript_tables/internal_revised_summary.csv",
        "04_results/03_manuscript_tables/cross_region_extended_summary.csv",
        "04_results/03_manuscript_tables/fair_internal_predictions.csv.gz",
        "05_figures/manuscript/figure_01_revised.pdf",
        "05_figures/manuscript/figure_07_revised.png",
        "FILE_MANIFEST.csv",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing required file: {relative}")


def verify_restricted_data_absent() -> None:
    for relative in [
        "03_data/01_raw/pone_global_2016",
        "03_data/02_processed/pone_global_2016",
    ]:
        directory = ROOT / relative
        require(directory.is_dir(), f"Missing placeholder directory: {relative}")
        unexpected = [path.name for path in directory.iterdir() if path.name != ".gitkeep"]
        require(not unexpected, f"Restricted data present in {relative}: {unexpected}")


def verify_headline_results() -> None:
    internal = ROOT / "04_results/03_manuscript_tables/internal_revised_summary.csv"
    cross = ROOT / "04_results/03_manuscript_tables/cross_region_extended_summary.csv"

    checks = [
        (find_metric(internal, "HistGB_fair", "mcc"), 0.513852),
        (find_metric(internal, "Triad_risk_fusion", "sensitivity"), 0.636200),
        (find_metric(cross, "Full_PBMSBAINGO", "brier"), 0.164769),
        (find_metric(cross, "HistGB", "ece_10"), 0.090951),
        (find_metric(cross, "KELM_score_fusion", "auc"), 0.843547),
        (find_metric(cross, "Balanced_logistic", "ap"), 0.855972),
    ]
    for actual, expected in checks:
        require(close(actual, expected), f"Headline result changed: {actual} != {expected}")


def verify_compressed_predictions() -> None:
    path = ROOT / "04_results/03_manuscript_tables/fair_internal_predictions.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    require("target" in header and "score" in header, "Prediction archive schema is invalid.")
    require(rows > 1000, "Prediction archive contains unexpectedly few rows.")


def verify_manifest() -> None:
    require(MANIFEST.is_file(), "FILE_MANIFEST.csv is missing.")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    listed = set()
    for row in rows:
        relative = row["Path"]
        path = ROOT / Path(relative)
        require(path.is_file(), f"Manifest entry is missing: {relative}")
        require(path.stat().st_size == int(row["Bytes"]), f"Size mismatch: {relative}")
        require(sha256(path) == row["SHA256"], f"SHA-256 mismatch: {relative}")
        listed.add(relative)

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path.relative_to(ROOT).as_posix() != MANIFEST.name
    }
    require(listed == actual, f"Manifest coverage mismatch: missing={actual-listed}, extra={listed-actual}")


def verify_github_size_limit() -> None:
    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 100_000_000
    ]
    require(not oversized, f"Files at or above GitHub's 100 MB limit: {oversized}")


def verify_markdown_links() -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or urlparse(target).scheme:
                continue
            resolved = (markdown.parent / unquote(target)).resolve()
            require(resolved.exists(), f"Broken Markdown link in {markdown.relative_to(ROOT)}: {target}")


def verify_no_local_paths() -> None:
    forbidden = ["C:\\Users\\", "/home/", "manuscript_revised.tex"]
    suffixes = {".md", ".py", ".json", ".cff", ".txt"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            require(token not in text, f"Local-only path/reference in {path.relative_to(ROOT)}: {token}")


def main() -> None:
    if sys.version_info < (3, 10):
        print("WARNING: release checks run here, but the full experiment code requires Python 3.10+.")
    verify_required_files()
    verify_restricted_data_absent()
    verify_headline_results()
    verify_compressed_predictions()
    verify_github_size_limit()
    verify_markdown_links()
    verify_no_local_paths()
    verify_manifest()
    print("PASS: public release structure, results, data boundaries, sizes, and hashes are valid.")


if __name__ == "__main__":
    main()
