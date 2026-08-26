from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "03_data" / "01_raw" / "pone_global_2016"
DEFAULT_OUTPUT = ROOT / "03_data" / "02_processed" / "pone_global_2016"
SPECIFICATIONS = [
    ("S1File.xlsx", "Asia", 1),
    ("S2File.xlsx", "Asia", 2),
    ("S3File.xlsx", "Asia", 3),
    ("S5File.xlsx", "Europe", 1),
    ("S6File.xlsx", "Europe", 2),
    ("S7File.xlsx", "Europe", 3),
    ("S11File.xlsx", "North America", 1),
    ("S12File.xlsx", "North America", 2),
    ("S13File.xlsx", "North America", 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the regional long-form CSV used by the manuscript."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def populated_sheet(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
        if frame.dropna(how="all").shape[0] > 2:
            return frame
    raise ValueError(f"No populated worksheet found in {path}")


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def parse_european_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    text = normalize_text(value)
    if not text or text.upper() in {"@NA", "NA"}:
        return None
    try:
        number = float(text.replace(".", "").replace(",", "."))
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def key_number(value: float | None, missing_label: str) -> str:
    return missing_label if value is None else format(value, ".15g")


def build_rows(input_dir: Path) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for filename, region, horizon in SPECIFICATIONS:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Follow 03_data/PONE_DATA_INSTRUCTIONS.md first."
            )
        matrix = populated_sheet(path)
        data_rows = matrix.iloc[2:].dropna(how="all")
        seen: set[str] = set()
        for source_row, (_, row) in enumerate(data_rows.iterrows(), start=3):
            values = list(row) + [None] * max(0, 15 - len(row))
            financial = [parse_european_number(value) for value in values[3:13]]
            gics = parse_european_number(values[13])
            target = parse_european_number(values[14])
            model_key = "\u241f".join(
                [
                    normalize_text(values[0]),
                    *[
                        key_number(value, f"NA_{index}")
                        for index, value in enumerate(financial)
                    ],
                    key_number(gics, "NA_GICS"),
                    key_number(target, "NA_OUTCOME"),
                ]
            )
            duplicate = model_key in seen
            seen.add(model_key)
            output_rows.append(
                {
                    "source_file": filename,
                    "source_row": source_row,
                    "region": region,
                    "horizon": horizon,
                    "country": normalize_text(values[0]),
                    "inactive_marker": normalize_text(values[1]),
                    "inactive_marker_date_raw": normalize_text(values[2]),
                    **{f"V{index + 1}": value for index, value in enumerate(financial)},
                    "GICS": gics,
                    "target": target,
                    "duplicate_model_row_within_file": int(duplicate),
                }
            )
    return output_rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args.input_dir)
    frame = pd.DataFrame(rows)
    deduplicated = frame.loc[frame["duplicate_model_row_within_file"].eq(0)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "pone_regional_long_raw.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    deduplicated.to_csv(
        args.output_dir / "pone_regional_long_deduplicated.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    manifest = {
        "input_files": [item[0] for item in SPECIFICATIONS],
        "excluded_global_files": ["S8File.xlsx", "S9File.xlsx", "S10File.xlsx"],
        "raw_rows": int(len(frame)),
        "deduplicated_rows": int(len(deduplicated)),
        "duplicate_rows_removed": int(len(frame) - len(deduplicated)),
        "variable_definitions_source": "S4File.pdf",
        "provenance_source": "S14File.txt",
    }
    (args.output_dir / "cleaning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    expected = (1296, 1288, 8)
    observed = (
        manifest["raw_rows"],
        manifest["deduplicated_rows"],
        manifest["duplicate_rows_removed"],
    )
    if observed != expected:
        raise RuntimeError(f"Unexpected row audit: observed={observed}, expected={expected}")


if __name__ == "__main__":
    main()
