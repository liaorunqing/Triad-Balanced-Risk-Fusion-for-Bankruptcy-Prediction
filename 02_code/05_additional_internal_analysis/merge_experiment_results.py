from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

METRICS = [
    "gmean",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "f05",
    "f2",
    "mcc",
    "triad_score",
    "balanced_accuracy",
    "ap",
    "auc",
    "feature_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge bankruptcy experiment CSV files into one statistical-test table.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "04_results" / "reproduction_check_merged_comparison.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for input_path in args.inputs:
        if not input_path.exists():
            print(f"Skipping missing file: {input_path}")
            continue
        df = pd.read_csv(input_path)
        for _, row in df.iterrows():
            rows.append(normalize_row(row, input_path.name))
    if not rows:
        raise ValueError("No rows found in input files.")
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["dataset", "seed", "method", "source"], keep="last")
    out = out.sort_values(["dataset", "seed", "method"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")
    print(out.groupby("method")[["gmean", "precision", "f2", "ap"]].mean().sort_values("f2", ascending=False).to_string())


def normalize_row(row: pd.Series, source: str) -> dict[str, object]:
    method = infer_method(row, source)
    normalized: dict[str, object] = {
        "dataset": row.get("dataset"),
        "seed": int(float(row.get("seed", 0))),
        "method": method,
        "category": row.get("category", infer_category(method)),
        "source": source,
    }
    for metric in METRICS:
        normalized[metric] = value_for(row, metric)
    if pd.isna(normalized["mcc"]):
        normalized["mcc"] = mcc_from_confusion(row)
    if pd.isna(normalized["triad_score"]):
        normalized["triad_score"] = triad_from_metrics(normalized)
    normalized["features"] = value_for(row, "features")
    if pd.isna(normalized["features"]):
        normalized["features"] = value_for(row, "feature_count")
    normalized["alpha"] = value_for(row, "blend_alpha")
    if pd.isna(normalized["alpha"]):
        normalized["alpha"] = value_for(row, "score_blend_alpha")
    return normalized


def infer_method(row: pd.Series, source: str) -> str:
    if "method" in row and not pd.isna(row["method"]):
        return str(row["method"])
    optimizer = str(row.get("optimizer", ""))
    risk_feature = str(row.get("risk_feature", "none"))
    focus = str(row.get("focus", "")).lower()
    source_lower = source.lower()
    if optimizer:
        if optimizer.upper() == "TIS_NGO":
            return "TIS_NGO-KELM"
        if optimizer.upper() == "PBMSBAINGO" and risk_feature == "histgb":
            if focus == "triad" or "triad" in source_lower:
                return "PBMSBAINGO-HGBKELM(Triad-mode)"
            if focus == "mcc_guarded" or "mcc_guarded" in source_lower:
                return "PBMSBAINGO-HGBKELM(MCC-guarded-focus)"
            if "mccfocus" in source_lower:
                return "PBMSBAINGO-HGBKELM(Sensitivity-mode)"
            return "PBMSBAINGO-HGBKELM"
        return f"{optimizer.upper()}-KELM"
    return source.removesuffix(".csv")


def infer_category(method: str) -> str:
    if "HGBKELM" in method:
        return "Proposed hybrid optimized KELM"
    if method.endswith("-KELM") or "KELM" in method:
        return "Optimizer-KELM"
    return "External ML baseline"


def value_for(row: pd.Series, metric: str) -> float:
    for key in [metric, f"test_{metric}"]:
        if key in row and not pd.isna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def mcc_from_confusion(row: pd.Series) -> float:
    values = {}
    for key in ["tp", "tn", "fp", "fn"]:
        value = value_for(row, key)
        if pd.isna(value):
            return float("nan")
        values[key] = value
    tp = values["tp"]
    tn = values["tn"]
    fp = values["fp"]
    fn = values["fn"]
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def triad_from_metrics(row: dict[str, object]) -> float:
    try:
        mcc = max(float(row["mcc"]), 0.0)
        sensitivity = float(row["sensitivity"])
        specificity = float(row["specificity"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return float(max(mcc * sensitivity * specificity, 0.0) ** (1.0 / 3.0))


if __name__ == "__main__":
    main()
