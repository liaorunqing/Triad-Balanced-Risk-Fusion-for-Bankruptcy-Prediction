from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METHOD_ORDER = [
    "NGO-KELM",
    "INGO-KELM",
    "TIS_NGO-KELM",
    "BAINGO-KELM",
    "MSBAINGO-KELM",
    "PBMSBAINGO-KELM",
    "HistGB",
    "PBMSBAINGO-HGBKELM",
]

COMPONENT_LABELS = {
    "NGO-KELM": "Base NGO optimizer + KELM",
    "INGO-KELM": "Improved NGO optimizer + KELM",
    "TIS_NGO-KELM": "Related TIS_NGO-KELM baseline",
    "BAINGO-KELM": "Balanced adaptive NGO + KELM",
    "MSBAINGO-KELM": "Multi-strategy BAINGO + KELM",
    "PBMSBAINGO-KELM": "Precision-balanced MSBAINGO + KELM",
    "HistGB": "Risk learner only",
    "PBMSBAINGO-HGBKELM": "Full risk-fusion optimized KELM",
    "PBMSBAINGO-HGBKELM(Triad-mode)": "Triad-balanced full risk-fusion optimized KELM",
    "PBMSBAINGO-HGBKELM(MCC-guarded)": "MCC-guarded full risk-fusion optimized KELM",
}

METRICS = ["gmean", "sensitivity", "specificity", "precision", "mcc", "triad_score", "f2", "ap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation summary table for bankruptcy experiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "04_results" / "reproduction_check_ablation_summary.csv",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHOD_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "method" not in df.columns:
        raise ValueError("Input table must contain a method column.")

    df = df[df["method"].isin(args.methods)].copy()
    if df.empty:
        raise ValueError("No requested ablation methods were found.")

    rows = []
    for method in args.methods:
        group = df[df["method"] == method]
        if group.empty:
            continue
        row: dict[str, object] = {
            "method": method,
            "component": COMPONENT_LABELS.get(method, method),
            "n_rows": int(len(group)),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else np.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = "" if np.isnan(mean) else f"{mean:.4f} +/- {std:.4f}"
        if "features" in group.columns:
            features = pd.to_numeric(group["features"], errors="coerce").dropna()
            row["features_mean"] = float(features.mean()) if len(features) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["f2_gain_vs_base"] = out["f2_mean"] - float(out.loc[out["method"] == out.iloc[0]["method"], "f2_mean"].iloc[0])
    out["ap_gain_vs_base"] = out["ap_mean"] - float(out.loc[out["method"] == out.iloc[0]["method"], "ap_mean"].iloc[0])
    out["precision_gain_vs_base"] = out["precision_mean"] - float(
        out.loc[out["method"] == out.iloc[0]["method"], "precision_mean"].iloc[0]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    preview_cols = [
        "method",
        "component",
        "gmean_mean_std",
        "precision_mean_std",
        "mcc_mean_std",
        "f2_mean_std",
        "ap_mean_std",
        "f2_gain_vs_base",
    ]
    print(f"Saved: {args.output}")
    print(out[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
