from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "04_results" / "02_pone_cross_region"
PRIMARY = RESULTS / "raw_seed_runs" / "seed_20260727" / "metrics.csv"
RAW = (
    RESULTS
    / "sensitivity"
    / "matched_partition_seed_20260727"
    / "metrics.csv"
)
OUTPUT = RESULTS / "consolidated"


def main() -> None:
    primary = pd.read_csv(PRIMARY)
    raw = pd.read_csv(RAW)
    keys = ["dataset", "method", "policy"]
    metrics = ["auc", "ap", "mcc", "balanced_accuracy", "brier", "log_loss"]
    primary = primary.loc[primary["policy"] == "triad", keys + metrics]
    raw = raw.loc[raw["policy"] == "triad", keys + metrics]
    comparison = primary.merge(raw, on=keys, suffixes=("_deduplicated", "_raw"))
    for metric in metrics:
        comparison[f"delta_raw_minus_deduplicated_{metric}"] = (
            comparison[f"{metric}_raw"] - comparison[f"{metric}_deduplicated"]
        )
    delta_columns = [column for column in comparison if column.startswith("delta_")]
    summary_rows = []
    for method, group in comparison.groupby("method"):
        for column in delta_columns:
            metric = column.rsplit("_", 1)[-1]
            if metric == "accuracy":
                metric = "balanced_accuracy"
            values = group[column]
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean_delta_raw_minus_deduplicated": values.mean(),
                    "mean_absolute_delta": values.abs().mean(),
                    "max_absolute_delta": values.abs().max(),
                    "tasks": len(values),
                }
            )
    comparison.to_csv(
        OUTPUT / "raw_vs_deduplicated_task_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT / "raw_vs_deduplicated_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT / "raw_sensitivity_payload.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(orient="records"),
                "comparison": comparison.to_dict(orient="records"),
                "interpretation": (
                    "Matched-partition raw-record sensitivity. Common observations retain "
                    "their primary fit/search/calibration assignment; duplicate rows follow "
                    "the original row. Balanced logistic remains the strongest MCC reference, "
                    "and Full PBMSBAINGO does not become universally superior."
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary.loc[summary["metric"].isin(["auc", "ap", "mcc", "balanced_accuracy"])]
        .sort_values(["metric", "method"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
