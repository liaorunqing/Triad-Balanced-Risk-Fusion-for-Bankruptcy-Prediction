from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "04_results" / "03_manuscript_tables"
FAIR_FILE = REVISION / "fair_internal_baselines.csv"
TRIAD_FILE = (
    ROOT
    / "04_results"
    / "01_polish_internal"
    / "source_tables"
    / "formal_pbmsbaingo_hgbkelm_triad_alpha005_10seed_complete.csv"
)
CROSS_FILE = (
    ROOT
    / "04_results"
    / "02_pone_cross_region"
    / "consolidated"
    / "all_metrics.csv"
)
METRICS = ["mcc", "sensitivity", "specificity", "precision", "f2", "ap"]


def exact_wilcoxon(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if np.allclose(difference, 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        left,
        right,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    return float(result.statistic), float(result.pvalue)


def prepare_internal() -> pd.DataFrame:
    fair = pd.read_csv(FAIR_FILE)
    fair = fair[["dataset", "seed", "method", *METRICS, "tp", "fp", "tn", "fn"]]
    triad_raw = pd.read_csv(TRIAD_FILE)
    triad = pd.DataFrame(
        {
            "dataset": triad_raw["dataset"],
            "seed": triad_raw["seed"].astype(int),
            "method": "Triad_risk_fusion",
            **{metric: triad_raw[f"test_{metric}"] for metric in METRICS},
            "tp": triad_raw["test_tp"],
            "fp": triad_raw["test_fp"],
            "tn": triad_raw["test_tn"],
            "fn": triad_raw["test_fn"],
        }
    )
    combined = pd.concat([fair, triad], ignore_index=True)
    combined.to_csv(REVISION / "internal_revised_blocks.csv", index=False)
    return combined


def summarize_internal(frame: pd.DataFrame) -> None:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        for metric in METRICS:
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "blocks": len(group),
                    "mean": group[metric].mean(),
                    "sd": group[metric].std(ddof=1),
                }
            )
    pd.DataFrame(rows).to_csv(REVISION / "internal_revised_summary.csv", index=False)

    dataset_means = (
        frame.groupby(["dataset", "method"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "method"])
    )
    dataset_means.to_csv(REVISION / "internal_revised_dataset_means.csv", index=False)

    selected = dataset_means.loc[
        dataset_means["method"].isin(["Triad_risk_fusion", "HistGB_fair"])
    ]
    wide = selected.pivot(index="dataset", columns="method", values=METRICS)
    comparison_rows = []
    pair_rows = []
    for metric in METRICS:
        triad = wide[(metric, "Triad_risk_fusion")].to_numpy()
        histgb = wide[(metric, "HistGB_fair")].to_numpy()
        statistic, p_value = exact_wilcoxon(triad, histgb)
        difference = triad - histgb
        comparison_rows.append(
            {
                "metric": metric,
                "mean_difference_triad_minus_histgb": difference.mean(),
                "wins": int(np.sum(difference > 0)),
                "losses": int(np.sum(difference < 0)),
                "ties": int(np.sum(np.isclose(difference, 0.0))),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            }
        )
        for dataset, triad_value, histgb_value, diff in zip(
            wide.index, triad, histgb, difference
        ):
            pair_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "triad": triad_value,
                    "histgb_fair": histgb_value,
                    "difference": diff,
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        REVISION / "triad_vs_fair_histgb_tests.csv", index=False
    )
    pd.DataFrame(pair_rows).to_csv(
        REVISION / "triad_vs_fair_histgb_dataset_pairs.csv", index=False
    )

    methods = list(dataset_means["method"].unique())
    friedman_rows = []
    for metric in METRICS:
        pivot = dataset_means.pivot(index="dataset", columns="method", values=metric)
        result = friedmanchisquare(*[pivot[method].to_numpy() for method in methods])
        ranks = pivot.rank(axis=1, ascending=False, method="average").mean(axis=0)
        friedman_rows.append(
            {
                "metric": metric,
                "datasets": len(pivot),
                "methods": len(methods),
                "statistic": result.statistic,
                "p_value": result.pvalue,
                "average_ranks_json": json.dumps(ranks.to_dict()),
            }
        )
    pd.DataFrame(friedman_rows).to_csv(
        REVISION / "internal_revised_friedman.csv", index=False
    )


def fixed_operating_cost(frame: pd.DataFrame) -> None:
    rows = []
    for _, row in frame.iterrows():
        for ratio in [1, 2, 5, 10, 20, 50]:
            numerator = ratio * row["fn"] + row["fp"]
            denominator = ratio * (row["tp"] + row["fn"]) + row["tn"] + row["fp"]
            rows.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "method": row["method"],
                    "cost_ratio": ratio,
                    "normalized_cost": numerator / denominator,
                }
            )
    costs = pd.DataFrame(rows)
    costs.to_csv(REVISION / "fixed_operating_cost_blocks.csv", index=False)
    summary = (
        costs.groupby(["method", "cost_ratio"], as_index=False)["normalized_cost"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(REVISION / "fixed_operating_cost_summary.csv", index=False)


def summarize_cross_region() -> None:
    cross = pd.read_csv(CROSS_FILE)
    triad_policy = cross.loc[cross["policy"] == "triad"].copy()
    metrics = [
        "auc",
        "ap",
        "mcc",
        "balanced_accuracy",
        "brier",
        "log_loss",
        "ece_10",
        "calibration_intercept",
        "calibration_slope",
    ]
    task_seed = (
        triad_policy.groupby(
            ["horizon", "target_region", "method", "seed"], as_index=False
        )[metrics]
        .mean()
    )
    task_means = (
        task_seed.groupby(["horizon", "target_region", "method"], as_index=False)[
            metrics
        ]
        .mean()
    )
    task_means["absolute_intercept"] = task_means["calibration_intercept"].abs()
    task_means["absolute_slope_error"] = (
        task_means["calibration_slope"] - 1.0
    ).abs()
    task_means.to_csv(REVISION / "cross_region_extended_task_means.csv", index=False)

    summary_metrics = [*metrics, "absolute_intercept", "absolute_slope_error"]
    summary_rows = []
    for method, group in task_means.groupby("method", sort=False):
        for metric in summary_metrics:
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "tasks": len(group),
                    "mean": group[metric].mean(),
                    "sd": group[metric].std(ddof=1),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        REVISION / "cross_region_extended_summary.csv", index=False
    )

    full = task_means.loc[task_means["method"] == "Full_PBMSBAINGO"].set_index(
        ["horizon", "target_region"]
    )
    comparison_rows = []
    for comparator in ["Balanced_logistic", "HistGB", "Full_random_search"]:
        other = task_means.loc[task_means["method"] == comparator].set_index(
            ["horizon", "target_region"]
        )
        common = full.index.intersection(other.index)
        for metric in [
            "brier",
            "log_loss",
            "ece_10",
            "absolute_intercept",
            "absolute_slope_error",
        ]:
            statistic, p_value = exact_wilcoxon(
                full.loc[common, metric].to_numpy(),
                other.loc[common, metric].to_numpy(),
            )
            difference = (
                full.loc[common, metric].to_numpy()
                - other.loc[common, metric].to_numpy()
            )
            comparison_rows.append(
                {
                    "metric": metric,
                    "comparator": comparator,
                    "mean_difference_full_minus_comparator": difference.mean(),
                    "full_lower_tasks": int(np.sum(difference < 0)),
                    "full_higher_tasks": int(np.sum(difference > 0)),
                    "wilcoxon_statistic": statistic,
                    "raw_p": p_value,
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        REVISION / "cross_region_calibration_pairwise.csv", index=False
    )


def main() -> None:
    REVISION.mkdir(parents=True, exist_ok=True)
    internal = prepare_internal()
    summarize_internal(internal)
    fixed_operating_cost(internal)
    summarize_cross_region()
    print(f"Saved revised analyses to {REVISION}")


if __name__ == "__main__":
    main()
