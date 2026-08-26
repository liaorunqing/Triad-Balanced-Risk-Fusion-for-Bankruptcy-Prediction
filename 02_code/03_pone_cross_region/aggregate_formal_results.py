from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "04_results" / "02_pone_cross_region" / "raw_seed_runs"
SEEDS = [20260727, 20260728, 20260729, 20260730, 20260731]
METRICS = [
    "auc",
    "ap",
    "mcc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "brier",
    "log_loss",
]
HIGHER_IS_BETTER = {
    "auc": True,
    "ap": True,
    "mcc": True,
    "balanced_accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "brier": False,
    "log_loss": False,
}
METHOD_ORDER = [
    "Balanced_logistic",
    "HistGB",
    "KELM_PBMSBAINGO",
    "KELM_score_fusion",
    "KELM_risk_feature",
    "Full_PBMSBAINGO",
    "Full_random_search",
    "Uncertainty_gated_fusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "04_results"
            / "02_pone_cross_region"
            / "reaggregated"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int):
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    prediction_frames = []
    selection_frames = []
    for seed in SEEDS:
        directory = BASE / f"seed_{seed}"
        metric_frames.append(pd.read_csv(directory / "metrics.csv"))
        prediction_frames.append(pd.read_csv(directory / "predictions.csv"))
        selection_frames.append(pd.read_csv(directory / "selection.csv"))
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        args.output_dir / "all_predictions.csv", index=False, encoding="utf-8-sig"
    )
    selections.to_csv(
        args.output_dir / "all_selections.csv", index=False, encoding="utf-8-sig"
    )

    triad = metrics.loc[metrics["policy"] == "triad"].copy()
    task_seed = triad[
        ["dataset", "horizon", "target_region", "seed", "method", *METRICS]
    ].copy()
    task_seed.to_csv(
        args.output_dir / "triad_task_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    task_means = (
        task_seed.groupby(
            ["dataset", "horizon", "target_region", "method"], as_index=False
        )[METRICS]
        .mean()
    )
    task_means.to_csv(
        args.output_dir / "triad_task_means_across_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(20260727)
    summary_rows = []
    for method in METHOD_ORDER:
        method_tasks = task_means.loc[task_means["method"] == method]
        for metric in METRICS:
            values = method_tasks[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, rng, args.bootstrap)
            seed_stds = (
                task_seed.loc[task_seed["method"] == method]
                .groupby("dataset")[metric]
                .std(ddof=1)
                .fillna(0.0)
            )
            wide = task_means.pivot(index="dataset", columns="method", values=metric)
            best = wide.max(axis=1) if HIGHER_IS_BETTER[metric] else wide.min(axis=1)
            wins = int(
                np.isclose(wide[method].to_numpy(), best.to_numpy(), atol=1e-12).sum()
            )
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "task_count": len(values),
                    "mean": values.mean(),
                    "task_sd": values.std(ddof=1),
                    "task_bootstrap_ci_low": low,
                    "task_bootstrap_ci_high": high,
                    "mean_within_task_seed_sd": seed_stds.mean(),
                    "task_wins_including_ties": wins,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.output_dir / "method_summary.csv", index=False, encoding="utf-8-sig"
    )

    friedman_rows = []
    for metric in METRICS:
        wide = task_means.pivot(index="dataset", columns="method", values=metric)[
            METHOD_ORDER
        ]
        statistic, p_value = friedmanchisquare(
            *(wide[method].to_numpy() for method in METHOD_ORDER)
        )
        ranks = wide.rank(
            axis=1,
            ascending=not HIGHER_IS_BETTER[metric],
            method="average",
        ).mean(axis=0)
        friedman_rows.append(
            {
                "metric": metric,
                "blocks": len(wide),
                "methods": len(METHOD_ORDER),
                "statistic": statistic,
                "p_value": p_value,
                "average_ranks_json": json.dumps(ranks.to_dict()),
            }
        )
    pd.DataFrame(friedman_rows).to_csv(
        args.output_dir / "friedman_tests.csv", index=False, encoding="utf-8-sig"
    )

    comparisons = [
        ("Uncertainty_gated_fusion", "Balanced_logistic"),
        ("Uncertainty_gated_fusion", "HistGB"),
        ("Uncertainty_gated_fusion", "Full_PBMSBAINGO"),
        ("Full_PBMSBAINGO", "Balanced_logistic"),
        ("Full_PBMSBAINGO", "HistGB"),
        ("Full_PBMSBAINGO", "Full_random_search"),
        ("KELM_risk_feature", "Balanced_logistic"),
        ("KELM_PBMSBAINGO", "Balanced_logistic"),
    ]
    paired_rows = []
    for metric in METRICS:
        wide = task_means.pivot(index="dataset", columns="method", values=metric)
        metric_rows = []
        for left, right in comparisons:
            difference = wide[left].to_numpy() - wide[right].to_numpy()
            if not HIGHER_IS_BETTER[metric]:
                advantage = -difference
            else:
                advantage = difference
            if np.allclose(advantage, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    advantage,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
            boot = rng.choice(
                advantage, size=(args.bootstrap, len(advantage)), replace=True
            ).mean(axis=1)
            metric_rows.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "blocks": len(advantage),
                    "mean_advantage_positive_favors_left": advantage.mean(),
                    "median_advantage_positive_favors_left": np.median(advantage),
                    "advantage_ci_low": np.quantile(boot, 0.025),
                    "advantage_ci_high": np.quantile(boot, 0.975),
                    "left_better_blocks": int(np.sum(advantage > 0)),
                    "ties": int(np.sum(np.isclose(advantage, 0.0))),
                    "right_better_blocks": int(np.sum(advantage < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_p_within_metric"] = adjusted_p
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        args.output_dir / "paired_task_tests.csv", index=False, encoding="utf-8-sig"
    )

    expected_metric_rows = len(SEEDS) * 9 * len(METHOD_ORDER) * 6
    expected_prediction_methods = len(SEEDS) * 9 * len(METHOD_ORDER)
    checks = {
        "seeds": SEEDS,
        "metric_rows": len(metrics),
        "expected_metric_rows": expected_metric_rows,
        "triad_rows": len(triad),
        "expected_triad_rows": len(SEEDS) * 9 * len(METHOD_ORDER),
        "prediction_method_blocks": predictions[
            ["dataset", "seed", "method"]
        ].drop_duplicates().shape[0],
        "expected_prediction_method_blocks": expected_prediction_methods,
        "selection_rows": len(selections),
        "duplicate_prediction_keys": int(
            predictions.duplicated(["dataset", "seed", "method", "row_id"]).sum()
        ),
        "missing_metric_values": int(metrics[METRICS].isna().sum().sum()),
    }
    (args.output_dir / "quality_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    workbook_payload = {
        "method_summary": summary.to_dict(orient="records"),
        "task_means": task_means.to_dict(orient="records"),
        "paired_tests": paired.to_dict(orient="records"),
        "friedman_tests": pd.DataFrame(friedman_rows).to_dict(orient="records"),
        "quality_checks": checks,
    }
    (args.output_dir / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
