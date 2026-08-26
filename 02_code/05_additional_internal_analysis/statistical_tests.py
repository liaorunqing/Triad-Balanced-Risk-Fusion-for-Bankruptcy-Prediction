from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t as student_t, wilcoxon


DEFAULT_METRICS = [
    "gmean",
    "sensitivity",
    "specificity",
    "precision",
    "mcc",
    "triad_score",
    "f2",
    "ap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate descriptive summaries and dataset-level statistical comparisons. "
            "Repeated random splits are aggregated within each dataset by default."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "04_results" / "reproduction_check_stat_tests",
    )
    parser.add_argument("--control", default=None, help="Control method for Wilcoxon tests.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument(
        "--block-cols",
        nargs="+",
        default=["dataset"],
        help=(
            "Independent analysis units. The default is dataset. Do not add seed unless "
            "you intentionally want a descriptive repeated-split analysis."
        ),
    )
    parser.add_argument("--repeat-col", default="seed")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.30,
        help="Outer test fraction used by the Nadeau-Bengio corrected resampled t-test.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--method-col",
        default="method",
        help="Column containing algorithm/model names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = normalize_columns(df)

    block_cols = [c for c in args.block_cols if c in df.columns]
    if not block_cols:
        raise ValueError("No valid block columns found. Use --block-cols dataset seed or similar.")
    if args.method_col not in df.columns:
        raise ValueError(f"Method column not found: {args.method_col}")

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        raise ValueError("No requested metrics are present in the input file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.repeat_col in block_cols:
        print(
            "WARNING: the repeat column is part of the analysis unit. Repeated splits "
            "overlap and should not be interpreted as independent datasets."
        )
    summary = build_summary(df, args.method_col, metrics)
    dataset_summary = build_dataset_level_summary(df, args.method_col, metrics)
    ranks = build_average_ranks(df, args.method_col, block_cols, metrics)
    friedman = build_friedman(df, args.method_col, block_cols, metrics)
    nemenyi_cd, nemenyi_pairs = build_nemenyi(
        df,
        args.method_col,
        block_cols,
        metrics,
        args.alpha,
    )

    summary.to_csv(args.output_dir / "mean_std.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(
        args.output_dir / "dataset_level_mean_std.csv", index=False, encoding="utf-8-sig"
    )
    ranks.to_csv(args.output_dir / "average_ranks.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(args.output_dir / "friedman_tests.csv", index=False, encoding="utf-8-sig")
    nemenyi_cd.to_csv(args.output_dir / "nemenyi_cd.csv", index=False, encoding="utf-8-sig")
    nemenyi_pairs.to_csv(args.output_dir / "nemenyi_pairs.csv", index=False, encoding="utf-8-sig")

    if args.control:
        wilcox = build_wilcoxon_vs_control(
            df,
            args.method_col,
            block_cols,
            metrics,
            args.control,
            args.alpha,
        )
        wilcox.to_csv(args.output_dir / "wilcoxon_vs_control.csv", index=False, encoding="utf-8-sig")
        corrected = build_corrected_resampled_tests(
            df=df,
            method_col=args.method_col,
            metrics=metrics,
            control=args.control,
            repeat_col=args.repeat_col,
            test_fraction=args.test_fraction,
            alpha=args.alpha,
        )
        corrected.to_csv(
            args.output_dir / "corrected_resampled_tests.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        wilcox = pd.DataFrame()
        corrected = pd.DataFrame()

    print(f"Saved summary: {args.output_dir / 'mean_std.csv'}")
    print(f"Saved dataset-level summary: {args.output_dir / 'dataset_level_mean_std.csv'}")
    print(f"Saved ranks: {args.output_dir / 'average_ranks.csv'}")
    print(f"Saved Friedman tests: {args.output_dir / 'friedman_tests.csv'}")
    print(f"Saved Nemenyi CD: {args.output_dir / 'nemenyi_cd.csv'}")
    print(f"Saved Nemenyi pairs: {args.output_dir / 'nemenyi_pairs.csv'}")
    if args.control:
        print(f"Saved Wilcoxon tests: {args.output_dir / 'wilcoxon_vs_control.csv'}")
        print(
            "Saved corrected repeated-split tests: "
            f"{args.output_dir / 'corrected_resampled_tests.csv'}"
        )
    print_table_preview(summary, ranks, friedman, wilcox)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col.startswith("test_"):
            rename[col] = col.removeprefix("test_")
    df = df.rename(columns=rename)
    if "optimizer" in df.columns and "method" not in df.columns:
        df["method"] = df["optimizer"]
    if "seed" not in df.columns:
        df["seed"] = 0
    return df


def build_summary(df: pd.DataFrame, method_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby(method_col):
        row: dict[str, float | str | int] = {
            "method": method,
            "n_rows": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean_std"] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f2_mean", ascending=False, na_position="last")


def build_dataset_level_summary(
    df: pd.DataFrame,
    method_col: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Summarize variability across datasets after averaging repeated splits.

    The five Polish horizons are treated as the analysis units. The standard
    deviation therefore describes cross-dataset heterogeneity, not the
    uncertainty of 50 independent observations.
    """
    if "dataset" not in df.columns:
        return pd.DataFrame()
    dataset_means = (
        df.groupby(["dataset", method_col], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    return build_summary(dataset_means, method_col, metrics)


def build_average_ranks(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pivot = paired_pivot(df, method_col, block_cols, metric)
        if pivot.empty:
            continue
        rank_rows = []
        for _, row in pivot.iterrows():
            values = row.to_numpy(dtype=float)
            ranks = rankdata(-values, method="average")
            rank_rows.append(ranks)
        rank_array = np.vstack(rank_rows)
        for method, avg_rank in zip(pivot.columns, np.mean(rank_array, axis=0)):
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "average_rank": float(avg_rank),
                    "n_blocks": int(len(pivot)),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "average_rank"])


def build_friedman(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pivot = paired_pivot(df, method_col, block_cols, metric)
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            rows.append(
                {
                    "metric": metric,
                    "n_blocks": int(pivot.shape[0]),
                    "n_methods": int(pivot.shape[1]),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "note": "Need at least 2 complete blocks and 3 methods.",
                }
            )
            continue
        stat, p_value = friedmanchisquare(*[pivot[col].to_numpy(dtype=float) for col in pivot.columns])
        rows.append(
            {
                "metric": metric,
                "n_blocks": int(pivot.shape[0]),
                "n_methods": int(pivot.shape[1]),
                "statistic": float(stat),
                "p_value": float(p_value),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def build_wilcoxon_vs_control(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metrics: list[str],
    control: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pivot = paired_pivot(df, method_col, block_cols, metric)
        if control not in pivot.columns:
            rows.append(
                {
                    "metric": metric,
                    "control": control,
                    "method": "",
                    "n_pairs": 0,
                    "control_mean": np.nan,
                    "method_mean": np.nan,
                    "mean_diff_control_minus_method": np.nan,
                    "statistic": np.nan,
                    "p_value_two_sided": np.nan,
                    "p_value_control_greater": np.nan,
                    "note": "Control method missing.",
                }
            )
            continue
        control_values = pivot[control].to_numpy(dtype=float)
        for method in pivot.columns:
            if method == control:
                continue
            method_values = pivot[method].to_numpy(dtype=float)
            diff = control_values - method_values
            n_pairs = int(np.sum(~np.isnan(diff)))
            wins = int(np.sum(diff > 0))
            losses = int(np.sum(diff < 0))
            ties = int(np.sum(np.isclose(diff, 0.0)))
            rank_biserial = matched_pairs_rank_biserial(diff)
            if n_pairs < 2 or np.allclose(diff, 0.0):
                stat_two = np.nan
                p_two = np.nan
                p_greater = np.nan
                note = "Need non-identical paired values."
            else:
                stat_two, p_two = wilcoxon(control_values, method_values, alternative="two-sided")
                _, p_greater = wilcoxon(control_values, method_values, alternative="greater")
                note = ""
            rows.append(
                {
                    "metric": metric,
                    "control": control,
                    "method": method,
                    "n_pairs": n_pairs,
                    "control_mean": float(np.mean(control_values)),
                    "method_mean": float(np.mean(method_values)),
                    "mean_diff_control_minus_method": float(np.mean(diff)),
                    "median_diff_control_minus_method": float(np.median(diff)),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "matched_pairs_rank_biserial": rank_biserial,
                    "statistic": float(stat_two) if not is_nan(stat_two) else np.nan,
                    "p_value_two_sided": float(p_two) if not is_nan(p_two) else np.nan,
                    "p_value_control_greater": float(p_greater) if not is_nan(p_greater) else np.nan,
                    "note": note,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    adjusted = []
    for _, group in out.groupby("metric", sort=False):
        group = group.copy()
        group["holm_p_two_sided"] = holm_adjusted_p_values(group["p_value_two_sided"].to_numpy(dtype=float))
        group["holm_reject_two_sided"] = group["holm_p_two_sided"] <= alpha
        group["holm_p_control_greater"] = holm_adjusted_p_values(
            group["p_value_control_greater"].to_numpy(dtype=float)
        )
        group["holm_reject_control_greater"] = group["holm_p_control_greater"] <= alpha
        adjusted.append(group)
    return pd.concat(adjusted, ignore_index=True).sort_values(
        ["metric", "p_value_control_greater"],
        na_position="last",
    )


def matched_pairs_rank_biserial(diff: np.ndarray) -> float:
    """Rank-biserial effect for paired differences, excluding exact ties."""
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if not len(values):
        return np.nan
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total else np.nan


def build_corrected_resampled_tests(
    df: pd.DataFrame,
    method_col: str,
    metrics: list[str],
    control: str,
    repeat_col: str,
    test_fraction: float,
    alpha: float,
) -> pd.DataFrame:
    """Nadeau-Bengio corrected paired t-tests within each dataset.

    This correction inflates the standard error to reflect overlap among
    repeated random train/test splits. Results remain secondary because only
    one underlying corpus is represented by the five horizons.
    """
    if "dataset" not in df.columns or repeat_col not in df.columns:
        return pd.DataFrame()
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must lie between 0 and 1.")

    correction_ratio = test_fraction / (1.0 - test_fraction)
    rows = []
    for metric in metrics:
        for dataset, group in df.groupby("dataset"):
            pivot = group.pivot_table(
                index=repeat_col,
                columns=method_col,
                values=metric,
                aggfunc="mean",
            )
            if control not in pivot.columns:
                continue
            for method in pivot.columns:
                if method == control:
                    continue
                paired = pivot[[control, method]].dropna()
                diff = paired[control].to_numpy(dtype=float) - paired[method].to_numpy(dtype=float)
                n = len(diff)
                mean_diff = float(np.mean(diff)) if n else np.nan
                sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
                if n > 1 and sd_diff > 0:
                    corrected_se = float(
                        np.sqrt((1.0 / n + correction_ratio) * sd_diff**2)
                    )
                    statistic = mean_diff / corrected_se
                    p_two = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
                    p_greater = float(student_t.sf(statistic, df=n - 1))
                    note = ""
                else:
                    corrected_se = np.nan
                    statistic = np.nan
                    p_two = np.nan
                    p_greater = np.nan
                    note = "Need at least two non-identical paired repeats."
                rows.append(
                    {
                        "metric": metric,
                        "dataset": dataset,
                        "control": control,
                        "method": method,
                        "n_repeats": n,
                        "test_fraction": test_fraction,
                        "train_fraction": 1.0 - test_fraction,
                        "mean_diff_control_minus_method": mean_diff,
                        "sd_paired_diff": sd_diff,
                        "corrected_se": corrected_se,
                        "statistic": statistic,
                        "p_value_two_sided": p_two,
                        "p_value_control_greater": p_greater,
                        "note": note,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    adjusted = []
    for _, group in out.groupby(["metric", "dataset"], sort=False):
        group = group.copy()
        group["holm_p_two_sided"] = holm_adjusted_p_values(
            group["p_value_two_sided"].to_numpy(dtype=float)
        )
        group["holm_reject_two_sided"] = group["holm_p_two_sided"] <= alpha
        group["holm_p_control_greater"] = holm_adjusted_p_values(
            group["p_value_control_greater"].to_numpy(dtype=float)
        )
        group["holm_reject_control_greater"] = (
            group["holm_p_control_greater"] <= alpha
        )
        adjusted.append(group)
    return pd.concat(adjusted, ignore_index=True).sort_values(
        ["metric", "dataset", "p_value_control_greater"],
        na_position="last",
    )


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(p_values))[0]
    m = len(valid_idx)
    if m == 0:
        return adjusted
    order = valid_idx[np.argsort(p_values[valid_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * float(p_values[idx]))
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def build_nemenyi(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metrics: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cd_rows = []
    pair_rows = []
    for metric in metrics:
        rank_map, n_blocks = average_rank_map(df, method_col, block_cols, metric)
        methods = list(rank_map.keys())
        n_methods = len(methods)
        if n_blocks < 2 or n_methods < 2:
            cd_rows.append(
                {
                    "metric": metric,
                    "n_blocks": n_blocks,
                    "n_methods": n_methods,
                    "alpha": alpha,
                    "q_alpha": np.nan,
                    "critical_difference": np.nan,
                    "note": "Need at least 2 complete blocks and 2 methods.",
                }
            )
            continue
        q_alpha = float(studentized_range.isf(alpha, n_methods, np.inf) / math.sqrt(2.0))
        critical_difference = q_alpha * math.sqrt(
            n_methods * (n_methods + 1) / (6.0 * n_blocks)
        )
        cd_rows.append(
            {
                "metric": metric,
                "n_blocks": n_blocks,
                "n_methods": n_methods,
                "alpha": alpha,
                "q_alpha": q_alpha,
                "critical_difference": critical_difference,
                "note": "",
            }
        )
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1 :]:
                rank_a = rank_map[method_a]
                rank_b = rank_map[method_b]
                rank_diff = abs(rank_a - rank_b)
                pair_rows.append(
                    {
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "rank_a": rank_a,
                        "rank_b": rank_b,
                        "rank_diff": rank_diff,
                        "critical_difference": critical_difference,
                        "significant": bool(rank_diff > critical_difference),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs = pairs.sort_values(["metric", "rank_diff"], ascending=[True, False])
    return pd.DataFrame(cd_rows), pairs


def average_rank_map(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metric: str,
) -> tuple[dict[str, float], int]:
    pivot = paired_pivot(df, method_col, block_cols, metric)
    if pivot.empty:
        return {}, 0
    rank_rows = []
    for _, row in pivot.iterrows():
        values = row.to_numpy(dtype=float)
        rank_rows.append(rankdata(-values, method="average"))
    avg_ranks = np.mean(np.vstack(rank_rows), axis=0)
    rank_map = {str(method): float(rank) for method, rank in zip(pivot.columns, avg_ranks)}
    return dict(sorted(rank_map.items(), key=lambda item: item[1])), int(len(pivot))


def paired_pivot(
    df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    temp = df[block_cols + [method_col, metric]].copy()
    temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.dropna(subset=[metric])
    pivot = temp.pivot_table(
        index=block_cols,
        columns=method_col,
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def format_mean_std(mean: float, std: float) -> str:
    if is_nan(mean):
        return ""
    return f"{mean:.4f} +/- {std:.4f}"


def is_nan(value: object) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def print_table_preview(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    friedman: pd.DataFrame,
    wilcox: pd.DataFrame,
) -> None:
    print("\nMean/std preview:")
    preview_cols = [c for c in ["method", "gmean_mean_std", "precision_mean_std", "f2_mean_std", "ap_mean_std"] if c in summary]
    print(summary[preview_cols].to_string(index=False))
    print("\nAverage ranks preview:")
    print(ranks[ranks["metric"].isin(["f2", "ap"])].to_string(index=False))
    print("\nFriedman preview:")
    print(friedman.to_string(index=False))
    if len(wilcox):
        print("\nWilcoxon preview:")
        print(wilcox[wilcox["metric"].isin(["f2", "ap"])].to_string(index=False))


if __name__ == "__main__":
    main()
