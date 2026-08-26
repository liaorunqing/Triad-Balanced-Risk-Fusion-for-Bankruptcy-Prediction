from __future__ import annotations

import argparse
import ast
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RATIOS = [1, 2, 5, 10, 20, 50]
DEFAULT_METHODS = [
    "PBMSBAINGO-HGBKELM(Triad-mode)",
    "PBMSBAINGO-HGBKELM(MCC-guarded)",
    "PBMSBAINGO-HGBKELM(F2-mode)",
    "PBMSBAINGO-HGBKELM(Sensitivity-mode)",
    "HistGB",
    "EasyEnsemble",
    "BalancedRF",
    "TIS_NGO-KELM",
]

ATTRIBUTE_NAMES = {
    5: "[(cash + short-term securities + receivables - short-term liabilities) / "
       "(operating expenses - depreciation)] * 365",
    6: "retained earnings / total assets",
    11: "(gross profit + extraordinary items + financial expenses) / total assets",
    13: "(gross profit + depreciation) / sales",
    15: "(total liabilities × 365) / (gross profit + depreciation)",
    18: "gross profit / total assets",
    19: "gross profit / sales",
    21: "sales(n) / sales(n-1)",
    24: "gross profit (3 years) / total assets",
    26: "(net profit + depreciation) / total liabilities",
    27: "profit on operating activities / financial expenses",
    39: "profit on sales / sales",
    41: "total liabilities / operating cash-flow proxy",
    42: "profit on operating activities / sales",
    65: "out-of-fold HistGB risk feature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate relative decision cost and feature-selection stability."
    )
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--triad-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(args.comparison)
    counts = pd.read_csv(args.counts)
    triad_raw = pd.read_csv(args.triad_raw)

    enriched = attach_confusion_counts(comparison, counts)
    profiles = build_cost_profiles(enriched, args.ratios)
    selected_profiles = profiles[profiles["method"].isin(args.methods)].copy()
    summary = summarize_costs(selected_profiles)
    dataset_summary = summarize_costs_by_dataset(selected_profiles)
    wins = count_cost_wins(selected_profiles)
    pareto = build_fpr_fnr_summary(enriched[enriched["method"].isin(args.methods)])

    feature_frequency, stability = feature_stability(triad_raw)

    profiles.to_csv(args.output_dir / "decision_cost_all_rows.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "decision_cost_summary.csv", index=False, encoding="utf-8-sig")
    dataset_summary.to_csv(
        args.output_dir / "decision_cost_by_dataset.csv", index=False, encoding="utf-8-sig"
    )
    wins.to_csv(args.output_dir / "decision_cost_block_wins.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(args.output_dir / "fpr_fnr_summary.csv", index=False, encoding="utf-8-sig")
    feature_frequency.to_csv(
        args.output_dir / "feature_selection_frequency.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        args.output_dir / "feature_selection_stability.csv", index=False, encoding="utf-8-sig"
    )

    plot_cost_profiles(summary, args.output_dir / "decision_cost_profile.png")
    plot_fpr_fnr(pareto, args.output_dir / "fpr_fnr_operating_points.png")
    plot_feature_frequency(
        feature_frequency, args.output_dir / "feature_selection_frequency.png"
    )

    print(summary.to_string(index=False))
    print("\nFeature stability:")
    print(stability.to_string(index=False))


def attach_confusion_counts(comparison: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "seed", "tp", "fp", "tn", "fn"}
    missing = required.difference(counts.columns)
    if missing:
        raise ValueError(f"Counts file lacks columns: {sorted(missing)}")

    block_counts = counts.groupby(["dataset", "seed"], as_index=False).agg(
        n_positive=("tp", lambda s: float(s.iloc[0])),
        n_negative=("tn", lambda s: float(s.iloc[0])),
        count_rows=("method", "size"),
    )
    # The first aggregation above contains TP/TN, not class totals. Reconstruct class
    # totals from one baseline row per split, after verifying all methods agree.
    check = counts.copy()
    check["n_positive"] = check["tp"] + check["fn"]
    check["n_negative"] = check["tn"] + check["fp"]
    nunique = check.groupby(["dataset", "seed"])[["n_positive", "n_negative"]].nunique()
    if (nunique > 1).any().any():
        raise ValueError("Inconsistent test-set class counts across methods.")
    block_counts = (
        check.groupby(["dataset", "seed"], as_index=False)[["n_positive", "n_negative"]]
        .first()
        .astype({"n_positive": float, "n_negative": float})
    )

    merged = comparison.merge(block_counts, on=["dataset", "seed"], how="left", validate="many_to_one")
    if merged[["n_positive", "n_negative"]].isna().any().any():
        raise ValueError("Could not match class counts for all comparison rows.")

    merged["fn"] = (1.0 - merged["sensitivity"]) * merged["n_positive"]
    merged["fp"] = (1.0 - merged["specificity"]) * merged["n_negative"]
    return merged


def build_cost_profiles(df: pd.DataFrame, ratios: list[float]) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError("Cost ratios must be positive.")
        part = df.copy()
        part["cost_ratio_fn_to_fp"] = float(ratio)
        part["normalized_cost"] = (
            float(ratio) * part["fn"] + part["fp"]
        ) / (float(ratio) * part["n_positive"] + part["n_negative"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["cost_ratio_fn_to_fp", "method"], as_index=False)
        .agg(
            mean_normalized_cost=("normalized_cost", "mean"),
            sd_normalized_cost=("normalized_cost", "std"),
            median_normalized_cost=("normalized_cost", "median"),
            n_blocks=("normalized_cost", "size"),
        )
        .sort_values(["cost_ratio_fn_to_fp", "mean_normalized_cost"])
    )
    out["rank"] = out.groupby("cost_ratio_fn_to_fp")["mean_normalized_cost"].rank(
        method="min", ascending=True
    )
    return out


def summarize_costs_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["cost_ratio_fn_to_fp", "dataset", "method"], as_index=False)
        .agg(
            mean_normalized_cost=("normalized_cost", "mean"),
            sd_across_repeats=("normalized_cost", "std"),
            n_repeats=("normalized_cost", "size"),
        )
        .sort_values(["cost_ratio_fn_to_fp", "dataset", "mean_normalized_cost"])
    )


def count_cost_wins(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    best = temp.groupby(
        ["cost_ratio_fn_to_fp", "dataset", "seed"]
    )["normalized_cost"].transform("min")
    temp["is_block_winner"] = np.isclose(temp["normalized_cost"], best, atol=1e-12)
    return (
        temp.groupby(["cost_ratio_fn_to_fp", "method"], as_index=False)
        .agg(
            wins=("is_block_winner", "sum"),
            n_blocks=("is_block_winner", "size"),
        )
        .sort_values(["cost_ratio_fn_to_fp", "wins"], ascending=[True, False])
    )


def build_fpr_fnr_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["fnr"] = 1.0 - temp["sensitivity"]
    temp["fpr"] = 1.0 - temp["specificity"]
    return (
        temp.groupby("method", as_index=False)
        .agg(
            mean_fnr=("fnr", "mean"),
            sd_fnr=("fnr", "std"),
            mean_fpr=("fpr", "mean"),
            sd_fpr=("fpr", "std"),
        )
        .sort_values(["mean_fnr", "mean_fpr"])
    )


def feature_stability(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "selected_features" not in raw:
        raise ValueError("Triad raw file lacks selected_features.")
    masks: list[set[int]] = []
    for text in raw["selected_features"].dropna():
        values = ast.literal_eval(str(text))
        masks.append({int(value) for value in values})
    if not masks:
        raise ValueError("No valid feature masks.")

    all_features = sorted(set().union(*masks))
    frequency_rows = []
    for feature in all_features:
        count = sum(feature in mask for mask in masks)
        frequency_rows.append(
            {
                "feature": feature,
                "attribute": f"Attr{feature}" if feature <= 64 else "Risk",
                "description": ATTRIBUTE_NAMES.get(feature, ""),
                "selected_count": count,
                "n_runs": len(masks),
                "selection_frequency": count / len(masks),
                "forced_feature": feature == 65,
            }
        )
    frequency = pd.DataFrame(frequency_rows).sort_values(
        ["forced_feature", "selection_frequency", "feature"],
        ascending=[True, False, True],
    )

    original_masks = [{f for f in mask if f <= 64} for mask in masks]
    jaccards = []
    for left, right in itertools.combinations(original_masks, 2):
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    sizes = np.array([len(mask) for mask in original_masks], dtype=float)
    stability = pd.DataFrame(
        [
            {
                "n_runs": len(original_masks),
                "n_pairs": len(jaccards),
                "mean_original_feature_count": float(sizes.mean()),
                "sd_original_feature_count": float(sizes.std(ddof=1)),
                "mean_pairwise_jaccard": float(np.mean(jaccards)),
                "median_pairwise_jaccard": float(np.median(jaccards)),
                "min_pairwise_jaccard": float(np.min(jaccards)),
                "max_pairwise_jaccard": float(np.max(jaccards)),
            }
        ]
    )
    return frequency, stability


def plot_cost_profiles(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for method, group in summary.groupby("method"):
        group = group.sort_values("cost_ratio_fn_to_fp")
        linewidth = 2.6 if "Triad" in method else 1.4
        ax.plot(
            group["cost_ratio_fn_to_fp"],
            group["mean_normalized_cost"],
            marker="o",
            linewidth=linewidth,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xticks(DEFAULT_RATIOS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"Relative cost ratio $C_{FN}/C_{FP}$")
    ax.set_ylabel("Mean normalized decision cost (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fpr_fnr(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for _, row in summary.iterrows():
        method = str(row["method"])
        size = 70 if "Triad" in method else 42
        ax.scatter(row["mean_fpr"], row["mean_fnr"], s=size)
        ax.annotate(method, (row["mean_fpr"], row["mean_fnr"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("Mean false-positive rate")
    ax.set_ylabel("Mean false-negative rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_frequency(frequency: pd.DataFrame, path: Path) -> None:
    top = frequency[~frequency["forced_feature"]].head(12).sort_values(
        "selection_frequency", ascending=True
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.barh(top["attribute"], top["selection_frequency"], color="#3274A1")
    ax.set_xlabel("Selection frequency across 50 dataset–seed runs")
    ax.set_xlim(0, 0.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
