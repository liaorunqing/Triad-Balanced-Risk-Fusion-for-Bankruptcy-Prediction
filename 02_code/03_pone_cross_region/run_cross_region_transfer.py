from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "02_code"
    / "02_external_validation"
    / "run_external_validation.py"
)
DEFAULT_DATA = (
    ROOT
    / "03_data"
    / "02_processed"
    / "pone_global_2016"
    / "pone_regional_long_deduplicated.csv"
)
FEATURES = [f"V{i}" for i in range(1, 11)]
REGIONS = ["Asia", "Europe", "North America"]


def load_runner():
    spec = importlib.util.spec_from_file_location("pone_cross_region_base", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--partition-reference",
        type=Path,
        default=None,
        help=(
            "Optional deduplicated reference data. Common source rows retain the "
            "reference fit/search/calibration assignment; duplicate rows follow "
            "their original model-row key."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "04_results"
            / "02_pone_cross_region"
            / "new_run"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260727])
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--landmarks", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=100)
    return parser.parse_args()


def model_row_key(row: pd.Series) -> tuple[object, ...]:
    values: list[object] = [row["country"]]
    for column in [*FEATURES, "GICS", "target"]:
        value = row[column]
        values.append("NA" if pd.isna(value) else float(value))
    return tuple(values)


def prepare_task(
    runner,
    frame: pd.DataFrame,
    horizon: int,
    target_region: str,
    split_seed: int,
    partition_reference: pd.DataFrame | None = None,
):
    horizon_frame = frame.loc[frame["horizon"] == horizon].reset_index(drop=True)
    source = horizon_frame.loc[horizon_frame["region"] != target_region].reset_index(drop=True)
    target = horizon_frame.loc[horizon_frame["region"] == target_region].reset_index(drop=True)
    reference_source = source
    if partition_reference is not None:
        reference_horizon = partition_reference.loc[
            partition_reference["horizon"] == horizon
        ].reset_index(drop=True)
        reference_source = reference_horizon.loc[
            reference_horizon["region"] != target_region
        ].reset_index(drop=True)
    reference_y = reference_source["target"].to_numpy(dtype=int)
    reference_idx = np.arange(len(reference_source))
    reference_fit_idx, reference_hold_idx = train_test_split(
        reference_idx,
        test_size=0.40,
        stratify=reference_y,
        random_state=split_seed + 100 * horizon + REGIONS.index(target_region),
    )
    reference_search_idx, reference_cal_idx = train_test_split(
        reference_hold_idx,
        test_size=0.50,
        stratify=reference_y[reference_hold_idx],
        random_state=split_seed + 1000 + 100 * horizon + REGIONS.index(target_region),
    )
    if partition_reference is None:
        fit_idx = reference_fit_idx
        search_idx = reference_search_idx
        cal_idx = reference_cal_idx
    else:
        partition_by_key: dict[tuple[object, ...], str] = {}
        for name, indices in [
            ("fit", reference_fit_idx),
            ("search", reference_search_idx),
            ("cal", reference_cal_idx),
        ]:
            for index in indices:
                key = model_row_key(reference_source.iloc[index])
                existing = partition_by_key.get(key)
                if existing is not None and existing != name:
                    raise RuntimeError(
                        f"Reference model-row key appears in multiple partitions: {key}"
                    )
                partition_by_key[key] = name
        assignments = source.apply(
            lambda row: partition_by_key.get(model_row_key(row)), axis=1
        )
        if assignments.isna().any():
            raise RuntimeError(
                f"{int(assignments.isna().sum())} raw source rows have no reference key"
            )
        fit_idx = np.flatnonzero(assignments.to_numpy() == "fit")
        search_idx = np.flatnonzero(assignments.to_numpy() == "search")
        cal_idx = np.flatnonzero(assignments.to_numpy() == "cal")
    source_y = source["target"].to_numpy(dtype=int)
    name = f"PONE2016_h{horizon}_target_{target_region.replace(' ', '_')}"
    raw = runner.SplitData(
        name=name,
        x_fit=source.iloc[fit_idx][FEATURES].to_numpy(dtype=float),
        y_fit=source_y[fit_idx],
        x_search=source.iloc[search_idx][FEATURES].to_numpy(dtype=float),
        y_search=source_y[search_idx],
        x_cal=source.iloc[cal_idx][FEATURES].to_numpy(dtype=float),
        y_cal=source_y[cal_idx],
        x_test=target[FEATURES].to_numpy(dtype=float),
        y_test=target["target"].to_numpy(dtype=int),
        feature_names=FEATURES,
        design=(
            f"Same-horizon leave-one-region-out transfer; source regions="
            f"{','.join(r for r in REGIONS if r != target_region)}; "
            f"target region={target_region}; source-only fit/search/calibration; "
            "target labels untouched until final evaluation; V1-V10 financial ratios only."
        ),
    )
    return runner.preprocess(raw), source, target


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    frame = pd.read_csv(args.data)
    partition_reference = (
        pd.read_csv(args.partition_reference)
        if args.partition_reference is not None
        else None
    )
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []

    for horizon in [1, 2, 3]:
        for target_region in REGIONS:
            data, source, target = prepare_task(
                runner,
                frame,
                horizon,
                target_region,
                args.split_seed,
                partition_reference=partition_reference,
            )
            for seed in args.seeds:
                print(
                    f"[task] horizon={horizon} target={target_region} seed={seed}",
                    flush=True,
                )
                task_metrics, task_predictions, task_selections = runner.run_dataset(
                    data=data,
                    seed=seed,
                    pop_size=args.pop_size,
                    iterations=args.iterations,
                    landmarks=args.landmarks,
                    bootstrap=args.bootstrap,
                )

                logistic = LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    solver="liblinear",
                    max_iter=4000,
                    random_state=seed + 1301,
                )
                logistic.fit(data.x_fit, data.y_fit)
                log_rows, log_pred = runner.evaluate_scores(
                    data=data,
                    method="Balanced_logistic",
                    search_scores=logistic.decision_function(data.x_search),
                    cal_scores=logistic.decision_function(data.x_cal),
                    test_scores=logistic.decision_function(data.x_test),
                    seed=seed + 1301,
                    bootstrap=args.bootstrap,
                    selection_note=(
                        "Class-weighted logistic reference using the same source-only "
                        "fit/search/calibration and untouched regional target."
                    ),
                )
                for row in log_rows:
                    row["model_seed"] = row.get("seed")
                    row["seed"] = seed
                log_pred["model_seed"] = log_pred["seed"]
                log_pred["seed"] = seed
                task_metrics.extend(log_rows)
                task_predictions.append(log_pred)

                for row in task_metrics:
                    row["horizon"] = horizon
                    row["target_region"] = target_region
                    row["source_n"] = len(source)
                    row["target_n"] = len(target)
                    row["feature_set"] = "V1-V10_financial_only"
                for selection in task_selections:
                    selection["horizon"] = horizon
                    selection["target_region"] = target_region
                target_meta = target[["source_file", "source_row", "country", "region"]]
                for prediction in task_predictions:
                    prediction["horizon"] = horizon
                    prediction["target_region"] = target_region
                    prediction["source_file"] = prediction["row_id"].map(
                        target_meta["source_file"]
                    )
                    prediction["source_row"] = prediction["row_id"].map(
                        target_meta["source_row"]
                    )
                    prediction["country"] = prediction["row_id"].map(
                        target_meta["country"]
                    )

                metrics.extend(task_metrics)
                predictions.extend(task_predictions)
                selections.extend(task_selections)
                pd.DataFrame(metrics).to_csv(
                    args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
                )
                pd.concat(predictions, ignore_index=True).to_csv(
                    args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
                )
                pd.DataFrame(selections).to_csv(
                    args.output_dir / "selection.csv", index=False, encoding="utf-8-sig"
                )

    manifest = {
        "data": str(args.data.resolve()),
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "pop_size": args.pop_size,
        "iterations": args.iterations,
        "landmarks": args.landmarks,
        "bootstrap": args.bootstrap,
        "features": FEATURES,
        "tasks": 9,
        "global_files_used": False,
        "partition_reference": (
            str(args.partition_reference.resolve())
            if args.partition_reference is not None
            else None
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved results under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
