from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

CORE_DIR = Path(__file__).resolve().parents[1] / "01_core_polish_model"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE_DIR))

from bankruptcy_baingo_kelm import (
    classification_metrics,
    focus_weights,
    load_arff_dataset,
    preprocess_splits,
    ranking_metrics,
    safe_roc_auc,
    select_threshold,
)

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
except Exception:  # pragma: no cover - optional dependency in some environments
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed external baselines for bankruptcy data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "03_data" / "01_raw" / "polish_bankruptcy",
    )
    parser.add_argument("--dataset", default="all", help="ARFF file name or 'all'.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-fit", type=int, default=1600)
    parser.add_argument("--max-val", type=int, default=900)
    parser.add_argument("--focus", choices=["balanced", "recall", "f2", "risk", "precision", "auprc", "triage", "ranking"], default="auprc")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "04_results" / "reproduction_check_external_baselines.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

    if args.dataset == "all":
        datasets = sorted(args.data_dir.glob("*.arff"))
    else:
        datasets = [args.data_dir / args.dataset]

    _, threshold_weights = focus_weights(args.focus)
    rows: list[dict[str, float | str | int]] = []
    total = len(datasets) * len(args.seeds) * len(model_factories())
    done = 0
    for dataset_path in datasets:
        x, y = load_arff_dataset(dataset_path)
        for seed in args.seeds:
            x_fit, y_fit, x_val, y_val, x_test, y_test, _ = preprocess_splits(
                x,
                y,
                seed=seed,
                max_fit=args.max_fit,
                max_val=args.max_val,
                risk_feature="none",
            )
            for method, factory in model_factories().items():
                done += 1
                print(f"[{done}/{total}] {dataset_path.name} seed={seed} model={method}", flush=True)
                model = factory(seed)
                model.fit(x_fit, y_fit)
                val_scores = predict_scores(model, x_val)
                test_scores = predict_scores(model, x_test)
                threshold, _ = select_threshold(y_val, val_scores, threshold_weights)
                metrics = classification_metrics(y_test, test_scores >= threshold)
                metrics.update(ranking_metrics(y_test, test_scores))
                metrics.update(
                    {
                        "ap": float(average_precision_score(y_test, test_scores)),
                        "auc": safe_roc_auc(y_test, test_scores),
                    }
                )
                row: dict[str, float | str | int] = {
                    "dataset": dataset_path.name,
                    "seed": seed,
                    "method": method,
                    "category": "External ML baseline",
                    "source": "run_external_baselines.py",
                }
                row.update(metrics)
                rows.append(row)
                write_rows(args.output, rows)
                print(
                    "  test "
                    f"gmean={metrics['gmean']:.4f} "
                    f"sens={metrics['sensitivity']:.4f} "
                    f"spec={metrics['specificity']:.4f} "
                    f"prec={metrics['precision']:.4f} "
                    f"f2={metrics['f2']:.4f} "
                    f"ap={metrics['ap']:.4f}",
                    flush=True,
                )
    print(f"Saved: {args.output}")


def model_factories() -> dict[str, object]:
    factories = {
        "LR_balanced": lambda seed: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "RF_balanced": lambda seed: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "ExtraTrees_balanced": lambda seed: ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGB": lambda seed: HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed,
        ),
    }
    if BalancedRandomForestClassifier is not None:
        factories["BalancedRF"] = lambda seed: BalancedRandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=seed,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
        )
    if EasyEnsembleClassifier is not None:
        factories["EasyEnsemble"] = lambda seed: EasyEnsembleClassifier(
            n_estimators=20,
            random_state=seed,
            n_jobs=-1,
        )
    return factories


def predict_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
