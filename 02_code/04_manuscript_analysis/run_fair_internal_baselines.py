from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "02_code" / "01_core_polish_model"
sys.path.insert(0, str(CORE_DIR))

import bankruptcy_baingo_kelm as core


DATA_DIR = ROOT / "03_data" / "01_raw" / "polish_bankruptcy"
OUTPUT_DIR = ROOT / "04_results" / "03_manuscript_tables"
SEEDS = list(range(1, 11))


class BalancedForestReference:
    """Transparent balanced-bootstrap forest used when imbalanced-learn is absent."""

    def __init__(self, seed: int, n_estimators: int = 300) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.models: list[DecisionTreeClassifier] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BalancedForestReference":
        rng = np.random.default_rng(self.seed)
        class_indices = [np.flatnonzero(y == label) for label in [0, 1]]
        sample_size = min(len(indices) for indices in class_indices)
        self.models = []
        for tree_index in range(self.n_estimators):
            selected = np.concatenate(
                [rng.choice(indices, sample_size, replace=True) for indices in class_indices]
            )
            rng.shuffle(selected)
            tree = DecisionTreeClassifier(
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=self.seed + tree_index + 1,
            )
            tree.fit(x[selected], y[selected])
            self.models.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.mean(
            [model.predict_proba(x)[:, 1] for model in self.models], axis=0
        )
        return np.c_[1.0 - probability, probability]


class EasyEnsembleReference:
    """EasyEnsemble-style reference with repeated majority undersampling."""

    def __init__(self, seed: int, n_ensembles: int = 20) -> None:
        self.seed = seed
        self.n_ensembles = n_ensembles
        self.models: list[AdaBoostClassifier] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EasyEnsembleReference":
        rng = np.random.default_rng(self.seed)
        negative = np.flatnonzero(y == 0)
        positive = np.flatnonzero(y == 1)
        self.models = []
        for ensemble_index in range(self.n_ensembles):
            sampled_negative = rng.choice(negative, len(positive), replace=False)
            selected = np.concatenate([positive, sampled_negative])
            rng.shuffle(selected)
            model = AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.seed + ensemble_index + 1,
            )
            model.fit(x[selected], y[selected])
            self.models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.mean(
            [model.predict_proba(x)[:, 1] for model in self.models], axis=0
        )
        return np.c_[1.0 - probability, probability]


def model_factories() -> dict[str, object]:
    return {
        "Balanced_logistic": lambda seed: LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "Random_forest": lambda seed: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "Extra_trees": lambda seed: ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGB_fair": lambda seed: HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed + 709,
        ),
        "Balanced_forest_reference": lambda seed: BalancedForestReference(seed),
        "EasyEnsemble_reference": lambda seed: EasyEnsembleReference(seed),
    }


def predict_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    return np.asarray(model.decision_function(x))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    _, threshold_weights = core.focus_weights("triad")
    factories = model_factories()

    for data_path in sorted(DATA_DIR.glob("*.arff")):
        x, y = core.load_arff_dataset(data_path)
        for seed in SEEDS:
            x_fit, y_fit, x_val, y_val, x_test, y_test, _ = core.preprocess_splits(
                x,
                y,
                seed=seed,
                max_fit=2800,
                max_val=1400,
                risk_feature="none",
            )
            for method, factory in factories.items():
                started = time.perf_counter()
                model = factory(seed)
                model.fit(x_fit, y_fit)
                fit_seconds = time.perf_counter() - started
                val_scores = predict_scores(model, x_val)
                test_scores = predict_scores(model, x_test)
                threshold, val_metrics = core.select_threshold(
                    y_val, val_scores, threshold_weights
                )
                metrics = core.classification_metrics(
                    y_test, test_scores >= threshold
                )
                metrics["ap"] = core.safe_average_precision(y_test, test_scores)
                metrics["auc"] = core.safe_roc_auc(y_test, test_scores)
                row: dict[str, object] = {
                    "dataset": data_path.name,
                    "seed": seed,
                    "method": method,
                    "fit_n": len(y_fit),
                    "validation_n": len(y_val),
                    "test_n": len(y_test),
                    "test_positive": int(y_test.sum()),
                    "threshold_policy": "triad",
                    "threshold": float(threshold),
                    "validation_threshold_utility": float(
                        val_metrics.get("threshold_utility", np.nan)
                    ),
                    "fit_seconds": fit_seconds,
                }
                row.update(metrics)
                rows.append(row)
                predictions.append(
                    pd.DataFrame(
                        {
                            "dataset": data_path.name,
                            "seed": seed,
                            "method": method,
                            "row_id": np.arange(len(y_test)),
                            "target": y_test,
                            "score": test_scores,
                            "threshold": threshold,
                        }
                    )
                )
                print(
                    data_path.name,
                    seed,
                    method,
                    f"MCC={metrics['mcc']:.4f}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(
                OUTPUT_DIR / "fair_internal_baselines.csv", index=False
            )
            pd.concat(predictions, ignore_index=True).to_csv(
                OUTPUT_DIR / "fair_internal_predictions.csv", index=False
            )

    manifest = {
        "datasets": [path.name for path in sorted(DATA_DIR.glob("*.arff"))],
        "seeds": SEEDS,
        "fit_cap": 2800,
        "validation_cap": 1400,
        "outer_split": "70% train / 30% test, stratified",
        "validation_split": "25% of outer training, stratified",
        "threshold_policy": "triad",
        "histgb": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "l2_regularization": 0.1,
        },
        "note": (
            "All references use the same split, capped fitting/validation samples, "
            "preprocessing, and validation-only triad threshold rule as the proposed model."
        ),
    }
    (OUTPUT_DIR / "fair_internal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
