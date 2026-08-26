from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12


@dataclass(frozen=True)
class EvalResult:
    loss: float
    metrics: dict[str, float]
    threshold: float
    selected_features: np.ndarray
    params: dict[str, float]


@dataclass
class OptimizerResult:
    name: str
    seed: int
    best_position: np.ndarray
    best_eval: EvalResult
    history: list[dict[str, float]]


class BankruptcyKELMFitness:
    """Cost-sensitive reduced RBF-KELM wrapper fitness for bankruptcy data."""

    def __init__(
        self,
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        mi_scores: np.ndarray,
        rng: np.random.Generator,
        n_landmarks: int = 160,
        feature_penalty: float = 0.02,
        min_features: int = 4,
        focus: str = "balanced",
        ensemble_size: int = 1,
        neg_pos_ratio: float = 3.0,
        feature_mode: str = "binary",
        force_last_feature: bool = False,
        score_blend_alpha: float = -1.0,
        auto_blend: bool = False,
    ) -> None:
        self.x_fit = np.asarray(x_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=int)
        self.x_val = np.asarray(x_val, dtype=float)
        self.y_val = np.asarray(y_val, dtype=int)
        self.y_signed = np.where(self.y_fit == 1, 1.0, -1.0)
        self.mi_scores = np.asarray(mi_scores, dtype=float)
        self.n_features = self.x_fit.shape[1]
        self.feature_penalty = feature_penalty
        self.min_features = min_features
        self.focus = focus
        self.ensemble_size = max(1, int(ensemble_size))
        self.neg_pos_ratio = max(1.0, float(neg_pos_ratio))
        self.feature_mode = feature_mode
        self.force_last_feature = force_last_feature
        self.score_blend_alpha = score_blend_alpha
        self.auto_blend = auto_blend
        self.blend_alpha_grid = np.array([0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
        self.loss_weights, self.threshold_weights = focus_weights(focus)
        self.eval_count = 0

        n_landmarks = min(n_landmarks, len(self.x_fit))
        self.landmark_idx = stratified_sample_indices(self.y_fit, n_landmarks, rng)
        self.centers_all = self.x_fit[self.landmark_idx]
        self.top_feature_order = np.argsort(-self.mi_scores)
        self.ensemble_specs = make_underbagging_specs(
            self.y_fit,
            n_landmarks,
            self.ensemble_size,
            self.neg_pos_ratio,
            rng,
        )

    @property
    def dim(self) -> int:
        return 3 + self.n_features

    @property
    def lb(self) -> np.ndarray:
        return np.r_[[-2.0, -4.0, -1.0], np.zeros(self.n_features)]

    @property
    def ub(self) -> np.ndarray:
        return np.r_[[4.0, 1.0, 2.3], np.ones(self.n_features)]

    def decode(self, position: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        clipped = np.clip(position, self.lb, self.ub)
        log10_c = float(clipped[0])
        log10_gamma = float(clipped[1])
        log10_cost_pos = float(clipped[2])
        feature_gene = clipped[3:]
        mask = feature_gene >= 0.5
        feature_weights = feature_gene.copy()
        if int(mask.sum()) < self.min_features:
            repaired = np.zeros_like(mask, dtype=bool)
            repaired[self.top_feature_order[: self.min_features]] = True
            repaired |= mask
            mask = repaired
            feature_weights[mask] = np.maximum(feature_weights[mask], 0.65)
        if self.force_last_feature:
            mask[-1] = True
            feature_weights[-1] = max(feature_weights[-1], 0.85)
        if self.feature_mode == "weighted":
            feature_weights = np.where(mask, 0.25 + 0.75 * feature_weights, 0.0)
        else:
            feature_weights = mask.astype(float)
        return 10.0**log10_c, 10.0**log10_gamma, 10.0**log10_cost_pos, mask, feature_weights

    def evaluate(self, position: np.ndarray) -> EvalResult:
        self.eval_count += 1
        c_value, gamma, cost_pos, mask, feature_weights = self.decode(position)
        params = {
            "C": c_value,
            "gamma": gamma,
            "cost_pos": cost_pos,
            "feature_count": float(mask.sum()),
        }

        try:
            feature_ratio = float(mask.mean())
            raw_scores = self._fit_predict_scores(
                c_value,
                gamma,
                cost_pos,
                mask,
                feature_weights,
                blend=False,
            )
            threshold, metrics, blend_alpha = self._select_validation_blend(
                raw_scores,
                feature_ratio,
                float(mask.sum()),
            )
            loss = metrics["loss"]
            params["blend_alpha"] = blend_alpha
        except Exception:
            loss = 1e6
            threshold = 0.0
            metrics = {
                "gmean": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "f05": 0.0,
                "f2": 0.0,
                "mcc": 0.0,
                "triad_score": 0.0,
                "balanced_accuracy": 0.0,
                "fpr": 1.0,
                "predicted_positive_rate": 1.0,
                "p_at_10": 0.0,
                "p_at_15": 0.0,
                "p_at_20": 0.0,
                "recall_at_10": 0.0,
                "recall_at_15": 0.0,
                "recall_at_20": 0.0,
                "ap": 0.0,
                "auc": 0.5,
                "feature_ratio": float(mask.mean()),
                "feature_count": float(mask.sum()),
                "loss": float(loss),
            }
            params["blend_alpha"] = -1.0
        return EvalResult(float(loss), metrics, float(threshold), mask.copy(), params)

    def test_best(
        self,
        position: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        c_value, gamma, cost_pos, mask, feature_weights = self.decode(position)
        raw_val_scores = self._fit_predict_scores(
            c_value,
            gamma,
            cost_pos,
            mask,
            feature_weights,
            blend=False,
        )
        threshold, _, blend_alpha = self._select_validation_blend(
            raw_val_scores,
            float(mask.mean()),
            float(mask.sum()),
        )
        if self.ensemble_size <= 1:
            selected_scale = np.sqrt(np.maximum(feature_weights[mask], EPS))
            model = fit_reduced_rbf_kelm(
                self.x_fit[:, mask] * selected_scale,
                self.y_fit,
                self.centers_all[:, mask] * selected_scale,
                c_value,
                gamma,
                cost_pos,
            )
            test_scores = predict_reduced_rbf_kelm(x_test[:, mask] * selected_scale, model)
        else:
            test_scores = self._ensemble_predict_scores(c_value, gamma, cost_pos, mask, feature_weights, x_test)
        test_scores = self._blend_scores(test_scores, x_test, blend_alpha)
        test_metrics = classification_metrics(y_test, test_scores >= threshold)
        test_metrics.update(ranking_metrics(y_test, test_scores))
        test_metrics.update(
            {
                "ap": safe_average_precision(y_test, test_scores),
                "auc": safe_roc_auc(y_test, test_scores),
                "threshold": float(threshold),
                "feature_count": float(mask.sum()),
                "C": float(c_value),
                "gamma": float(gamma),
                "cost_pos": float(cost_pos),
                "blend_alpha": float(blend_alpha),
            }
        )
        return test_metrics

    def _fit_predict_scores(
        self,
        c_value: float,
        gamma: float,
        cost_pos: float,
        mask: np.ndarray,
        feature_weights: np.ndarray,
        blend: bool = True,
    ) -> np.ndarray:
        if self.ensemble_size > 1:
            scores = self._ensemble_predict_scores(c_value, gamma, cost_pos, mask, feature_weights, self.x_val)
            return self._blend_scores(scores, self.x_val) if blend else scores
        selected_scale = np.sqrt(np.maximum(feature_weights[mask], EPS))
        model = fit_reduced_rbf_kelm(
            self.x_fit[:, mask] * selected_scale,
            self.y_fit,
            self.centers_all[:, mask] * selected_scale,
            c_value,
            gamma,
            cost_pos,
        )
        scores = predict_reduced_rbf_kelm(self.x_val[:, mask] * selected_scale, model)
        return self._blend_scores(scores, self.x_val) if blend else scores

    def _ensemble_predict_scores(
        self,
        c_value: float,
        gamma: float,
        cost_pos: float,
        mask: np.ndarray,
        feature_weights: np.ndarray,
        x_eval: np.ndarray,
    ) -> np.ndarray:
        scores = np.zeros(len(x_eval), dtype=float)
        selected_scale = np.sqrt(np.maximum(feature_weights[mask], EPS))
        for train_idx, center_idx in self.ensemble_specs:
            model = fit_reduced_rbf_kelm(
                self.x_fit[train_idx][:, mask] * selected_scale,
                self.y_fit[train_idx],
                self.x_fit[center_idx][:, mask] * selected_scale,
                c_value,
                gamma,
                cost_pos,
            )
            scores += predict_reduced_rbf_kelm(x_eval[:, mask] * selected_scale, model)
        return scores / max(len(self.ensemble_specs), 1)

    def _blend_scores(
        self,
        kelm_scores: np.ndarray,
        x_eval: np.ndarray,
        alpha: float | None = None,
    ) -> np.ndarray:
        if alpha is None:
            alpha = self.score_blend_alpha
        if alpha < 0.0:
            return kelm_scores
        alpha = float(np.clip(alpha, 0.0, 1.0))
        kelm_prob = 1.0 / (1.0 + np.exp(-np.clip(kelm_scores, -50.0, 50.0)))
        risk_prob = np.clip(x_eval[:, -1], EPS, 1.0 - EPS)
        return alpha * kelm_prob + (1.0 - alpha) * risk_prob

    def _select_validation_blend(
        self,
        raw_scores: np.ndarray,
        feature_ratio: float,
        feature_count: float,
    ) -> tuple[float, dict[str, float], float]:
        if self.auto_blend:
            alpha_grid = self.blend_alpha_grid
        elif self.score_blend_alpha >= 0.0:
            alpha_grid = np.array([float(self.score_blend_alpha)])
        else:
            alpha_grid = np.array([-1.0])

        best_threshold = 0.0
        best_metrics: dict[str, float] | None = None
        best_alpha = float(alpha_grid[0])
        best_loss = np.inf
        for alpha in alpha_grid:
            scores = self._blend_scores(raw_scores, self.x_val, float(alpha))
            threshold, threshold_metrics = select_threshold(
                self.y_val,
                scores,
                self.threshold_weights,
            )
            metrics = dict(threshold_metrics)
            metrics.update(
                {
                    "ap": safe_average_precision(self.y_val, scores),
                    "auc": safe_roc_auc(self.y_val, scores),
                    "feature_ratio": feature_ratio,
                    "feature_count": feature_count,
                    "blend_alpha": float(alpha),
                }
            )
            metrics.update(ranking_metrics(self.y_val, scores))
            loss = weighted_loss(metrics, self.loss_weights) + self.feature_penalty * feature_ratio
            metrics["loss"] = float(loss)
            if loss < best_loss:
                best_loss = float(loss)
                best_threshold = float(threshold)
                best_metrics = metrics
                best_alpha = float(alpha)

        assert best_metrics is not None
        return best_threshold, best_metrics, best_alpha


def load_arff_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data, _ = arff.loadarff(path)
    frame = pd.DataFrame(data)
    x = frame.iloc[:, :-1].astype(float).to_numpy()
    raw_y = frame.iloc[:, -1].to_numpy()
    y = np.array([int(v.decode() if isinstance(v, bytes) else v) for v in raw_y], dtype=int)
    return x, y


def preprocess_splits(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.3,
    val_size: float = 0.25,
    max_fit: int | None = None,
    max_val: int | None = None,
    risk_feature: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=seed + 17,
    )
    rng = np.random.default_rng(seed + 31)
    if max_fit is not None and len(x_fit) > max_fit:
        idx = stratified_sample_indices(y_fit, max_fit, rng)
        x_fit, y_fit = x_fit[idx], y_fit[idx]
    if max_val is not None and len(x_val) > max_val:
        idx = stratified_sample_indices(y_val, max_val, rng)
        x_val, y_val = x_val[idx], y_val[idx]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    x_fit = imputer.fit_transform(x_fit)
    x_val = imputer.transform(x_val)
    x_test = imputer.transform(x_test)
    x_fit = np.clip(scaler.fit_transform(x_fit), -8.0, 8.0)
    x_val = np.clip(scaler.transform(x_val), -8.0, 8.0)
    x_test = np.clip(scaler.transform(x_test), -8.0, 8.0)

    if risk_feature == "histgb":
        x_fit, x_val, x_test = append_histgb_risk_feature(
            x_fit,
            y_fit,
            x_val,
            x_test,
            seed,
        )

    mi = mutual_info_classif(x_fit, y_fit, discrete_features=False, random_state=seed)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    if float(mi.max()) > 0.0:
        mi = mi / float(mi.max())
    return x_fit, y_fit, x_val, y_val, x_test, y_test, mi


def append_histgb_risk_feature(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_class_count = int(np.bincount(y_fit).min())
    n_splits = max(2, min(5, min_class_count))
    oof_risk = np.zeros(len(y_fit), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 701)
    for fold, (train_idx, hold_idx) in enumerate(cv.split(x_fit, y_fit)):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed + 701 + fold,
        )
        model.fit(x_fit[train_idx], y_fit[train_idx])
        oof_risk[hold_idx] = model.predict_proba(x_fit[hold_idx])[:, 1]

    final_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=seed + 709,
    )
    final_model.fit(x_fit, y_fit)
    val_risk = final_model.predict_proba(x_val)[:, 1]
    test_risk = final_model.predict_proba(x_test)[:, 1]
    return (
        np.c_[x_fit, oof_risk],
        np.c_[x_val, val_risk],
        np.c_[x_test, test_risk],
    )


def fit_reduced_rbf_kelm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    centers: np.ndarray,
    c_value: float,
    gamma: float,
    cost_pos: float,
) -> dict[str, np.ndarray | float]:
    h = rbf_design(x_train, centers, gamma)
    weights = np.where(y_train == 1, cost_pos, 1.0).astype(float)
    weighted_h = h * weights[:, None]
    reg = np.eye(h.shape[1]) / max(c_value, EPS)
    a = h.T @ weighted_h + reg + 1e-8 * np.eye(h.shape[1])
    b = h.T @ (weights * np.where(y_train == 1, 1.0, -1.0))
    beta = np.linalg.solve(a, b)
    return {"centers": centers, "gamma": float(gamma), "beta": beta}


def predict_reduced_rbf_kelm(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    centers = np.asarray(model["centers"], dtype=float)
    gamma = float(model["gamma"])
    beta = np.asarray(model["beta"], dtype=float)
    return rbf_design(x, centers, gamma) @ beta


def rbf_design(x: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    sq_dist = np.maximum(x_norm + c_norm - 2.0 * (x @ centers.T), 0.0)
    return np.exp(-gamma * sq_dist)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred.astype(int), labels=labels).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    predicted_positive_rate = (tp + fp) / max(tp + fp + tn + fn, 1)
    gmean = math.sqrt(max(sensitivity * specificity, 0.0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    triad_score = float(max(max(mcc, 0.0) * sensitivity * specificity, 0.0) ** (1.0 / 3.0))
    return {
        "gmean": float(gmean),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f05": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "mcc": mcc,
        "triad_score": triad_score,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": float(fpr),
        "predicted_positive_rate": float(predicted_positive_rate),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores)
    total_pos = max(int(np.sum(y_true == 1)), 1)
    metrics: dict[str, float] = {}
    for rate, suffix in [(0.10, "10"), (0.15, "15"), (0.20, "20")]:
        k = max(1, int(math.ceil(rate * len(y_true))))
        top = y_true[order[:k]]
        hits = int(np.sum(top == 1))
        metrics[f"p_at_{suffix}"] = float(hits / k)
        metrics[f"recall_at_{suffix}"] = float(hits / total_pos)
    return metrics


def focus_weights(focus: str) -> tuple[dict[str, float], dict[str, float]]:
    if focus == "recall":
        loss_weights = {"gmean": 0.45, "sensitivity": 0.35, "f2": 0.15, "ap": 0.05}
        threshold_weights = {"gmean": 0.45, "sensitivity": 0.35, "f2": 0.15, "specificity": 0.05}
    elif focus == "f2":
        loss_weights = {"gmean": 0.40, "sensitivity": 0.25, "f2": 0.30, "ap": 0.05}
        threshold_weights = {"gmean": 0.40, "sensitivity": 0.25, "f2": 0.30, "specificity": 0.05}
    elif focus == "risk":
        loss_weights = {
            "gmean": 0.30,
            "sensitivity": 0.18,
            "specificity": 0.22,
            "precision": 0.16,
            "f1": 0.09,
            "ap": 0.05,
        }
        threshold_weights = {
            "gmean": 0.30,
            "sensitivity": 0.18,
            "specificity": 0.22,
            "precision": 0.16,
            "f1": 0.09,
            "ap": 0.05,
            "_min_sensitivity": 0.50,
            "_min_specificity": 0.68,
            "_constraint_penalty": 0.65,
        }
    elif focus == "precision":
        loss_weights = {
            "gmean": 0.22,
            "sensitivity": 0.10,
            "specificity": 0.26,
            "precision": 0.24,
            "f05": 0.13,
            "ap": 0.05,
        }
        threshold_weights = {
            "gmean": 0.22,
            "sensitivity": 0.10,
            "specificity": 0.26,
            "precision": 0.24,
            "f05": 0.13,
            "ap": 0.05,
            "_min_sensitivity": 0.38,
            "_min_specificity": 0.78,
            "_constraint_penalty": 0.80,
        }
    elif focus == "auprc":
        loss_weights = {
            "ap": 0.30,
            "gmean": 0.20,
            "sensitivity": 0.12,
            "specificity": 0.16,
            "precision": 0.14,
            "f1": 0.08,
        }
        threshold_weights = {
            "gmean": 0.20,
            "sensitivity": 0.12,
            "specificity": 0.20,
            "precision": 0.22,
            "f1": 0.16,
            "f05": 0.10,
            "_min_sensitivity": 0.45,
            "_min_specificity": 0.72,
            "_constraint_penalty": 0.70,
        }
    elif focus == "mcc":
        loss_weights = {
            "mcc": 0.30,
            "gmean": 0.24,
            "sensitivity": 0.16,
            "specificity": 0.16,
            "f1": 0.08,
            "ap": 0.06,
        }
        threshold_weights = {
            "mcc": 0.34,
            "gmean": 0.24,
            "sensitivity": 0.15,
            "specificity": 0.17,
            "f1": 0.07,
            "precision": 0.03,
            "_min_sensitivity": 0.45,
            "_min_specificity": 0.82,
            "_min_tp": 1.0,
            "_constraint_penalty": 0.85,
        }
    elif focus == "mcc_guarded":
        loss_weights = {
            "mcc": 0.34,
            "triad_score": 0.26,
            "gmean": 0.16,
            "specificity": 0.10,
            "sensitivity": 0.08,
            "f1": 0.04,
            "ap": 0.02,
        }
        threshold_weights = {
            "mcc": 0.44,
            "triad_score": 0.24,
            "specificity": 0.12,
            "sensitivity": 0.08,
            "gmean": 0.08,
            "precision": 0.04,
            "_min_sensitivity": 0.45,
            "_min_specificity": 0.94,
            "_min_tp": 1.0,
            "_max_predicted_positive_rate": 0.13,
            "_constraint_penalty": 1.25,
        }
    elif focus == "triad":
        loss_weights = {
            "triad_score": 0.36,
            "mcc": 0.24,
            "gmean": 0.16,
            "sensitivity": 0.10,
            "specificity": 0.10,
            "f1": 0.03,
            "ap": 0.01,
        }
        threshold_weights = {
            "triad_score": 0.42,
            "mcc": 0.26,
            "gmean": 0.14,
            "sensitivity": 0.08,
            "specificity": 0.08,
            "f1": 0.02,
            "_min_sensitivity": 0.50,
            "_min_specificity": 0.90,
            "_min_tp": 1.0,
            "_max_predicted_positive_rate": 0.18,
            "_constraint_penalty": 1.10,
        }
    elif focus == "triage":
        loss_weights = {
            "ap": 0.32,
            "precision": 0.22,
            "specificity": 0.18,
            "f05": 0.12,
            "gmean": 0.10,
            "sensitivity": 0.06,
        }
        threshold_weights = {
            "precision": 0.30,
            "specificity": 0.24,
            "f05": 0.18,
            "gmean": 0.12,
            "f1": 0.08,
            "sensitivity": 0.08,
            "_min_sensitivity": 0.32,
            "_min_specificity": 0.82,
            "_min_tp": 1.0,
            "_max_predicted_positive_rate": 0.20,
            "_constraint_penalty": 0.90,
        }
    elif focus == "ranking":
        loss_weights = {
            "ap": 0.22,
            "p_at_10": 0.18,
            "p_at_15": 0.14,
            "recall_at_15": 0.14,
            "gmean": 0.14,
            "sensitivity": 0.10,
            "specificity": 0.08,
        }
        threshold_weights = {
            "precision": 0.24,
            "specificity": 0.22,
            "f05": 0.16,
            "gmean": 0.16,
            "sensitivity": 0.12,
            "f1": 0.10,
            "_min_sensitivity": 0.30,
            "_min_specificity": 0.75,
            "_min_tp": 1.0,
            "_max_predicted_positive_rate": 0.18,
            "_constraint_penalty": 0.75,
        }
    else:
        loss_weights = {"gmean": 0.55, "sensitivity": 0.25, "f2": 0.15, "ap": 0.05}
        threshold_weights = {"gmean": 0.55, "sensitivity": 0.25, "f2": 0.15, "specificity": 0.05}
    return loss_weights, threshold_weights


def weighted_loss(metrics: dict[str, float], weights: dict[str, float]) -> float:
    loss = 0.0
    for metric, weight in weights.items():
        if metric.startswith("_"):
            continue
        loss += weight * (1.0 - metrics.get(metric, 0.0))
    return float(loss)


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    if weights is None:
        _, weights = focus_weights("balanced")
    min_sensitivity = weights.get("_min_sensitivity", 0.0)
    min_specificity = weights.get("_min_specificity", 0.0)
    min_tp = weights.get("_min_tp", 0.0)
    max_predicted_positive_rate = weights.get("_max_predicted_positive_rate", 1.0)
    constraint_penalty = weights.get("_constraint_penalty", 0.0)
    quantiles = np.r_[
        np.linspace(0.35, 0.98, 80),
        np.linspace(0.981, 0.9995, 35),
    ]
    thresholds = np.unique(np.quantile(scores, quantiles))
    thresholds = np.r_[thresholds, [float(scores.min()) - EPS, 0.0, float(scores.max()) + EPS]]
    best_threshold = float(thresholds[0])
    best_metrics: dict[str, float] | None = None
    best_utility = -np.inf
    feasible_threshold = float(thresholds[0])
    feasible_metrics: dict[str, float] | None = None
    feasible_utility = -np.inf
    for threshold in thresholds:
        metrics = classification_metrics(y_true, scores >= threshold)
        utility = 0.0
        for metric, weight in weights.items():
            if metric.startswith("_") or metric == "ap":
                continue
            utility += weight * metrics.get(metric, 0.0)
        if min_sensitivity > 0.0:
            utility -= constraint_penalty * max(0.0, min_sensitivity - metrics["sensitivity"])
        if min_specificity > 0.0:
            utility -= constraint_penalty * max(0.0, min_specificity - metrics["specificity"])
        if max_predicted_positive_rate < 1.0:
            utility -= constraint_penalty * max(
                0.0,
                metrics["predicted_positive_rate"] - max_predicted_positive_rate,
            )
        feasible = (
            metrics["sensitivity"] >= min_sensitivity
            and metrics["specificity"] >= min_specificity
            and metrics["tp"] >= min_tp
            and metrics["predicted_positive_rate"] <= max_predicted_positive_rate
        )
        if feasible and utility > feasible_utility:
            feasible_utility = utility
            feasible_threshold = float(threshold)
            feasible_metrics = metrics
        if utility > best_utility:
            best_utility = utility
            best_threshold = float(threshold)
            best_metrics = metrics
    if feasible_metrics is not None:
        feasible_metrics["threshold_utility"] = float(feasible_utility)
        return feasible_threshold, feasible_metrics
    assert best_metrics is not None
    best_metrics["threshold_utility"] = float(best_utility)
    return best_threshold, best_metrics


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return 0.0


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def stratified_sample_indices(
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples >= len(y):
        return np.arange(len(y))
    idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        take = max(1, int(round(n_samples * len(cls_idx) / len(y))))
        take = min(take, len(cls_idx))
        idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    if len(idx) > n_samples:
        idx = rng.choice(idx, size=n_samples, replace=False)
    elif len(idx) < n_samples:
        rest = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
        extra = rng.choice(rest, size=n_samples - len(idx), replace=False)
        idx = np.r_[idx, extra]
    rng.shuffle(idx)
    return idx


def make_underbagging_specs(
    y: np.ndarray,
    n_landmarks: int,
    ensemble_size: int,
    neg_pos_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if ensemble_size <= 1:
        idx = np.arange(len(y))
        center_local = stratified_sample_indices(y, min(n_landmarks, len(y)), rng)
        return [(idx, center_local)]

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        idx = np.arange(len(y))
        center_local = stratified_sample_indices(y, min(n_landmarks, len(y)), rng)
        return [(idx, center_local)]

    specs: list[tuple[np.ndarray, np.ndarray]] = []
    n_neg = min(len(neg_idx), max(1, int(round(neg_pos_ratio * len(pos_idx)))))
    for _ in range(ensemble_size):
        chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        train_idx = np.r_[pos_idx, chosen_neg]
        rng.shuffle(train_idx)
        local_centers = stratified_sample_indices(
            y[train_idx],
            min(n_landmarks, len(train_idx)),
            rng,
        )
        center_idx = train_idx[local_centers]
        specs.append((train_idx.astype(int), center_idx.astype(int)))
    return specs


def initialize_population(
    objective: BankruptcyKELMFitness,
    pop_size: int,
    rng: np.random.Generator,
    mi_guided: bool,
) -> np.ndarray:
    lb, ub = objective.lb, objective.ub
    pop = rng.uniform(lb, ub, size=(pop_size, objective.dim))
    if not mi_guided:
        return pop

    order = objective.top_feature_order
    for i in range(pop_size):
        k = int(rng.integers(objective.min_features, objective.n_features + 1))
        if i < min(5, pop_size):
            k = [8, 12, 16, 24, 32][i]
            k = min(k, objective.n_features)
        mask = np.zeros(objective.n_features, dtype=bool)
        guided_count = int(round(0.75 * k))
        mask[order[:guided_count]] = True
        remaining = np.setdiff1d(np.arange(objective.n_features), order[:guided_count])
        if k > guided_count and len(remaining) > 0:
            mask[rng.choice(remaining, size=k - guided_count, replace=False)] = True
        genes = rng.uniform(0.0, 0.35, size=objective.n_features)
        genes[mask] = rng.uniform(0.65, 1.0, size=int(mask.sum()))
        pop[i, 3:] = genes
        pop[i, 0] = rng.uniform(0.0, 3.5)
        pop[i, 1] = rng.uniform(-3.5, 0.3)
        if objective.focus in {"risk", "precision", "auprc", "mcc", "mcc_guarded", "triad", "triage", "ranking"}:
            pop[i, 2] = rng.uniform(-0.4, 1.6)
        else:
            pop[i, 2] = rng.uniform(0.2, 1.8)
    return np.clip(pop, lb, ub)


def optimize_ngo(
    objective: BankruptcyKELMFitness,
    pop_size: int,
    iterations: int,
    seed: int,
    improved: bool = False,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    pop = initialize_population(objective, pop_size, rng, mi_guided=False)
    evals = [objective.evaluate(ind) for ind in pop]
    history = []
    lb, ub = objective.lb, objective.ub
    span = ub - lb

    for t in range(1, iterations + 1):
        progress = t / max(iterations, 1)
        for i in range(pop_size):
            if improved:
                size = int(round(5 - (5 - 2) * progress))
                candidates = rng.choice(
                    np.delete(np.arange(pop_size), i),
                    size=min(size, pop_size - 1),
                    replace=False,
                )
                prey_idx = min(candidates, key=lambda idx: evals[int(idx)].loss)
            else:
                prey_idx = int(rng.choice(np.delete(np.arange(pop_size), i)))
            prey = pop[prey_idx]
            integer_factor = rng.integers(1, 3, size=objective.dim)
            r = rng.random(objective.dim)
            if evals[prey_idx].loss < evals[i].loss:
                trial = pop[i] + r * (prey - integer_factor * pop[i])
            else:
                trial = pop[i] + r * (pop[i] - prey)
            trial = np.clip(trial, lb, ub)
            trial_eval = objective.evaluate(trial)
            if trial_eval.loss < evals[i].loss:
                pop[i], evals[i] = trial, trial_eval

            if improved:
                r_scale = 0.001 + (0.05 - 0.001) * math.cos(math.pi * progress / 2.0)
                trial = pop[i] + r_scale * (2.0 * rng.random(objective.dim) - 1.0) * span
            else:
                r_scale = 0.02 * (1.0 - progress)
                trial = pop[i] + r_scale * (2.0 * rng.random(objective.dim) - 1.0) * pop[i]
            trial = np.clip(trial, lb, ub)
            trial_eval = objective.evaluate(trial)
            if trial_eval.loss < evals[i].loss:
                pop[i], evals[i] = trial, trial_eval

        best_idx = int(np.argmin([ev.loss for ev in evals]))
        history.append(history_row(t, objective.eval_count, evals[best_idx]))

    best_idx = int(np.argmin([ev.loss for ev in evals]))
    return OptimizerResult(
        name="INGO" if improved else "NGO",
        seed=seed,
        best_position=pop[best_idx].copy(),
        best_eval=evals[best_idx],
        history=history,
    )


def optimize_baingo(
    objective: BankruptcyKELMFitness,
    pop_size: int,
    iterations: int,
    seed: int,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    pop = initialize_population(objective, pop_size, rng, mi_guided=True)
    evals = [objective.evaluate(ind) for ind in pop]
    lb, ub = objective.lb, objective.ub
    span = ub - lb
    history = []
    stagnation = np.zeros(pop_size, dtype=int)
    mi = objective.mi_scores
    mi = mi / max(float(mi.max()), EPS)

    for t in range(1, iterations + 1):
        progress = t / max(iterations, 1)
        losses = np.array([ev.loss for ev in evals])
        sensitivities = np.array([ev.metrics["sensitivity"] for ev in evals])
        best_idx = int(np.argmin(losses))
        best = pop[best_idx].copy()
        elite_indices = np.argsort(losses)[: max(2, pop_size // 4)]
        recall_indices = np.argsort(-sensitivities)[: max(2, pop_size // 4)]

        for i in range(pop_size):
            candidates = []
            candidates.append(make_baingo_prey_trial(pop, evals, i, best, progress, rng, objective))
            candidates.append(make_elite_differential_trial(pop, i, best, elite_indices, rng, lb, ub, progress))
            candidates.append(make_binary_recall_trial(pop[i], best, objective, mi, rng, progress))

            if rng.random() < 0.35 * (1.0 - progress) + 0.05:
                recall_teacher = pop[int(rng.choice(recall_indices))]
                trial = pop[i].copy()
                trial[:3] += rng.normal(0.0, 0.15 + 0.25 * (1.0 - progress), size=3) * span[:3]
                recall_mask = recall_teacher[3:] >= 0.5
                copy = rng.random(objective.n_features) < (0.25 + 0.35 * progress)
                trial[3:][copy] = np.where(recall_mask[copy], 0.85, 0.15)
                candidates.append(np.clip(trial, lb, ub))

            best_trial = None
            best_trial_eval = None
            for trial in candidates:
                trial = np.clip(trial, lb, ub)
                trial_eval = objective.evaluate(trial)
                if best_trial_eval is None or trial_eval.loss < best_trial_eval.loss:
                    best_trial = trial
                    best_trial_eval = trial_eval

            assert best_trial is not None and best_trial_eval is not None
            if best_trial_eval.loss < evals[i].loss:
                pop[i], evals[i] = best_trial, best_trial_eval
                stagnation[i] = 0
            else:
                stagnation[i] += 1

            if stagnation[i] >= max(4, iterations // 4):
                restart = initialize_population(objective, 1, rng, mi_guided=True)[0]
                restart[:3] = 0.55 * pop[best_idx, :3] + 0.45 * restart[:3]
                restart_eval = objective.evaluate(restart)
                if restart_eval.loss < evals[i].loss or i != best_idx:
                    pop[i], evals[i] = restart, restart_eval
                stagnation[i] = 0

        best_idx = int(np.argmin([ev.loss for ev in evals]))
        history.append(history_row(t, objective.eval_count, evals[best_idx]))

    best_idx = int(np.argmin([ev.loss for ev in evals]))
    return OptimizerResult(
        name="BAINGO",
        seed=seed,
        best_position=pop[best_idx].copy(),
        best_eval=evals[best_idx],
        history=history,
    )


def optimize_tisngo(
    objective: BankruptcyKELMFitness,
    pop_size: int,
    iterations: int,
    seed: int,
) -> OptimizerResult:
    """Approximate TIS_NGO baseline with TIS, DE prey attack, and centroid opposition.

    This is an in-code reproduction baseline inspired by the published
    TIS_NGO-KELM mechanism. It keeps the same KELM/feature-selection encoding
    used by the present experiments for fair data-split and objective control.
    """
    rng = np.random.default_rng(seed)
    pop = initialize_population(objective, pop_size, rng, mi_guided=True)
    evals = [objective.evaluate(ind) for ind in pop]
    lb, ub = objective.lb, objective.ub
    span = ub - lb
    history = []

    for t in range(1, iterations + 1):
        progress = t / max(iterations, 1)
        losses = np.array([ev.loss for ev in evals])
        best_idx = int(np.argmin(losses))
        best = pop[best_idx].copy()
        centroid = np.mean(pop, axis=0)
        elite_indices = np.argsort(losses)[: max(3, pop_size // 3)]

        for i in range(pop_size):
            candidates = [
                make_tis_cognitive_trial(pop, i, best, centroid, rng, objective, progress),
                make_tis_de_prey_attack_trial(pop, evals, i, elite_indices, rng, objective, progress),
                make_tis_centroid_opposition_trial(pop[i], centroid, best, lb, ub, rng, progress),
            ]

            best_trial = None
            best_trial_eval = None
            for trial in candidates:
                trial = np.clip(trial, lb, ub)
                trial_eval = objective.evaluate(trial)
                if best_trial_eval is None or trial_eval.loss < best_trial_eval.loss:
                    best_trial = trial
                    best_trial_eval = trial_eval

            assert best_trial is not None and best_trial_eval is not None
            if best_trial_eval.loss < evals[i].loss:
                pop[i], evals[i] = best_trial, best_trial_eval
            elif rng.random() < 0.08 * (1.0 - progress):
                # Boundary-control diversification around centroid opposition.
                trial = np.clip(lb + ub - pop[i] + rng.normal(0.0, 0.04, objective.dim) * span, lb, ub)
                trial_eval = objective.evaluate(trial)
                if trial_eval.loss < evals[i].loss:
                    pop[i], evals[i] = trial, trial_eval

        best_idx = int(np.argmin([ev.loss for ev in evals]))
        history.append(history_row(t, objective.eval_count, evals[best_idx]))

    best_idx = int(np.argmin([ev.loss for ev in evals]))
    return OptimizerResult(
        name="TIS_NGO",
        seed=seed,
        best_position=pop[best_idx].copy(),
        best_eval=evals[best_idx],
        history=history,
    )


def optimize_msbaingo(
    objective: BankruptcyKELMFitness,
    pop_size: int,
    iterations: int,
    seed: int,
    precision_aware: bool = False,
) -> OptimizerResult:
    rng = np.random.default_rng(seed)
    pop = initialize_population(objective, pop_size, rng, mi_guided=True)
    random_count = max(2, pop_size // 3)
    pop[:random_count] = initialize_population(objective, random_count, rng, mi_guided=False)
    evals = [objective.evaluate(ind) for ind in pop]
    lb, ub = objective.lb, objective.ub
    span = ub - lb
    history = []
    stagnation = np.zeros(pop_size, dtype=int)
    mi = objective.mi_scores / max(float(objective.mi_scores.max()), EPS)
    operator_credit = np.ones(8 if precision_aware else 6, dtype=float)

    for t in range(1, iterations + 1):
        progress = t / max(iterations, 1)
        losses = np.array([ev.loss for ev in evals])
        sensitivities = np.array([ev.metrics["sensitivity"] for ev in evals])
        precisions = np.array([ev.metrics["precision"] for ev in evals])
        specificities = np.array([ev.metrics["specificity"] for ev in evals])
        f1_scores = np.array([ev.metrics.get("f1", 0.0) for ev in evals])
        best_idx = int(np.argmin(losses))
        best = pop[best_idx].copy()
        elite_indices = np.argsort(losses)[: max(2, pop_size // 4)]
        recall_indices = np.argsort(-sensitivities)[: max(2, pop_size // 4)]
        precision_rank = 0.45 * precisions + 0.35 * specificities + 0.20 * f1_scores
        precision_indices = np.argsort(-precision_rank)[: max(2, pop_size // 4)]

        for i in range(pop_size):
            trial_builders: list[Callable[[], np.ndarray]] = [
                lambda i=i: make_standard_ngo_trial(pop, evals, i, rng, objective, progress),
                lambda i=i: make_ingo_range_trial(pop, evals, i, rng, objective, progress),
                lambda i=i, best=best: make_baingo_prey_trial(
                    pop, evals, i, best, progress, rng, objective
                ),
                lambda i=i, best=best, elite_indices=elite_indices: make_elite_differential_trial(
                    pop, i, best, elite_indices, rng, lb, ub, progress
                ),
                lambda i=i, best=best: make_binary_recall_trial(
                    pop[i], best, objective, mi, rng, progress
                ),
                lambda i=i, recall_indices=recall_indices: make_recall_archive_trial(
                    pop, i, recall_indices, rng, objective, progress
                ),
            ]
            if precision_aware:
                trial_builders.extend(
                    [
                        lambda i=i, precision_indices=precision_indices: make_precision_archive_trial(
                            pop, i, precision_indices, rng, objective, progress
                        ),
                        lambda i=i, best=best: make_fp_control_trial(
                            pop[i], best, objective, mi, rng, progress
                        ),
                    ]
                )

            probs = operator_credit / operator_credit.sum()
            chosen_ops = rng.choice(
                np.arange(len(trial_builders)),
                size=min(4, len(trial_builders)),
                replace=False,
                p=probs,
            )
            best_trial = None
            best_trial_eval = None
            best_op = -1
            for op_idx in chosen_ops:
                trial = np.clip(trial_builders[int(op_idx)](), lb, ub)
                trial_eval = objective.evaluate(trial)
                if best_trial_eval is None or trial_eval.loss < best_trial_eval.loss:
                    best_trial = trial
                    best_trial_eval = trial_eval
                    best_op = int(op_idx)

            assert best_trial is not None and best_trial_eval is not None
            improvement = max(evals[i].loss - best_trial_eval.loss, 0.0)
            if best_trial_eval.loss < evals[i].loss:
                pop[i], evals[i] = best_trial, best_trial_eval
                stagnation[i] = 0
                operator_credit[best_op] = 0.85 * operator_credit[best_op] + 0.15 * (1.0 + improvement)
            else:
                stagnation[i] += 1
                operator_credit[best_op] = 0.97 * operator_credit[best_op] + 0.03

            if stagnation[i] >= max(4, iterations // 4):
                restart = initialize_population(objective, 1, rng, mi_guided=True)[0]
                if rng.random() < 0.5:
                    restart[:3] = 0.65 * pop[best_idx, :3] + 0.35 * restart[:3]
                else:
                    restart = lb + ub - pop[i]
                    restart[3:] = np.clip(restart[3:] + rng.normal(0.0, 0.15, objective.n_features), 0.0, 1.0)
                restart_eval = objective.evaluate(np.clip(restart, lb, ub))
                if restart_eval.loss < evals[i].loss or i != best_idx:
                    pop[i], evals[i] = np.clip(restart, lb, ub), restart_eval
                stagnation[i] = 0

        best_idx = int(np.argmin([ev.loss for ev in evals]))
        polished = polish_best(pop[best_idx], objective, rng, progress, span)
        polished_eval = objective.evaluate(polished)
        if polished_eval.loss < evals[best_idx].loss:
            pop[best_idx], evals[best_idx] = polished, polished_eval

        best_idx = int(np.argmin([ev.loss for ev in evals]))
        history.append(history_row(t, objective.eval_count, evals[best_idx]))

    best_idx = int(np.argmin([ev.loss for ev in evals]))
    return OptimizerResult(
        name="PBMSBAINGO" if precision_aware else "MSBAINGO",
        seed=seed,
        best_position=pop[best_idx].copy(),
        best_eval=evals[best_idx],
        history=history,
    )


def make_standard_ngo_trial(
    pop: np.ndarray,
    evals: list[EvalResult],
    i: int,
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    prey_idx = int(rng.choice(np.delete(np.arange(len(pop)), i)))
    prey = pop[prey_idx]
    integer_factor = rng.integers(1, 3, size=objective.dim)
    r = rng.random(objective.dim)
    if evals[prey_idx].loss < evals[i].loss:
        trial = pop[i] + r * (prey - integer_factor * pop[i])
    else:
        trial = pop[i] + r * (pop[i] - prey)
    if rng.random() < 0.5:
        local_r = 0.02 * (1.0 - progress)
        trial = trial + local_r * (2.0 * rng.random(objective.dim) - 1.0) * np.maximum(np.abs(pop[i]), 1e-3)
    return trial


def make_tis_cognitive_trial(
    pop: np.ndarray,
    i: int,
    best: np.ndarray,
    centroid: np.ndarray,
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    xi = pop[i]
    span = objective.ub - objective.lb
    thinking_weight = 0.35 + 0.45 * progress
    self_reflection = centroid + rng.random(objective.dim) * (best - xi)
    random_insight = rng.normal(0.0, 0.08 + 0.12 * (1.0 - progress), size=objective.dim) * span
    trial = (1.0 - thinking_weight) * xi + thinking_weight * self_reflection + random_insight
    if objective.n_features:
        best_mask = best[3:] >= 0.5
        copy = rng.random(objective.n_features) < (0.20 + 0.40 * progress)
        trial[3:][copy] = np.where(best_mask[copy], 0.85, 0.15)
    return trial


def make_tis_de_prey_attack_trial(
    pop: np.ndarray,
    evals: list[EvalResult],
    i: int,
    elite_indices: Iterable[int],
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    idx_pool = np.delete(np.arange(len(pop)), i)
    r1, r2 = rng.choice(idx_pool, size=2, replace=False)
    prey_idx = int(rng.choice(list(elite_indices)))
    prey = pop[prey_idx]
    f = 0.35 + 0.45 * (1.0 - progress)
    integer_factor = rng.integers(1, 3, size=objective.dim)
    if evals[prey_idx].loss < evals[i].loss:
        attack = pop[i] + rng.random(objective.dim) * (prey - integer_factor * pop[i])
    else:
        attack = pop[i] + rng.random(objective.dim) * (pop[i] - prey)
    return attack + f * rng.random(objective.dim) * (pop[int(r1)] - pop[int(r2)])


def make_tis_centroid_opposition_trial(
    xi: np.ndarray,
    centroid: np.ndarray,
    best: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    progress: float,
) -> np.ndarray:
    opposite = lb + ub - xi
    centroid_opposite = 2.0 * centroid - xi
    blend = 0.25 + 0.55 * progress
    trial = (1.0 - blend) * opposite + blend * centroid_opposite
    trial = 0.85 * trial + 0.15 * best
    reflection_noise = rng.normal(0.0, 0.03 + 0.08 * (1.0 - progress), size=len(xi)) * (ub - lb)
    return np.clip(trial + reflection_noise, lb, ub)


def make_ingo_range_trial(
    pop: np.ndarray,
    evals: list[EvalResult],
    i: int,
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    size = int(round(2 + (5 - 2) * progress))
    candidates = rng.choice(
        np.delete(np.arange(len(pop)), i),
        size=min(size, len(pop) - 1),
        replace=False,
    )
    prey_idx = min(candidates, key=lambda idx: evals[int(idx)].loss)
    prey = pop[int(prey_idx)]
    r = rng.random(objective.dim)
    integer_factor = rng.integers(1, 3, size=objective.dim)
    if evals[int(prey_idx)].loss < evals[i].loss:
        trial = pop[i] + r * (prey - integer_factor * pop[i])
    else:
        trial = pop[i] + r * (pop[i] - prey)
    r_scale = 0.001 + (0.05 - 0.001) * math.cos(math.pi * progress / 2.0)
    return trial + r_scale * (2.0 * rng.random(objective.dim) - 1.0) * (objective.ub - objective.lb)


def make_recall_archive_trial(
    pop: np.ndarray,
    i: int,
    recall_indices: Iterable[int],
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    teacher = pop[int(rng.choice(list(recall_indices)))]
    trial = pop[i].copy()
    span = objective.ub - objective.lb
    trial[:3] = (
        (0.45 + 0.35 * progress) * teacher[:3]
        + (0.55 - 0.35 * progress) * trial[:3]
        + rng.normal(0.0, 0.05 + 0.10 * (1.0 - progress), size=3) * span[:3]
    )
    teacher_mask = teacher[3:] >= 0.5
    swap = rng.random(objective.n_features) < (0.30 + 0.30 * progress)
    trial[3:][swap] = np.where(teacher_mask[swap], 0.88, 0.12)
    return trial


def make_precision_archive_trial(
    pop: np.ndarray,
    i: int,
    precision_indices: Iterable[int],
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
    progress: float,
) -> np.ndarray:
    teacher = pop[int(rng.choice(list(precision_indices)))]
    trial = pop[i].copy()
    span = objective.ub - objective.lb
    blend = 0.45 + 0.35 * progress
    trial[:3] = (
        blend * teacher[:3]
        + (1.0 - blend) * trial[:3]
        + rng.normal(0.0, 0.035 + 0.08 * (1.0 - progress), size=3) * span[:3]
    )
    trial[2] -= abs(rng.normal(0.0, 0.08 + 0.08 * (1.0 - progress)))
    teacher_mask = teacher[3:] >= 0.5
    copy = rng.random(objective.n_features) < (0.35 + 0.35 * progress)
    trial[3:][copy] = np.where(teacher_mask[copy], 0.82, 0.10)
    return trial


def make_fp_control_trial(
    xi: np.ndarray,
    best: np.ndarray,
    objective: BankruptcyKELMFitness,
    mi: np.ndarray,
    rng: np.random.Generator,
    progress: float,
) -> np.ndarray:
    trial = xi.copy()
    span = objective.ub - objective.lb
    trial[:3] = 0.70 * trial[:3] + 0.30 * best[:3]
    trial[:2] += rng.normal(0.0, 0.025 + 0.05 * (1.0 - progress), size=2) * span[:2]
    trial[2] -= rng.uniform(0.03, 0.18) * (1.0 + 0.5 * progress)

    selected = trial[3:] >= 0.5
    low_mi_cutoff = np.quantile(mi, 0.45)
    high_mi_cutoff = np.quantile(mi, 0.75)
    remove = selected & (mi <= low_mi_cutoff) & (rng.random(objective.n_features) < (0.12 + 0.10 * progress))
    add = (~selected) & (mi >= high_mi_cutoff) & (rng.random(objective.n_features) < (0.04 + 0.05 * (1.0 - progress)))
    trial[3:][remove] = rng.uniform(0.0, 0.30, size=int(remove.sum()))
    trial[3:][add] = rng.uniform(0.65, 1.0, size=int(add.sum()))
    return trial


def polish_best(
    best: np.ndarray,
    objective: BankruptcyKELMFitness,
    rng: np.random.Generator,
    progress: float,
    span: np.ndarray,
) -> np.ndarray:
    trial = best.copy()
    trial[:3] += rng.normal(0.0, 0.025 + 0.05 * (1.0 - progress), size=3) * span[:3]
    mi = objective.mi_scores / max(float(objective.mi_scores.max()), EPS)
    low_mi_selected = (trial[3:] >= 0.5) & (mi < np.quantile(mi, 0.35))
    high_mi_unselected = (trial[3:] < 0.5) & (mi > np.quantile(mi, 0.70))
    remove = low_mi_selected & (rng.random(objective.n_features) < 0.04)
    add = high_mi_unselected & (rng.random(objective.n_features) < 0.05)
    trial[3:][remove] = rng.uniform(0.0, 0.35, size=int(remove.sum()))
    trial[3:][add] = rng.uniform(0.65, 1.0, size=int(add.sum()))
    return np.clip(trial, objective.lb, objective.ub)


def make_baingo_prey_trial(
    pop: np.ndarray,
    evals: list[EvalResult],
    i: int,
    best: np.ndarray,
    progress: float,
    rng: np.random.Generator,
    objective: BankruptcyKELMFitness,
) -> np.ndarray:
    pop_size = len(pop)
    candidate_size = min(pop_size - 1, int(round(3 + (7 - 3) * (1.0 - progress))))
    candidate_idx = rng.choice(np.delete(np.arange(pop_size), i), size=candidate_size, replace=False)
    xi = pop[i]
    distances = np.linalg.norm(pop[candidate_idx] - xi, axis=1)
    max_dist = max(float(distances.max()), EPS)
    losses = np.array([evals[int(idx)].loss for idx in candidate_idx])
    recalls = np.array([evals[int(idx)].metrics["sensitivity"] for idx in candidate_idx])
    norm_loss = (losses - losses.min()) / max(float(losses.max() - losses.min()), EPS)
    norm_dist = distances / max_dist
    score = (
        (0.55 + 0.25 * progress) * norm_loss
        - (0.25 * (1.0 - progress)) * norm_dist
        - (0.20 + 0.15 * progress) * recalls
    )
    prey_idx = int(candidate_idx[int(np.argmin(score))])
    prey = pop[prey_idx]
    integer_factor = rng.integers(1, 3, size=objective.dim)
    r = rng.random(objective.dim)
    if evals[prey_idx].loss < evals[i].loss:
        trial = xi + r * (prey - integer_factor * xi) + 0.25 * progress * rng.random(objective.dim) * (best - xi)
    else:
        trial = xi + r * (xi - prey) + 0.35 * rng.random(objective.dim) * (best - xi)
    trial[:3] += rng.standard_cauchy(3).clip(-3.0, 3.0) * (0.04 + 0.08 * (1.0 - progress))
    return trial


def make_elite_differential_trial(
    pop: np.ndarray,
    i: int,
    best: np.ndarray,
    elite_indices: Iterable[int],
    rng: np.random.Generator,
    lb: np.ndarray,
    ub: np.ndarray,
    progress: float,
) -> np.ndarray:
    idx_pool = np.delete(np.arange(len(pop)), i)
    r1, r2 = rng.choice(idx_pool, size=2, replace=False)
    elite = pop[int(rng.choice(list(elite_indices)))]
    f = 0.35 + 0.35 * (1.0 - progress)
    trial = pop[i] + f * (elite - pop[i]) + f * rng.random(pop.shape[1]) * (pop[r1] - pop[r2])
    trial = 0.85 * trial + 0.15 * best if progress > 0.6 else trial
    return np.clip(trial, lb, ub)


def make_binary_recall_trial(
    xi: np.ndarray,
    best: np.ndarray,
    objective: BankruptcyKELMFitness,
    mi: np.ndarray,
    rng: np.random.Generator,
    progress: float,
) -> np.ndarray:
    trial = xi.copy()
    span = objective.ub - objective.lb
    local_sigma = 0.10 * (1.0 - progress) + 0.015
    trial[:3] += rng.normal(0.0, local_sigma, size=3) * span[:3]

    best_mask = best[3:] >= 0.5
    copy_prob = 0.20 + 0.55 * progress
    copy = rng.random(objective.n_features) < copy_prob
    trial[3:][copy] = np.where(best_mask[copy], 0.85, 0.15)

    selected = trial[3:] >= 0.5
    remove_prob = (0.04 + 0.10 * (1.0 - progress)) * (1.0 - mi)
    add_prob = (0.03 + 0.12 * (1.0 - progress)) * mi
    remove = selected & (rng.random(objective.n_features) < remove_prob)
    add = (~selected) & (rng.random(objective.n_features) < add_prob)
    trial[3:][remove] = rng.uniform(0.0, 0.35, size=int(remove.sum()))
    trial[3:][add] = rng.uniform(0.65, 1.0, size=int(add.sum()))
    return trial


def history_row(iteration: int, eval_count: int, ev: EvalResult) -> dict[str, float]:
    return {
        "iteration": float(iteration),
        "eval_count": float(eval_count),
        "loss": float(ev.loss),
        "gmean": float(ev.metrics["gmean"]),
        "sensitivity": float(ev.metrics["sensitivity"]),
        "specificity": float(ev.metrics["specificity"]),
        "mcc": float(ev.metrics["mcc"]),
        "triad_score": float(ev.metrics.get("triad_score", 0.0)),
        "f2": float(ev.metrics["f2"]),
        "ap": float(ev.metrics["ap"]),
        "feature_count": float(ev.metrics["feature_count"]),
    }


def run_one(
    data_path: Path,
    optimizer_name: str,
    seed: int,
    pop_size: int,
    iterations: int,
    landmarks: int,
    max_fit: int | None,
    max_val: int | None,
    feature_penalty: float,
    focus: str,
    ensemble_size: int,
    neg_pos_ratio: float,
    feature_mode: str,
    risk_feature: str,
    score_blend_alpha: float,
    auto_blend: bool,
) -> dict[str, float | str]:
    x, y = load_arff_dataset(data_path)
    x_fit, y_fit, x_val, y_val, x_test, y_test, mi = preprocess_splits(
        x,
        y,
        seed=seed,
        max_fit=max_fit,
        max_val=max_val,
        risk_feature=risk_feature,
    )
    rng = np.random.default_rng(seed + 101)
    objective = BankruptcyKELMFitness(
        x_fit,
        y_fit,
        x_val,
        y_val,
        mi,
        rng,
        n_landmarks=landmarks,
        feature_penalty=feature_penalty,
        focus=focus,
        ensemble_size=ensemble_size,
        neg_pos_ratio=neg_pos_ratio,
        feature_mode=feature_mode,
        force_last_feature=(risk_feature != "none"),
        score_blend_alpha=score_blend_alpha if risk_feature != "none" else -1.0,
        auto_blend=auto_blend and risk_feature != "none",
    )

    if optimizer_name == "ngo":
        opt = optimize_ngo(objective, pop_size, iterations, seed, improved=False)
    elif optimizer_name == "ingo":
        opt = optimize_ngo(objective, pop_size, iterations, seed, improved=True)
    elif optimizer_name == "tisngo":
        opt = optimize_tisngo(objective, pop_size, iterations, seed)
    elif optimizer_name == "baingo":
        opt = optimize_baingo(objective, pop_size, iterations, seed)
    elif optimizer_name == "msbaingo":
        opt = optimize_msbaingo(objective, pop_size, iterations, seed)
    elif optimizer_name == "pbmsbaingo":
        opt = optimize_msbaingo(objective, pop_size, iterations, seed, precision_aware=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    test_metrics = objective.test_best(opt.best_position, x_test, y_test)
    row: dict[str, float | str] = {
        "dataset": data_path.name,
        "optimizer": opt.name,
        "seed": seed,
        "ensemble_size": ensemble_size,
        "neg_pos_ratio": neg_pos_ratio,
        "feature_mode": feature_mode,
        "risk_feature": risk_feature,
        "score_blend_alpha": score_blend_alpha,
        "auto_blend": auto_blend,
        "focus": focus,
        "evals": float(objective.eval_count),
        "val_loss": opt.best_eval.loss,
        "val_gmean": opt.best_eval.metrics["gmean"],
        "val_sensitivity": opt.best_eval.metrics["sensitivity"],
        "val_specificity": opt.best_eval.metrics["specificity"],
        "val_f2": opt.best_eval.metrics["f2"],
        "val_ap": opt.best_eval.metrics["ap"],
    }
    for key, value in test_metrics.items():
        row[f"test_{key}"] = value
    row["selected_features"] = json.dumps(
        (np.flatnonzero(opt.best_eval.selected_features) + 1).astype(int).tolist()
    )
    return row


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize cost-sensitive reduced RBF-KELM for Polish bankruptcy prediction."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "03_data" / "01_raw" / "polish_bankruptcy",
    )
    parser.add_argument("--dataset", default="1year.arff", help="ARFF file name or 'all'.")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["ngo", "ingo", "baingo"],
        choices=["ngo", "ingo", "tisngo", "baingo", "msbaingo", "pbmsbaingo"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pop-size", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--landmarks", type=int, default=160)
    parser.add_argument("--max-fit", type=int, default=2800)
    parser.add_argument("--max-val", type=int, default=1400)
    parser.add_argument("--feature-penalty", type=float, default=0.02)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--neg-pos-ratio", type=float, default=3.0)
    parser.add_argument("--feature-mode", choices=["binary", "weighted"], default="binary")
    parser.add_argument("--risk-feature", choices=["none", "histgb"], default="none")
    parser.add_argument(
        "--score-blend-alpha",
        type=float,
        default=-1.0,
        help="KELM score weight when blending with a supervised risk feature; -1 disables blending.",
    )
    parser.add_argument(
        "--auto-blend",
        action="store_true",
        help="Select the KELM/HistGB score blend weight on the validation set.",
    )
    parser.add_argument(
        "--focus",
        choices=[
            "balanced",
            "recall",
            "f2",
            "risk",
            "precision",
            "auprc",
            "mcc",
            "mcc_guarded",
            "triad",
            "triage",
            "ranking",
        ],
        default="balanced",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "04_results" / "reproduction_check_polish.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "all":
        datasets = sorted(args.data_dir.glob("*.arff"))
    else:
        datasets = [args.data_dir / args.dataset]

    rows = []
    total = len(datasets) * len(args.optimizers) * len(args.seeds)
    done = 0
    for data_path in datasets:
        for seed in args.seeds:
            for optimizer in args.optimizers:
                done += 1
                print(
                    f"[{done}/{total}] {data_path.name} optimizer={optimizer} seed={seed}",
                    flush=True,
                )
                row = run_one(
                    data_path,
                    optimizer,
                    seed,
                    args.pop_size,
                    args.iterations,
                    args.landmarks,
                    args.max_fit,
                    args.max_val,
                    args.feature_penalty,
                    args.focus,
                    args.ensemble_size,
                    args.neg_pos_ratio,
                    args.feature_mode,
                    args.risk_feature,
                    args.score_blend_alpha,
                    args.auto_blend,
                )
                rows.append(row)
                print(
                    "  test "
                    f"gmean={row['test_gmean']:.4f} "
                    f"sens={row['test_sensitivity']:.4f} "
                    f"spec={row['test_specificity']:.4f} "
                    f"prec={row['test_precision']:.4f} "
                    f"f2={row['test_f2']:.4f} "
                    f"features={row['test_feature_count']:.0f}",
                    flush=True,
                )
                write_rows(args.output, rows)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
