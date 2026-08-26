from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external_validation"
RESULTS = EXT / "results"
BASE_PATH = (
    ROOT
    / "02_code"
    / "01_core_polish_model"
    / "bankruptcy_baingo_kelm.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bankruptcy_baingo_kelm", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import method implementation: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
EPS = 1e-12


@dataclass
class SplitData:
    name: str
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_search: np.ndarray
    y_search: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    design: str


@dataclass
class PreparedData:
    name: str
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_search: np.ndarray
    y_search: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    design: str


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_selection: list[dict[str, object]] = []

    seeds = [args.seed] if args.seed is not None else args.seeds
    for seed in seeds:
        for dataset in args.datasets:
            print(f"[dataset] {dataset} seed={seed}", flush=True)
            raw = load_dataset(dataset, args.split_seed)
            data = preprocess(raw)
            metrics, predictions, selection = run_dataset(
                data=data,
                seed=seed,
                pop_size=args.pop_size,
                iterations=args.iterations,
                landmarks=args.landmarks,
                bootstrap=args.bootstrap,
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            all_selection.extend(selection)
            pd.DataFrame(all_metrics).to_csv(
                output_dir / "pilot_metrics.csv", index=False, encoding="utf-8-sig"
            )
            pd.concat(all_predictions, ignore_index=True).to_csv(
                output_dir / "pilot_predictions.csv", index=False
            )
            pd.DataFrame(all_selection).to_csv(
                output_dir / "pilot_selection.csv", index=False, encoding="utf-8-sig"
            )

    print(f"Saved results under {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-controlled pilot validation for external bankruptcy datasets. "
            "Model/alpha selection, probability calibration, and final testing use "
            "separate partitions."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["taiwan", "us", "slovak"],
        default=["taiwan", "us", "slovak"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260723])
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260723,
        help=(
            "Fixed data-partition seed. Keep this constant when varying optimizer "
            "seeds so the locked test observations do not change."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Backward-compatible single-seed override.",
    )
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--landmarks", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS,
        help="Versioned output directory; use a new directory for formal runs.",
    )
    return parser.parse_args()


def load_dataset(name: str, seed: int) -> SplitData:
    if name == "taiwan":
        frame = pd.read_csv(EXT / "clean" / "taiwan_uci.csv")
        y = frame.pop("target").to_numpy(dtype=int)
        x = frame.to_numpy(dtype=float)
        idx = np.arange(len(y))
        inner_idx, test_idx = train_test_split(
            idx, test_size=0.30, stratify=y, random_state=seed
        )
        fit_idx, hold_idx = train_test_split(
            inner_idx,
            test_size=0.34,
            stratify=y[inner_idx],
            random_state=seed + 11,
        )
        search_idx, cal_idx = train_test_split(
            hold_idx,
            test_size=0.50,
            stratify=y[hold_idx],
            random_state=seed + 17,
        )
        return SplitData(
            name="Taiwan_UCI",
            x_fit=x[fit_idx],
            y_fit=y[fit_idx],
            x_search=x[search_idx],
            y_search=y[search_idx],
            x_cal=x[cal_idx],
            y_cal=y[cal_idx],
            x_test=x[test_idx],
            y_test=y[test_idx],
            feature_names=list(frame.columns),
            design=(
                "locked stratified 30% test; remaining observations split into "
                "fit/search/calibration partitions"
            ),
        )
    if name == "us":
        train = pd.read_csv(EXT / "clean" / "us_official_train.csv")
        validation = pd.read_csv(EXT / "clean" / "us_official_validation.csv")
        test = pd.read_csv(EXT / "clean" / "us_official_test.csv")
        feature_names = [
            c for c in train.columns if c not in {"cik", "fyear", "target"}
        ]
        val_idx = np.arange(len(validation))
        search_idx, cal_idx = train_test_split(
            val_idx,
            test_size=0.50,
            stratify=validation["target"],
            random_state=seed + 23,
        )
        return SplitData(
            name="US_official_windows",
            x_fit=train[feature_names].to_numpy(dtype=float),
            y_fit=train["target"].to_numpy(dtype=int),
            x_search=validation.iloc[search_idx][feature_names].to_numpy(dtype=float),
            y_search=validation.iloc[search_idx]["target"].to_numpy(dtype=int),
            x_cal=validation.iloc[cal_idx][feature_names].to_numpy(dtype=float),
            y_cal=validation.iloc[cal_idx]["target"].to_numpy(dtype=int),
            x_test=test[feature_names].to_numpy(dtype=float),
            y_test=test["target"].to_numpy(dtype=int),
            feature_names=feature_names,
            design=(
                "official disjoint company splits; official validation divided into "
                "search and calibration halves; official test untouched"
            ),
        )
    if name == "slovak":
        frames = {
            year: pd.read_csv(EXT / "clean" / f"slovak_pooled_{year}.csv")
            for year in [2013, 2014, 2015, 2016]
        }
        feature_names = [f"V{i}" for i in range(1, 64)]
        fit = pd.concat([frames[2013], frames[2014]], ignore_index=True)
        validation = frames[2015]
        test = frames[2016]
        val_idx = np.arange(len(validation))
        search_idx, cal_idx = train_test_split(
            val_idx,
            test_size=0.50,
            stratify=validation["target"],
            random_state=seed + 29,
        )
        return SplitData(
            name="Slovak_pooled_temporal",
            x_fit=fit[feature_names].to_numpy(dtype=float),
            y_fit=fit["target"].to_numpy(dtype=int),
            x_search=validation.iloc[search_idx][feature_names].to_numpy(dtype=float),
            y_search=validation.iloc[search_idx]["target"].to_numpy(dtype=int),
            x_cal=validation.iloc[cal_idx][feature_names].to_numpy(dtype=float),
            y_cal=validation.iloc[cal_idx]["target"].to_numpy(dtype=int),
            x_test=test[feature_names].to_numpy(dtype=float),
            y_test=test["target"].to_numpy(dtype=int),
            feature_names=feature_names,
            design=(
                "four industries pooled; 2013-2014 fit, 2015 search/calibration, "
                "2016 untouched temporal test; cross-year entity overlap is unknown"
            ),
        )
    raise ValueError(name)


def preprocess(data: SplitData) -> PreparedData:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    x_fit = imputer.fit_transform(data.x_fit)
    transformed = [
        imputer.transform(x)
        for x in [data.x_search, data.x_cal, data.x_test]
    ]
    x_fit = np.clip(scaler.fit_transform(x_fit), -8.0, 8.0)
    x_search, x_cal, x_test = [
        np.clip(scaler.transform(x), -8.0, 8.0) for x in transformed
    ]
    return PreparedData(
        name=data.name,
        x_fit=x_fit,
        y_fit=data.y_fit,
        x_search=x_search,
        y_search=data.y_search,
        x_cal=x_cal,
        y_cal=data.y_cal,
        x_test=x_test,
        y_test=data.y_test,
        feature_names=data.feature_names,
        design=data.design,
    )


def run_dataset(
    data: PreparedData,
    seed: int,
    pop_size: int,
    iterations: int,
    landmarks: int,
    bootstrap: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    risk_fit, risk_search, risk_cal, risk_test = histgb_risk_scores(data, seed)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    selection: list[dict[str, object]] = []

    hist_metrics, hist_pred = evaluate_scores(
        data=data,
        method="HistGB",
        search_scores=risk_search,
        cal_scores=risk_cal,
        test_scores=risk_test,
        seed=seed,
        bootstrap=bootstrap,
        selection_note="No metaheuristic; threshold and Platt calibration use calibration partition.",
    )
    metrics.extend(hist_metrics)
    predictions.append(hist_pred)

    plain = optimize_kelm(
        data=data,
        x_fit=data.x_fit,
        x_search=data.x_search,
        risk_feature=False,
        auto_blend=False,
        seed=seed + 101,
        pop_size=pop_size,
        iterations=iterations,
        landmarks=landmarks,
        optimizer="pbmsbaingo",
    )
    plain_search, plain_cal, plain_test = score_kelm(
        plain, data.x_search, data.x_cal, data.x_test
    )
    rows, pred = evaluate_scores(
        data,
        "KELM_PBMSBAINGO",
        plain_search,
        plain_cal,
        plain_test,
        seed + 101,
        bootstrap,
        selection_note="Plain KELM; hyperparameters/features selected only on search partition.",
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(selection_row(data, "KELM_PBMSBAINGO", plain))

    fusion_alpha = select_blend_alpha(
        data.y_search, plain_search, risk_search
    )
    rows, pred = evaluate_scores(
        data,
        "KELM_score_fusion",
        blend(plain_search, risk_search, fusion_alpha),
        blend(plain_cal, risk_cal, fusion_alpha),
        blend(plain_test, risk_test, fusion_alpha),
        seed + 103,
        bootstrap,
        selection_note=(
            f"Plain KELM plus HistGB score-level fusion; alpha={fusion_alpha:.2f} "
            "selected only on search partition."
        ),
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(
        {
            "dataset": data.name,
            "seed": seed,
            "method": "KELM_score_fusion",
            "optimizer": "reuses_plain_KELM",
            "evals": 0,
            "blend_alpha": fusion_alpha,
            "selected_features": json.dumps(
                plain["selected_features"], ensure_ascii=False
            ),
        }
    )

    x_fit_aug = np.c_[data.x_fit, risk_fit]
    x_search_aug = np.c_[data.x_search, risk_search]
    x_cal_aug = np.c_[data.x_cal, risk_cal]
    x_test_aug = np.c_[data.x_test, risk_test]
    risk_kelm = optimize_kelm(
        data=data,
        x_fit=x_fit_aug,
        x_search=x_search_aug,
        risk_feature=True,
        auto_blend=False,
        seed=seed + 211,
        pop_size=pop_size,
        iterations=iterations,
        landmarks=landmarks,
        optimizer="pbmsbaingo",
    )
    risk_search_k, risk_cal_k, risk_test_k = score_kelm(
        risk_kelm, x_search_aug, x_cal_aug, x_test_aug
    )
    rows, pred = evaluate_scores(
        data,
        "KELM_risk_feature",
        risk_search_k,
        risk_cal_k,
        risk_test_k,
        seed + 211,
        bootstrap,
        selection_note=(
            "OOF HistGB risk is a forced KELM input; no score-level fusion."
        ),
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(selection_row(data, "KELM_risk_feature", risk_kelm))

    full = optimize_kelm(
        data=data,
        x_fit=x_fit_aug,
        x_search=x_search_aug,
        risk_feature=True,
        auto_blend=True,
        seed=seed + 307,
        pop_size=pop_size,
        iterations=iterations,
        landmarks=landmarks,
        optimizer="pbmsbaingo",
    )
    full_search_k, full_cal_k, full_test_k = score_kelm(
        full, x_search_aug, x_cal_aug, x_test_aug
    )
    full_alpha = float(full["blend_alpha"])
    full_search_scores = blend(full_search_k, risk_search, full_alpha)
    full_cal_scores = blend(full_cal_k, risk_cal, full_alpha)
    full_test_scores = blend(full_test_k, risk_test, full_alpha)
    rows, pred = evaluate_scores(
        data,
        "Full_PBMSBAINGO",
        full_search_scores,
        full_cal_scores,
        full_test_scores,
        seed + 307,
        bootstrap,
        selection_note=(
            f"Risk feature plus score fusion; alpha={full_alpha:.2f} selected "
            "only on search partition."
        ),
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(selection_row(data, "Full_PBMSBAINGO", full))

    gate = select_uncertainty_gated_fusion(
        data.y_search,
        simple_scores=risk_search_k,
        fused_scores=full_search_scores,
        seed=seed + 353,
        n_boot=100,
        z=1.0,
    )
    if gate["use_fusion"]:
        gated_search, gated_cal, gated_test = (
            full_search_scores,
            full_cal_scores,
            full_test_scores,
        )
        gated_source = "Full_PBMSBAINGO"
    else:
        gated_search, gated_cal, gated_test = (
            risk_search_k,
            risk_cal_k,
            risk_test_k,
        )
        gated_source = "KELM_risk_feature"
    rows, pred = evaluate_scores(
        data,
        "Uncertainty_gated_fusion",
        gated_search,
        gated_cal,
        gated_test,
        seed + 353,
        bootstrap,
        selection_note=(
            f"One-standard-error paired-bootstrap gate selected {gated_source}; "
            f"search utility difference={gate['paired_difference']:.4f}, "
            f"required margin={gate['required_margin']:.4f}."
        ),
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(
        {
            "dataset": data.name,
            "seed": seed,
            "method": "Uncertainty_gated_fusion",
            "optimizer": "architecture_gate",
            "evals": 0,
            "val_loss": np.nan,
            "val_triad": np.nan,
            "val_ap": np.nan,
            "C": np.nan,
            "gamma": np.nan,
            "cost_pos": np.nan,
            "blend_alpha": full_alpha if gate["use_fusion"] else -1.0,
            "selected_feature_count": np.nan,
            "selected_features": "[]",
            "gate_selected": gated_source,
            "gate_utility_simple": gate["utility_simple"],
            "gate_utility_fused": gate["utility_fused"],
            "gate_difference": gate["paired_difference"],
            "gate_standard_error": gate["bootstrap_standard_error"],
            "gate_required_margin": gate["required_margin"],
        }
    )

    random_full = optimize_kelm(
        data=data,
        x_fit=x_fit_aug,
        x_search=x_search_aug,
        risk_feature=True,
        auto_blend=True,
        seed=seed + 401,
        pop_size=pop_size,
        iterations=iterations,
        landmarks=landmarks,
        optimizer="random",
        random_budget=int(full["evals"]),
    )
    random_search_k, random_cal_k, random_test_k = score_kelm(
        random_full, x_search_aug, x_cal_aug, x_test_aug
    )
    random_alpha = float(random_full["blend_alpha"])
    rows, pred = evaluate_scores(
        data,
        "Full_random_search",
        blend(random_search_k, risk_search, random_alpha),
        blend(random_cal_k, risk_cal, random_alpha),
        blend(random_test_k, risk_test, random_alpha),
        seed + 401,
        bootstrap,
        selection_note=(
            f"Same full architecture and {int(full['evals'])} evaluations as "
            f"PBMSBAINGO; alpha={random_alpha:.2f}."
        ),
    )
    metrics.extend(rows)
    predictions.append(pred)
    selection.append(selection_row(data, "Full_random_search", random_full))
    for row in metrics:
        row["model_seed"] = row.get("seed")
        row["seed"] = seed
    for frame in predictions:
        frame["model_seed"] = frame["seed"]
        frame["seed"] = seed
    for row in selection:
        row["model_seed"] = row.get("seed")
        row["seed"] = seed
    return metrics, predictions, selection


def histgb_risk_scores(
    data: PreparedData, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_class = int(np.bincount(data.y_fit).min())
    n_splits = max(2, min(5, min_class))
    oof = np.zeros(len(data.y_fit), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 701)
    for fold, (train_idx, hold_idx) in enumerate(cv.split(data.x_fit, data.y_fit)):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=seed + 701 + fold,
        )
        model.fit(data.x_fit[train_idx], data.y_fit[train_idx])
        oof[hold_idx] = model.predict_proba(data.x_fit[hold_idx])[:, 1]
    final = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=seed + 709,
    )
    final.fit(data.x_fit, data.y_fit)
    return (
        oof,
        final.predict_proba(data.x_search)[:, 1],
        final.predict_proba(data.x_cal)[:, 1],
        final.predict_proba(data.x_test)[:, 1],
    )


def optimize_kelm(
    data: PreparedData,
    x_fit: np.ndarray,
    x_search: np.ndarray,
    risk_feature: bool,
    auto_blend: bool,
    seed: int,
    pop_size: int,
    iterations: int,
    landmarks: int,
    optimizer: str,
    random_budget: int | None = None,
) -> dict[str, object]:
    mi = mutual_info_classif(
        x_fit, data.y_fit, discrete_features=False, random_state=seed
    )
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    if float(mi.max()) > 0:
        mi = mi / float(mi.max())
    objective = BASE.BankruptcyKELMFitness(
        x_fit=x_fit,
        y_fit=data.y_fit,
        x_val=x_search,
        y_val=data.y_search,
        mi_scores=mi,
        rng=np.random.default_rng(seed + 7),
        n_landmarks=min(landmarks, len(x_fit)),
        feature_penalty=0.02,
        min_features=4,
        focus="triad",
        ensemble_size=1,
        neg_pos_ratio=3.0,
        feature_mode="binary",
        force_last_feature=risk_feature,
        score_blend_alpha=-1.0,
        auto_blend=auto_blend,
    )
    if optimizer == "pbmsbaingo":
        opt = BASE.optimize_msbaingo(
            objective, pop_size, iterations, seed, precision_aware=True
        )
        best_position = opt.best_position
        best_eval = opt.best_eval
        optimizer_name = opt.name
    elif optimizer == "random":
        if random_budget is None:
            raise ValueError("random_budget is required")
        rng = np.random.default_rng(seed)
        best_position = None
        best_eval = None
        for _ in range(random_budget):
            position = objective.lb + rng.random(objective.dim) * (
                objective.ub - objective.lb
            )
            ev = objective.evaluate(position)
            if best_eval is None or ev.loss < best_eval.loss:
                best_position = position.copy()
                best_eval = ev
        if best_position is None or best_eval is None:
            raise RuntimeError("Random search produced no evaluation.")
        optimizer_name = "Random"
    else:
        raise ValueError(optimizer)

    c_value, gamma, cost_pos, mask, feature_weights = objective.decode(best_position)
    selected_scale = np.sqrt(np.maximum(feature_weights[mask], EPS))
    model = BASE.fit_reduced_rbf_kelm(
        x_fit[:, mask] * selected_scale,
        data.y_fit,
        objective.centers_all[:, mask] * selected_scale,
        c_value,
        gamma,
        cost_pos,
    )
    return {
        "model": model,
        "mask": mask,
        "selected_scale": selected_scale,
        "selected_features": (np.flatnonzero(mask) + 1).astype(int).tolist(),
        "C": c_value,
        "gamma": gamma,
        "cost_pos": cost_pos,
        "blend_alpha": float(best_eval.params.get("blend_alpha", -1.0)),
        "val_loss": float(best_eval.loss),
        "val_triad": float(best_eval.metrics["triad_score"]),
        "val_ap": float(best_eval.metrics["ap"]),
        "evals": int(objective.eval_count),
        "optimizer": optimizer_name,
        "seed": seed,
    }


def score_kelm(
    fitted: dict[str, object],
    x_search: np.ndarray,
    x_cal: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(fitted["mask"], dtype=bool)
    scale = np.asarray(fitted["selected_scale"], dtype=float)
    model = fitted["model"]
    return tuple(
        BASE.predict_reduced_rbf_kelm(x[:, mask] * scale, model)
        for x in [x_search, x_cal, x_test]
    )


def blend(
    kelm_scores: np.ndarray, risk_scores: np.ndarray, alpha: float
) -> np.ndarray:
    kelm_prob = 1.0 / (
        1.0 + np.exp(-np.clip(np.asarray(kelm_scores), -50.0, 50.0))
    )
    return (
        float(alpha) * kelm_prob
        + (1.0 - float(alpha)) * np.asarray(risk_scores)
    )


def select_blend_alpha(
    y: np.ndarray, kelm_scores: np.ndarray, risk_scores: np.ndarray
) -> float:
    _, threshold_weights = BASE.focus_weights("triad")
    best_alpha = 0.0
    best_utility = -np.inf
    for alpha in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
        scores = blend(kelm_scores, risk_scores, alpha)
        _, metrics = BASE.select_threshold(y, scores, threshold_weights)
        utility = float(metrics["threshold_utility"])
        if utility > best_utility:
            best_alpha = alpha
            best_utility = utility
    return float(best_alpha)


def select_uncertainty_gated_fusion(
    y: np.ndarray,
    simple_scores: np.ndarray,
    fused_scores: np.ndarray,
    seed: int,
    n_boot: int = 100,
    z: float = 1.0,
) -> dict[str, float | bool]:
    """Select fusion only when paired search improvement exceeds one SE."""
    y = np.asarray(y, dtype=int)
    simple_scores = np.asarray(simple_scores, dtype=float)
    fused_scores = np.asarray(fused_scores, dtype=float)
    if not (len(y) == len(simple_scores) == len(fused_scores)):
        raise ValueError("Fusion gate inputs must be paired.")
    if np.unique(y).size != 2:
        raise ValueError("Fusion gate requires both classes.")
    _, weights = BASE.focus_weights("triad")

    def utility(index: np.ndarray, scores: np.ndarray) -> float:
        _, metrics = BASE.select_threshold(y[index], scores[index], weights)
        return float(metrics["threshold_utility"])

    full_index = np.arange(len(y))
    utility_simple = utility(full_index, simple_scores)
    utility_fused = utility(full_index, fused_scores)
    difference = utility_fused - utility_simple
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    bootstrap_difference = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        index = np.r_[
            rng.choice(neg, size=len(neg), replace=True),
            rng.choice(pos, size=len(pos), replace=True),
        ]
        bootstrap_difference[b] = utility(index, fused_scores) - utility(
            index, simple_scores
        )
    standard_error = float(np.std(bootstrap_difference, ddof=1))
    required_margin = float(z * standard_error)
    return {
        "use_fusion": bool(difference > required_margin),
        "utility_simple": utility_simple,
        "utility_fused": utility_fused,
        "paired_difference": float(difference),
        "bootstrap_standard_error": standard_error,
        "required_margin": required_margin,
    }


def fast_select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Equivalent threshold scan with direct confusion-matrix arithmetic.

    The manuscript implementation calls several sklearn metric functions for
    every candidate threshold. This version preserves its candidate grid and
    utility/constraint rules but removes that avoidable pilot-run overhead.
    """
    if weights is None:
        _, weights = BASE.focus_weights("balanced")
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    quantiles = np.r_[
        np.linspace(0.35, 0.98, 80),
        np.linspace(0.981, 0.9995, 35),
    ]
    thresholds = np.unique(np.quantile(s, quantiles))
    thresholds = np.r_[thresholds, [float(s.min()) - EPS, 0.0, float(s.max()) + EPS]]
    min_sens = weights.get("_min_sensitivity", 0.0)
    min_spec = weights.get("_min_specificity", 0.0)
    min_tp = weights.get("_min_tp", 0.0)
    max_ppr = weights.get("_max_predicted_positive_rate", 1.0)
    penalty = weights.get("_constraint_penalty", 0.0)
    best: tuple[float, float, dict[str, float]] | None = None
    feasible_best: tuple[float, float, dict[str, float]] | None = None
    for threshold in thresholds:
        pred = s >= threshold
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        tn = float(np.sum(~pred & (y == 0)))
        fn = float(np.sum(~pred & (y == 1)))
        sensitivity = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, EPS)
        f2 = 5.0 * precision * sensitivity / max(4.0 * precision + sensitivity, EPS)
        f05 = 1.25 * precision * sensitivity / max(0.25 * precision + sensitivity, EPS)
        denominator = math.sqrt(
            max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0)
        )
        mcc = ((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0
        gmean = math.sqrt(max(sensitivity * specificity, 0.0))
        ppr = (tp + fp) / max(len(y), 1)
        metrics = {
            "gmean": gmean,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "f05": f05,
            "f2": f2,
            "mcc": mcc,
            "triad_score": max(max(mcc, 0.0) * sensitivity * specificity, 0.0)
            ** (1.0 / 3.0),
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "fpr": fp / max(fp + tn, 1.0),
            "predicted_positive_rate": ppr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        utility = sum(
            weight * metrics.get(metric, 0.0)
            for metric, weight in weights.items()
            if not metric.startswith("_") and metric != "ap"
        )
        utility -= penalty * max(0.0, min_sens - sensitivity)
        utility -= penalty * max(0.0, min_spec - specificity)
        utility -= penalty * max(0.0, ppr - max_ppr)
        candidate = (float(utility), float(threshold), metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate
        feasible = (
            sensitivity >= min_sens
            and specificity >= min_spec
            and tp >= min_tp
            and ppr <= max_ppr
        )
        if feasible and (feasible_best is None or candidate[0] > feasible_best[0]):
            feasible_best = candidate
    chosen = feasible_best if feasible_best is not None else best
    if chosen is None:
        raise RuntimeError("No threshold candidate.")
    chosen[2]["threshold_utility"] = chosen[0]
    return chosen[1], chosen[2]


def evaluate_scores(
    data: PreparedData,
    method: str,
    search_scores: np.ndarray,
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    seed: int,
    bootstrap: int,
    selection_note: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    del search_scores  # Selection was completed upstream; calibration remains separate.
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(np.asarray(cal_scores).reshape(-1, 1), data.y_cal)
    cal_prob = calibrator.predict_proba(
        np.asarray(cal_scores).reshape(-1, 1)
    )[:, 1]
    test_prob = calibrator.predict_proba(
        np.asarray(test_scores).reshape(-1, 1)
    )[:, 1]

    _, threshold_weights = BASE.focus_weights("triad")
    triad_threshold, _ = BASE.select_threshold(
        data.y_cal, cal_prob, threshold_weights
    )
    thresholds = {"triad": float(triad_threshold)}
    for ratio in [1, 2, 5, 10, 20]:
        thresholds[f"cost_{ratio}"] = select_cost_threshold(
            data.y_cal, cal_prob, false_negative_cost=float(ratio)
        )

    rows = []
    for policy, threshold in thresholds.items():
        point = threshold_metrics(data.y_test, test_prob, threshold)
        cost_ratio = (
            float(policy.split("_", 1)[1])
            if policy.startswith("cost_")
            else np.nan
        )
        if np.isfinite(cost_ratio):
            total_cost = cost_ratio * point["fn"] + point["fp"]
            point["relative_cost_ratio"] = cost_ratio
            point["cost_per_observation"] = total_cost / max(len(data.y_test), 1)
            point["normalized_cost"] = total_cost / max(
                cost_ratio * float(np.sum(data.y_test == 1))
                + float(np.sum(data.y_test == 0)),
                1.0,
            )
        else:
            point["relative_cost_ratio"] = np.nan
            point["cost_per_observation"] = np.nan
            point["normalized_cost"] = np.nan
        ci = stratified_bootstrap_ci(
            data.y_test,
            test_prob,
            threshold,
            n_boot=bootstrap,
            seed=seed + sum(ord(c) for c in policy),
        )
        row: dict[str, object] = {
            "dataset": data.name,
            "seed": seed,
            "method": method,
            "policy": policy,
            "test_n": len(data.y_test),
            "test_positive": int(data.y_test.sum()),
            "threshold": threshold,
            "design": data.design,
            "selection_note": selection_note,
            "calibration": "Platt_logistic_on_separate_calibration_partition",
        }
        row.update(point)
        row.update(ci)
        rows.append(row)

    pred = pd.DataFrame(
        {
            "dataset": data.name,
            "seed": seed,
            "method": method,
            "row_id": np.arange(len(data.y_test)),
            "target": data.y_test,
            "raw_score": np.asarray(test_scores),
            "calibrated_probability": test_prob,
            "triad_threshold": triad_threshold,
        }
    )
    return rows, pred


def select_cost_threshold(
    y_true: np.ndarray, probability: np.ndarray, false_negative_cost: float
) -> float:
    candidates = np.unique(
        np.r_[
            np.quantile(probability, np.linspace(0.0, 1.0, 301)),
            [0.0, 0.5, 1.0],
        ]
    )
    best_threshold = float(candidates[0])
    best_key = (np.inf, -np.inf)
    for threshold in candidates:
        metrics = BASE.classification_metrics(
            y_true, probability >= threshold
        )
        normalized_cost = (
            false_negative_cost * metrics["fn"] + metrics["fp"]
        ) / max(len(y_true), 1)
        key = (normalized_cost, -metrics["mcc"])
        if key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def threshold_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    metrics = BASE.classification_metrics(y_true, probability >= threshold)
    metrics["ap"] = BASE.safe_average_precision(y_true, probability)
    metrics["auc"] = BASE.safe_roc_auc(y_true, probability)
    metrics["brier"] = float(brier_score_loss(y_true, probability))
    metrics["log_loss"] = float(
        log_loss(y_true, np.clip(probability, EPS, 1.0 - EPS), labels=[0, 1])
    )
    metrics["ece_10"] = expected_calibration_error(y_true, probability, 10)
    metrics["calibration_intercept"], metrics["calibration_slope"] = (
        calibration_intercept_slope(y_true, probability)
    )
    return metrics


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges, right=True) - 1, 0, bins - 1)
    error = 0.0
    for bin_idx in range(bins):
        mask = index == bin_idx
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(probability[mask]))
            )
    return float(error)


def calibration_intercept_slope(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y_true == 0)
    pos = np.flatnonzero(y_true == 1)
    keys = [
        "sensitivity",
        "specificity",
        "precision",
        "f2",
        "mcc",
        "triad_score",
        "ap",
        "auc",
        "brier",
    ]
    samples = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = np.r_[
            rng.choice(neg, size=len(neg), replace=True),
            rng.choice(pos, size=len(pos), replace=True),
        ]
        boot_y = y_true[idx]
        boot_p = probability[idx]
        values = BASE.classification_metrics(boot_y, boot_p >= threshold)
        values["ap"] = BASE.safe_average_precision(boot_y, boot_p)
        values["auc"] = BASE.safe_roc_auc(boot_y, boot_p)
        values["brier"] = float(brier_score_loss(boot_y, boot_p))
        for key in keys:
            samples[key].append(float(values[key]))
    output = {}
    for key in keys:
        low, high = np.quantile(samples[key], [0.025, 0.975])
        output[f"{key}_ci_low"] = float(low)
        output[f"{key}_ci_high"] = float(high)
    return output


def selection_row(
    data: PreparedData, method: str, fitted: dict[str, object]
) -> dict[str, object]:
    return {
        "dataset": data.name,
        "seed": fitted["seed"],
        "method": method,
        "optimizer": fitted["optimizer"],
        "evals": fitted["evals"],
        "val_loss": fitted["val_loss"],
        "val_triad": fitted["val_triad"],
        "val_ap": fitted["val_ap"],
        "C": fitted["C"],
        "gamma": fitted["gamma"],
        "cost_pos": fitted["cost_pos"],
        "blend_alpha": fitted["blend_alpha"],
        "selected_feature_count": len(fitted["selected_features"]),
        "selected_features": json.dumps(
            fitted["selected_features"], ensure_ascii=False
        ),
    }


BASE.select_threshold = fast_select_threshold


if __name__ == "__main__":
    main()
