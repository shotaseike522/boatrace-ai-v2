from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import (
    INDEX_TO_TRIFECTA,
    TRIFECTA_PERMUTATIONS,
    actual_trifecta_strings,
    normalize_probability_matrix,
    target_indices,
    top_prediction_strings,
)
from .models import ModelPrediction


def safe_model_column_name(model_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", model_name).strip("_")


@dataclass
class EvaluationResult:
    metrics: dict[str, object]
    calibration: pd.DataFrame
    details: pd.DataFrame
    errors: pd.DataFrame


def _rank_of_true_class(order: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(y_true), dtype=int)
    for idx, true_idx in enumerate(y_true):
        ranks[idx] = int(np.where(order[idx] == true_idx)[0][0]) + 1
    return ranks


def evaluate_prediction(
    test_df: pd.DataFrame,
    prediction: ModelPrediction,
    *,
    n_bins: int = 10,
) -> EvaluationResult:
    proba = normalize_probability_matrix(prediction.proba)
    y_true = target_indices(test_df)
    actual = actual_trifecta_strings(test_df).to_numpy()
    order = np.argsort(-proba, axis=1)
    top1_idx = order[:, 0]
    top1_labels = np.asarray([INDEX_TO_TRIFECTA[int(idx)] for idx in top1_idx])
    top1_parts = np.asarray([label.split("-") for label in top1_labels], dtype=int)
    actual_parts = test_df[["r1", "r2", "r3"]].to_numpy(dtype=int)

    true_rank = _rank_of_true_class(order, y_true)
    true_proba = proba[np.arange(len(test_df)), y_true]
    top1_proba = proba[np.arange(len(test_df)), top1_idx]
    exact_hit = top1_idx == y_true
    first_hit = top1_parts[:, 0] == actual_parts[:, 0]
    first2_hit = (top1_parts[:, :2] == actual_parts[:, :2]).all(axis=1)

    eps = 1e-15
    logloss = float(-np.mean(np.log(np.clip(true_proba, eps, 1.0))))
    brier = float(np.mean(np.sum(proba * proba, axis=1) - 2.0 * true_proba + 1.0))

    metrics = {
        "model": prediction.model_name,
        "model_type": prediction.model_type,
        "leakage_risk": prediction.leakage_risk,
        "n_races": len(test_df),
        "first_place_accuracy": float(first_hit.mean()),
        "first2_order_accuracy": float(first2_hit.mean()),
        "trifecta_top1_accuracy": float(exact_hit.mean()),
        "top3_contains_actual": float((true_rank <= 3).mean()),
        "top5_contains_actual": float((true_rank <= 5).mean()),
        "top10_contains_actual": float((true_rank <= 10).mean()),
        "mean_actual_rank": float(true_rank.mean()),
        "median_actual_rank": float(np.median(true_rank)),
        "logloss": logloss,
        "brier_score": brier,
        "notes": prediction.notes,
    }

    details = pd.DataFrame(
        {
            "jcd": test_df["jcd"].astype(str).to_numpy(),
            "r": test_df["r"].astype(int).to_numpy(),
            "actual_result": actual,
            "top1_prediction": top1_labels,
            "top5_predictions": top_prediction_strings(proba, top_n=5, with_probability=True),
            "actual_rank": true_rank,
            "actual_probability": true_proba,
            "top1_probability": top1_proba,
            "first_place_hit": first_hit,
            "first2_order_hit": first2_hit,
            "trifecta_top1_hit": exact_hit,
            "model": prediction.model_name,
            "leakage_risk": prediction.leakage_risk,
        }
    )

    calibration = calibration_table(
        prediction.model_name,
        top1_proba,
        exact_hit.astype(int),
        prediction.leakage_risk,
        n_bins=n_bins,
    )

    errors = details.loc[~details["trifecta_top1_hit"]].copy()
    errors["rank_bucket"] = pd.cut(
        errors["actual_rank"],
        bins=[0, 3, 5, 10, 30, 120],
        labels=["4_top3", "5_top5", "6_10", "11_30", "31_120"],
        include_lowest=True,
    )
    return EvaluationResult(metrics=metrics, calibration=calibration, details=details, errors=errors)


def calibration_table(
    model_name: str,
    predicted_probability: np.ndarray,
    observed_hit: np.ndarray,
    leakage_risk: bool,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = pd.cut(predicted_probability, bins=bins, include_lowest=True, right=True)
    frame = pd.DataFrame(
        {
            "bucket": bucket,
            "predicted_probability": predicted_probability,
            "observed_hit": observed_hit,
        }
    )
    rows = []
    for interval, group in frame.groupby("bucket", observed=True):
        rows.append(
            {
                "model": model_name,
                "leakage_risk": leakage_risk,
                "probability_type": "top1_trifecta_probability",
                "bin_low": float(interval.left),
                "bin_high": float(interval.right),
                "n": int(len(group)),
                "avg_predicted_probability": float(group["predicted_probability"].mean()),
                "observed_accuracy": float(group["observed_hit"].mean()),
                "calibration_error": float(group["predicted_probability"].mean() - group["observed_hit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def combine_prediction_details(test_df: pd.DataFrame, results: list[EvaluationResult]) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "jcd": test_df["jcd"].astype(str).to_numpy(),
            "r": test_df["r"].astype(int).to_numpy(),
            "actual_result": actual_trifecta_strings(test_df).to_numpy(),
        }
    )
    for result in results:
        model = safe_model_column_name(str(result.metrics["model"]))
        detail = result.details
        base[f"{model}_top1_prediction"] = detail["top1_prediction"].to_numpy()
        base[f"{model}_top5_predictions"] = detail["top5_predictions"].to_numpy()
        base[f"{model}_actual_rank"] = detail["actual_rank"].to_numpy()
        base[f"{model}_actual_probability"] = detail["actual_probability"].to_numpy()
        base[f"{model}_first_place_hit"] = detail["first_place_hit"].to_numpy()
        base[f"{model}_trifecta_top1_hit"] = detail["trifecta_top1_hit"].to_numpy()
    return base


def summarize_error_analysis(results: list[EvaluationResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        errors = result.errors.copy()
        if errors.empty:
            continue
        frames.append(errors)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

