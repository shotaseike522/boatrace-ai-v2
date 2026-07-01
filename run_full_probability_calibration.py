from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boat_model.data_loader import load_race_data
from boat_model.features import INDEX_TO_TRIFECTA, add_basic_features, target_indices
from run_outer_in_23_boost_analysis import (
    N_TICKETS,
    TICKET_BOATS,
    actual_rank_fast,
    apply_outer_boost,
    brier_multiclass,
    fit_log_ticket_correction,
    split_past_by_row_order,
)


DEFAULT_PAST = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_1_past_201307_202602.csv"
DEFAULT_RECENT = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_2_recent_202603_202605.csv"

MODEL_RAW = "raw_direct120"
MODEL_JCDR = "direct120_jcd_r_top20_rerank"
MODEL_OUTER = "outer_in_23_boost_by_outer_winrate_score0.40_clip0.40_top20"
MODEL_NAMES = [MODEL_RAW, MODEL_JCDR, MODEL_OUTER]

K_CANDIDATES = [300, 500, 1000, 3000]
MIN_COUNT_CANDIDATES = [300, 500, 1000]

PROB_BUCKETS = [
    ("prob_0_0_25", 0.0000, 0.0025, "0-0.25%"),
    ("prob_0_25_0_5", 0.0025, 0.0050, "0.25-0.5%"),
    ("prob_0_5_1", 0.0050, 0.0100, "0.5-1%"),
    ("prob_1_1_5", 0.0100, 0.0150, "1-1.5%"),
    ("prob_1_5_2", 0.0150, 0.0200, "1.5-2%"),
    ("prob_2_3", 0.0200, 0.0300, "2-3%"),
    ("prob_3_5", 0.0300, 0.0500, "3-5%"),
    ("prob_5_7", 0.0500, 0.0700, "5-7%"),
    ("prob_7_10", 0.0700, 0.1000, "7-10%"),
    ("prob_10_12", 0.1000, 0.1200, "10-12%"),
    ("prob_12_15", 0.1200, 0.1500, "12-15%"),
    ("prob_15_plus", 0.1500, np.inf, "15%+"),
]
PROB_EDGES = np.asarray([b[2] for b in PROB_BUCKETS[:-1]], dtype=np.float32)
PROB_LABELS = [b[0] for b in PROB_BUCKETS]
PROB_DISPLAY = {b[0]: b[3] for b in PROB_BUCKETS}

RANK_BUCKETS = [
    ("rank1", 1, 1),
    ("rank2_3", 2, 3),
    ("rank4_5", 4, 5),
    ("rank6_10", 6, 10),
    ("rank11_20", 11, 20),
    ("rank21_50", 21, 50),
    ("rank51_120", 51, 120),
]
RANK_LABELS = [b[0] for b in RANK_BUCKETS]

PATTERN_LABELS = [
    "1_head_outer",
    "123_only",
    "1_head",
    "outer_head",
    "outer_in_23",
    "1_second",
    "1_third",
    "1_absent",
]

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full 120-ticket probability calibration for EV calculations.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST)
    parser.add_argument("--recent-csv", default=DEFAULT_RECENT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split-a-size", type=int, default=200_000)
    parser.add_argument("--split-b-size", type=int, default=200_000)
    return parser.parse_args()


def load_featured(path: str, *, require_registrations: bool, name: str) -> pd.DataFrame:
    raw, _ = load_race_data(path, require_registrations=require_registrations, name=name)
    return add_basic_features(raw).reset_index(drop=True)


def probability_bucket_indices(proba: np.ndarray) -> np.ndarray:
    return np.searchsorted(PROB_EDGES, proba, side="right").astype(np.int8)


def rank_matrix(proba: np.ndarray) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(len(proba))[:, None], order] = np.arange(1, N_TICKETS + 1, dtype=np.int16)
    return ranks


def rank_bucket_indices(ranks: np.ndarray) -> np.ndarray:
    out = np.empty_like(ranks, dtype=np.int8)
    out[ranks == 1] = 0
    out[(ranks >= 2) & (ranks <= 3)] = 1
    out[(ranks >= 4) & (ranks <= 5)] = 2
    out[(ranks >= 6) & (ranks <= 10)] = 3
    out[(ranks >= 11) & (ranks <= 20)] = 4
    out[(ranks >= 21) & (ranks <= 50)] = 5
    out[ranks >= 51] = 6
    return out


def ticket_pattern_indices() -> np.ndarray:
    first = TICKET_BOATS[:, 0]
    second = TICKET_BOATS[:, 1]
    third = TICKET_BOATS[:, 2]
    boats = TICKET_BOATS
    outer_in_23 = np.isin(second, [4, 5, 6]) | np.isin(third, [4, 5, 6])
    is_123_only = np.isin(boats, [1, 2, 3]).all(axis=1)
    out = np.full(N_TICKETS, PATTERN_LABELS.index("1_absent"), dtype=np.int8)
    out[outer_in_23] = PATTERN_LABELS.index("outer_in_23")
    out[second == 1] = PATTERN_LABELS.index("1_second")
    out[third == 1] = PATTERN_LABELS.index("1_third")
    out[first == 1] = PATTERN_LABELS.index("1_head")
    out[is_123_only] = PATTERN_LABELS.index("123_only")
    out[first >= 4] = PATTERN_LABELS.index("outer_head")
    out[(first == 1) & outer_in_23] = PATTERN_LABELS.index("1_head_outer")
    return out


TICKET_PATTERN_IDX = ticket_pattern_indices()


def ticket_flags(ticket_idx: np.ndarray) -> dict[str, np.ndarray]:
    boats = TICKET_BOATS[ticket_idx]
    first = boats[:, 0]
    second = boats[:, 1]
    third = boats[:, 2]
    outer_23 = np.isin(second, [4, 5, 6]) | np.isin(third, [4, 5, 6])
    return {
        "is_1_head": first == 1,
        "is_1_second": second == 1,
        "is_1_third": third == 1,
        "is_1_absent": (first != 1) & (second != 1) & (third != 1),
        "outer_in_23": outer_23,
        "pattern_1_head_outer": (first == 1) & outer_23,
    }


def actual_ticket_strings(df: pd.DataFrame) -> pd.Series:
    return df[["r1", "r2", "r3"]].astype(int).astype(str).agg("-".join, axis=1)


def ticket_strings(indices: np.ndarray) -> list[str]:
    return [INDEX_TO_TRIFECTA[int(idx)] for idx in indices]


def summarize_counts(
    *,
    proba: np.ndarray,
    actual_idx: np.ndarray,
    prob_idx: np.ndarray,
    ranks: np.ndarray | None,
    grouping: str,
) -> dict[str, np.ndarray]:
    flat_prob = proba.reshape(-1)
    n_prob = len(PROB_BUCKETS)
    if grouping == "prob":
        cell_idx = prob_idx.reshape(-1).astype(np.int32)
        actual_cell = prob_idx[np.arange(len(actual_idx)), actual_idx].astype(np.int32)
        size = n_prob
    elif grouping == "prob_rank":
        if ranks is None:
            raise ValueError("ranks are required for prob_rank grouping")
        rank_idx = rank_bucket_indices(ranks)
        cell_idx = (prob_idx.astype(np.int32) * len(RANK_BUCKETS) + rank_idx.astype(np.int32)).reshape(-1)
        actual_cell = (
            prob_idx[np.arange(len(actual_idx)), actual_idx].astype(np.int32) * len(RANK_BUCKETS)
            + rank_idx[np.arange(len(actual_idx)), actual_idx].astype(np.int32)
        )
        size = n_prob * len(RANK_BUCKETS)
    elif grouping == "prob_pattern":
        pattern_idx = np.tile(TICKET_PATTERN_IDX.reshape(1, -1), (len(proba), 1))
        cell_idx = (prob_idx.astype(np.int32) * len(PATTERN_LABELS) + pattern_idx.astype(np.int32)).reshape(-1)
        actual_cell = (
            prob_idx[np.arange(len(actual_idx)), actual_idx].astype(np.int32) * len(PATTERN_LABELS)
            + TICKET_PATTERN_IDX[actual_idx].astype(np.int32)
        )
        size = n_prob * len(PATTERN_LABELS)
    else:
        raise ValueError(grouping)

    n_rows = np.bincount(cell_idx, minlength=size).astype(np.float64)
    sum_prob = np.bincount(cell_idx, weights=flat_prob, minlength=size).astype(np.float64)
    hits = np.bincount(actual_cell, minlength=size).astype(np.float64)
    avg_prob = np.divide(sum_prob, n_rows, out=np.zeros_like(sum_prob), where=n_rows > 0)
    hit_rate = np.divide(hits, n_rows, out=np.zeros_like(hits), where=n_rows > 0)
    local_lift = np.divide(hits, sum_prob, out=np.ones_like(hits), where=sum_prob > EPS)
    return {
        "n_rows": n_rows,
        "sum_prob": sum_prob,
        "hits": hits,
        "avg_prob": avg_prob,
        "hit_rate": hit_rate,
        "local_lift": local_lift,
    }


@dataclass
class CalibrationModel:
    model_name: str
    calibration_period: str
    calibration_method: str
    k: int
    min_count: int | None
    prob_lift: np.ndarray
    cell_lift: np.ndarray | None = None

    @property
    def config_name(self) -> str:
        if self.calibration_method == "prob_bucket_calibration":
            return f"{self.calibration_method}_k{self.k}"
        return f"{self.calibration_method}_k{self.k}_min{self.min_count}"


def build_calibration_models(
    *,
    model_name: str,
    calibration_period: str,
    proba: np.ndarray,
    df: pd.DataFrame,
) -> tuple[list[CalibrationModel], pd.DataFrame]:
    actual_idx = target_indices(df)
    prob_idx = probability_bucket_indices(proba)
    ranks = rank_matrix(proba)
    prob_counts = summarize_counts(proba=proba, actual_idx=actual_idx, prob_idx=prob_idx, ranks=None, grouping="prob")
    rows: list[dict[str, object]] = []
    models: list[CalibrationModel] = []

    for k in K_CANDIDATES:
        shrink = prob_counts["n_rows"] / (prob_counts["n_rows"] + k)
        prob_lift = shrink * prob_counts["local_lift"] + (1.0 - shrink) * 1.0
        models.append(
            CalibrationModel(
                model_name=model_name,
                calibration_period=calibration_period,
                calibration_method="prob_bucket_calibration",
                k=k,
                min_count=None,
                prob_lift=prob_lift.astype(np.float32),
            )
        )
        for p_idx, label in enumerate(PROB_LABELS):
            rows.append(
                calibration_row(
                    model_name=model_name,
                    calibration_period=calibration_period,
                    method="prob_bucket_calibration",
                    k=k,
                    min_count=np.nan,
                    prob_idx=p_idx,
                    prob_counts=prob_counts,
                    local_lift=prob_counts["local_lift"][p_idx],
                    calibration_lift=prob_lift[p_idx],
                    fallback_source="global_1.0",
                )
            )

    for grouping, method, n_secondary, secondary_labels in [
        ("prob_rank", "prob_bucket_rank_bucket_calibration", len(RANK_BUCKETS), RANK_LABELS),
        ("prob_pattern", "prob_bucket_ticket_pattern_calibration", len(PATTERN_LABELS), PATTERN_LABELS),
    ]:
        counts = summarize_counts(proba=proba, actual_idx=actual_idx, prob_idx=prob_idx, ranks=ranks, grouping=grouping)
        for k in K_CANDIDATES:
            shrink_prob = prob_counts["n_rows"] / (prob_counts["n_rows"] + k)
            prob_lift = shrink_prob * prob_counts["local_lift"] + (1.0 - shrink_prob) * 1.0
            for min_count in MIN_COUNT_CANDIDATES:
                cell_lift = np.zeros_like(counts["local_lift"], dtype=np.float64)
                for p_idx in range(len(PROB_BUCKETS)):
                    for s_idx in range(n_secondary):
                        cell = p_idx * n_secondary + s_idx
                        n = counts["n_rows"][cell]
                        base = prob_lift[p_idx]
                        if n < min_count:
                            lift = base
                            fallback = "prob_bucket_lift"
                        else:
                            shrink = n / (n + k)
                            lift = shrink * counts["local_lift"][cell] + (1.0 - shrink) * base
                            fallback = "cell_shrink_to_prob_bucket"
                        cell_lift[cell] = lift
                        rows.append(
                            calibration_row(
                                model_name=model_name,
                                calibration_period=calibration_period,
                                method=method,
                                k=k,
                                min_count=min_count,
                                prob_idx=p_idx,
                                prob_counts=counts,
                                local_lift=counts["local_lift"][cell],
                                calibration_lift=lift,
                                fallback_source=fallback,
                                secondary_label=secondary_labels[s_idx],
                                cell=cell,
                            )
                        )
                models.append(
                    CalibrationModel(
                        model_name=model_name,
                        calibration_period=calibration_period,
                        calibration_method=method,
                        k=k,
                        min_count=min_count,
                        prob_lift=prob_lift.astype(np.float32),
                        cell_lift=cell_lift.astype(np.float32),
                    )
                )
    return models, pd.DataFrame(rows)


def calibration_row(
    *,
    model_name: str,
    calibration_period: str,
    method: str,
    k: int,
    min_count: float,
    prob_idx: int,
    prob_counts: dict[str, np.ndarray],
    local_lift: float,
    calibration_lift: float,
    fallback_source: str,
    secondary_label: str = "",
    cell: int | None = None,
) -> dict[str, object]:
    idx = prob_idx if cell is None else cell
    prob_label = PROB_LABELS[prob_idx]
    return {
        "calibration_period": calibration_period,
        "model_name": model_name,
        "calibration_method": method,
        "k": k,
        "min_count": min_count,
        "prob_bucket": prob_label,
        "prob_bucket_display": PROB_DISPLAY[prob_label],
        "rank_bucket": secondary_label if "rank" in method else "",
        "ticket_pattern": secondary_label if "pattern" in method else "",
        "n_rows": int(prob_counts["n_rows"][idx]),
        "sum_raw_prob": float(prob_counts["sum_prob"][idx]),
        "actual_hits": int(prob_counts["hits"][idx]),
        "avg_raw_prob": float(prob_counts["avg_prob"][idx]),
        "actual_hit_rate": float(prob_counts["hit_rate"][idx]),
        "local_lift": float(local_lift),
        "calibration_lift": float(calibration_lift),
        "fallback_source": fallback_source,
    }


def lift_matrix(model: CalibrationModel, proba: np.ndarray, ranks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prob_idx = probability_bucket_indices(proba)
    if model.calibration_method == "prob_bucket_calibration":
        lift = model.prob_lift[prob_idx]
        cell_n = np.full_like(lift, np.nan, dtype=np.float32)
        source_idx = prob_idx.astype(np.int16)
    elif model.calibration_method == "prob_bucket_rank_bucket_calibration":
        if ranks is None:
            ranks = rank_matrix(proba)
        rank_idx = rank_bucket_indices(ranks)
        source_idx = prob_idx.astype(np.int16) * len(RANK_BUCKETS) + rank_idx.astype(np.int16)
        lift = model.cell_lift[source_idx]
        cell_n = np.full_like(lift, np.nan, dtype=np.float32)
    elif model.calibration_method == "prob_bucket_ticket_pattern_calibration":
        pattern_idx = np.tile(TICKET_PATTERN_IDX.reshape(1, -1), (len(proba), 1))
        source_idx = prob_idx.astype(np.int16) * len(PATTERN_LABELS) + pattern_idx.astype(np.int16)
        lift = model.cell_lift[source_idx]
        cell_n = np.full_like(lift, np.nan, dtype=np.float32)
    else:
        raise ValueError(model.calibration_method)
    return lift.astype(np.float32), prob_idx, source_idx, cell_n


def apply_calibration(model: CalibrationModel, proba: np.ndarray, ranks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lift, prob_idx, source_idx, _ = lift_matrix(model, proba, ranks)
    score = proba.astype(np.float64) * lift.astype(np.float64)
    denom = score.sum(axis=1, keepdims=True)
    calibrated = np.divide(score, denom, out=np.full_like(score, 1.0 / N_TICKETS), where=denom > EPS)
    return calibrated.astype(np.float32), lift, source_idx


def metrics(df: pd.DataFrame, proba: np.ndarray, prefix: str) -> dict[str, float]:
    actual_idx = target_indices(df)
    ranks = actual_rank_fast(proba, actual_idx)
    actual_prob = np.clip(proba[np.arange(len(df)), actual_idx], EPS, 1.0)
    return {
        f"{prefix}_logloss": float((-np.log(actual_prob)).mean()),
        f"{prefix}_brier_score": float(brier_multiclass(proba, actual_idx).mean()),
        f"{prefix}_top1_hit_rate": float((ranks == 1).mean()),
        f"{prefix}_top3_contains_rate": float((ranks <= 3).mean()),
        f"{prefix}_top5_contains_rate": float((ranks <= 5).mean()),
        f"{prefix}_top10_contains_rate": float((ranks <= 10).mean()),
        f"{prefix}_mean_actual_rank": float(ranks.mean()),
    }


def ece_by_prob_bucket(df: pd.DataFrame, proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    actual_idx = target_indices(df)
    prob_idx = probability_bucket_indices(proba)
    counts = summarize_counts(proba=proba, actual_idx=actual_idx, prob_idx=prob_idx, ranks=None, grouping="prob")
    total = float(counts["n_rows"].sum())
    abs_err = np.abs(counts["hit_rate"] - counts["avg_prob"])
    ece = float(np.sum(counts["n_rows"] * abs_err) / total) if total else np.nan
    rows = []
    for p_idx, label in enumerate(PROB_LABELS):
        rows.append(
            {
                "prob_bucket": label,
                "prob_bucket_display": PROB_DISPLAY[label],
                "n_rows": int(counts["n_rows"][p_idx]),
                "avg_prob": float(counts["avg_prob"][p_idx]),
                "actual_hit_rate": float(counts["hit_rate"][p_idx]),
                "calibration_error": float(counts["hit_rate"][p_idx] - counts["avg_prob"][p_idx]),
                "abs_calibration_error": float(abs_err[p_idx]),
            }
        )
    return ece, pd.DataFrame(rows)


def evaluate_models(
    *,
    dataset: str,
    df: pd.DataFrame,
    raw_models: dict[str, np.ndarray],
    calibration_models: dict[str, list[CalibrationModel]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_rows: list[dict[str, object]] = []
    bucket_rows: list[pd.DataFrame] = []
    for model_name, raw_proba in raw_models.items():
        raw_metrics = metrics(df, raw_proba, "raw")
        raw_ece, raw_bucket = ece_by_prob_bucket(df, raw_proba)
        raw_bucket.insert(0, "probability_type", "raw_prob")
        raw_bucket.insert(0, "config_name", "no_calibration")
        raw_bucket.insert(0, "calibration_method", "no_calibration")
        raw_bucket.insert(0, "model_name", model_name)
        raw_bucket.insert(0, "dataset", dataset)
        bucket_rows.append(raw_bucket)
        eval_rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "config_name": "no_calibration",
                "calibration_method": "no_calibration",
                "k": np.nan,
                "min_count": np.nan,
                "calibration_period": "",
                **raw_metrics,
                "raw_ECE": raw_ece,
                **{k.replace("raw_", "calibrated_"): v for k, v in raw_metrics.items()},
                "calibrated_ECE": raw_ece,
            }
        )
        ranks = rank_matrix(raw_proba)
        for cal_model in calibration_models[model_name]:
            calibrated, _, _ = apply_calibration(cal_model, raw_proba, ranks)
            cal_metrics = metrics(df, calibrated, "calibrated")
            cal_ece, cal_bucket = ece_by_prob_bucket(df, calibrated)
            cal_bucket.insert(0, "probability_type", "calibrated_prob")
            cal_bucket.insert(0, "config_name", cal_model.config_name)
            cal_bucket.insert(0, "calibration_method", cal_model.calibration_method)
            cal_bucket.insert(0, "model_name", model_name)
            cal_bucket.insert(0, "dataset", dataset)
            bucket_rows.append(cal_bucket)
            row = {
                "dataset": dataset,
                "model_name": model_name,
                "config_name": cal_model.config_name,
                "calibration_method": cal_model.calibration_method,
                "k": cal_model.k,
                "min_count": cal_model.min_count if cal_model.min_count is not None else np.nan,
                "calibration_period": cal_model.calibration_period,
                **raw_metrics,
                "raw_ECE": raw_ece,
                **cal_metrics,
                "calibrated_ECE": cal_ece,
            }
            for metric in ["logloss", "brier_score", "ECE", "top1_hit_rate", "top3_contains_rate", "top5_contains_rate", "top10_contains_rate", "mean_actual_rank"]:
                row[f"delta_{metric}"] = row[f"calibrated_{metric}"] - row[f"raw_{metric}"]
            eval_rows.append(row)
    return pd.DataFrame(eval_rows), pd.concat(bucket_rows, ignore_index=True)


def choose_best_configs(c_eval: pd.DataFrame) -> pd.DataFrame:
    candidates = c_eval[c_eval["calibration_method"].ne("no_calibration")].copy()
    # EV probability calibration prioritizes logloss/ECE while requiring ranking not to collapse.
    candidates["rank_penalty"] = np.maximum(candidates["calibrated_top5_contains_rate"] - candidates["raw_top5_contains_rate"], -0.01)
    candidates = candidates.sort_values(
        ["model_name", "calibrated_logloss", "calibrated_ECE", "delta_top5_contains_rate"],
        ascending=[True, True, True, False],
    )
    return candidates.groupby("model_name", as_index=False).head(1).reset_index(drop=True)


def source_details_for_top20(model: CalibrationModel, calibration_table: pd.DataFrame) -> pd.DataFrame:
    rows = calibration_table[
        (calibration_table["calibration_period"].eq(model.calibration_period))
        & (calibration_table["model_name"].eq(model.model_name))
        & (calibration_table["calibration_method"].eq(model.calibration_method))
        & (calibration_table["k"].eq(model.k))
    ].copy()
    if model.min_count is None:
        rows = rows[rows["min_count"].isna()]
        rows["source_idx"] = [PROB_LABELS.index(x) for x in rows["prob_bucket"]]
    elif model.calibration_method == "prob_bucket_rank_bucket_calibration":
        rows = rows[rows["min_count"].eq(model.min_count)]
        rows["source_idx"] = [PROB_LABELS.index(p) * len(RANK_BUCKETS) + RANK_LABELS.index(r) for p, r in zip(rows["prob_bucket"], rows["rank_bucket"])]
    else:
        rows = rows[rows["min_count"].eq(model.min_count)]
        rows["source_idx"] = [PROB_LABELS.index(p) * len(PATTERN_LABELS) + PATTERN_LABELS.index(t) for p, t in zip(rows["prob_bucket"], rows["ticket_pattern"])]
    return rows.set_index("source_idx")


def odds_payout_for_ticket(df: pd.DataFrame, row_idx: int, ticket: str) -> float:
    ticket_key = ticket.replace("-", "_")
    for col in [f"odds_{ticket_key}", f"odds_{ticket}", f"payout_{ticket_key}", f"payout_{ticket}"]:
        if col in df.columns:
            value = pd.to_numeric(pd.Series([df.iloc[row_idx][col]]), errors="coerce").iloc[0]
            if pd.notna(value):
                value = float(value)
                return value * 100.0 if value <= 50 else value
    return np.nan


def build_top20_output(
    *,
    df: pd.DataFrame,
    raw_models: dict[str, np.ndarray],
    best_models: list[CalibrationModel],
    calibration_table: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    actual_ticket = actual_ticket_strings(df)
    for model in best_models:
        raw = raw_models[model.model_name]
        raw_ranks = rank_matrix(raw)
        calibrated, lift, source_idx = apply_calibration(model, raw, raw_ranks)
        calibrated_ranks = rank_matrix(calibrated)
        top20 = np.argpartition(-calibrated, kth=19, axis=1)[:, :20]
        scores = calibrated[np.arange(len(df))[:, None], top20]
        order = np.argsort(-scores, axis=1)
        top20 = top20[np.arange(len(df))[:, None], order]
        source_details = source_details_for_top20(model, calibration_table)
        rows: list[dict[str, object]] = []
        for race_i in range(len(df)):
            for ticket_idx in top20[race_i]:
                ticket_idx = int(ticket_idx)
                ticket = INDEX_TO_TRIFECTA[ticket_idx]
                source = source_details.loc[int(source_idx[race_i, ticket_idx])]
                odds_payout = odds_payout_for_ticket(df, race_i, ticket)
                calibrated_prob = float(calibrated[race_i, ticket_idx])
                ev_index = calibrated_prob * (odds_payout / 100.0) if np.isfinite(odds_payout) else np.nan
                rows.append(
                    {
                        "race_id": f"D_{race_i:06d}",
                        "dataset": "validation_202603_202605",
                        "model_name": model.model_name,
                        "calibration_config": model.config_name,
                        "jcd": str(df.iloc[race_i]["jcd"]).zfill(2),
                        "r": int(df.iloc[race_i]["r"]),
                        "actual_ticket": actual_ticket.iloc[race_i],
                        "3rt": df.iloc[race_i]["3rt"] if "3rt" in df.columns else np.nan,
                        "ticket": ticket,
                        "raw_prob": float(raw[race_i, ticket_idx]),
                        "calibrated_prob": calibrated_prob,
                        "raw_rank": int(raw_ranks[race_i, ticket_idx]),
                        "calibrated_rank": int(calibrated_ranks[race_i, ticket_idx]),
                        "ticket_pattern": PATTERN_LABELS[int(TICKET_PATTERN_IDX[ticket_idx])],
                        "prob_bucket": PROB_LABELS[int(probability_bucket_indices(np.asarray([raw[race_i, ticket_idx]], dtype=np.float32))[0])],
                        "calibration_lift": float(lift[race_i, ticket_idx]),
                        "calibration_source": source["fallback_source"],
                        "calibration_n": int(source["n_rows"]),
                        "odds_payout": odds_payout,
                        "ev_index": ev_index,
                    }
                )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def model_probabilities(
    *,
    df: pd.DataFrame,
    raw_direct: np.ndarray,
    direct_jcdr,
) -> dict[str, np.ndarray]:
    jcd_r = direct_jcdr.apply(df, raw_direct, strength=1.0, clip=0.60, top_n=20)
    outer = apply_outer_boost(
        df,
        jcd_r,
        pattern="outer_in_23_boost_by_outer_winrate",
        boost_score=0.40,
        clip=0.40,
        top_n=20,
    )
    return {MODEL_RAW: raw_direct, MODEL_JCDR: jcd_r, MODEL_OUTER: outer}


def markdown_table(df: pd.DataFrame, n: int = 12) -> str:
    if df.empty:
        return "(no rows)"
    small = df.head(n).copy()
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.5f}")
        else:
            small[col] = small[col].astype(str)
    header = "| " + " | ".join(small.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in small.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def write_summary(
    path: Path,
    c_eval: pd.DataFrame,
    recent_eval: pd.DataFrame,
    best: pd.DataFrame,
    bucket_eval: pd.DataFrame,
) -> None:
    c_sorted = c_eval[c_eval["calibration_method"].ne("no_calibration")].sort_values(["calibrated_logloss", "calibrated_ECE"]).copy()
    recent_best_names = best[["model_name", "config_name"]].merge(recent_eval, on=["model_name", "config_name"], how="left")
    selected_configs = best[["model_name", "config_name"]].drop_duplicates()
    high_bucket = bucket_eval[
        (bucket_eval["dataset"].eq("C_future_check"))
        & (bucket_eval["probability_type"].eq("calibrated_prob"))
        & (bucket_eval["prob_bucket"].isin(["prob_7_10", "prob_10_12", "prob_12_15", "prob_15_plus"]))
    ].merge(selected_configs, on=["model_name", "config_name"], how="inner")
    high_bucket = high_bucket.sort_values(["model_name", "config_name", "prob_bucket"])
    lines = [
        "# Full 120-Ticket Probability Calibration Summary",
        "",
        "120通り全体を対象に、ランキング用 `raw_prob` とは別のEV計算用 `calibrated_prob` を作成しました。",
        "C区間を主評価、2026/3-5を最終確認として扱い、2026/3-5の結果は校正値作成に使っていません。",
        "",
        "## C区間: 上位校正方式",
        markdown_table(
            c_sorted[
                [
                    "model_name",
                    "config_name",
                    "calibration_method",
                    "k",
                    "min_count",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_brier_score",
                    "calibrated_brier_score",
                    "delta_brier_score",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                ]
            ],
            18,
        ),
        "",
        "## C区間で選んだEV用設定",
        markdown_table(
            best[
                [
                    "model_name",
                    "config_name",
                    "calibration_method",
                    "k",
                    "min_count",
                    "calibrated_logloss",
                    "calibrated_brier_score",
                    "calibrated_ECE",
                    "delta_logloss",
                    "delta_brier_score",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                ]
            ],
            10,
        ),
        "",
        "## 2026/3-5: 選定設定の固定適用",
        markdown_table(
            recent_best_names[
                [
                    "model_name",
                    "config_name",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_brier_score",
                    "calibrated_brier_score",
                    "delta_brier_score",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top1_hit_rate",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ],
            10,
        ),
        "",
        "## C区間: 高確率帯の校正後確認",
        markdown_table(
            high_bucket[
                [
                    "model_name",
                    "config_name",
                    "prob_bucket_display",
                    "n_rows",
                    "avg_prob",
                    "actual_hit_rate",
                    "calibration_error",
                    "abs_calibration_error",
                ]
            ],
            24,
        ),
        "",
        "## Notes",
        "- `prob_bucket_calibration` はraw_prob帯だけでliftを作ります。",
        "- `prob_bucket_rank_bucket_calibration` はraw_prob帯×raw_rank帯でliftを作り、少数セルはprob_bucket liftへフォールバックします。",
        "- `prob_bucket_ticket_pattern_calibration` はraw_prob帯×ticket_patternでliftを作り、少数セルはprob_bucket liftへフォールバックします。",
        "- `calibrated_score = raw_prob * calibration_lift` のあと、各レース内120通りで再正規化しています。",
        "- corrected系モデルのB区間側校正は、既存のjcd_r補正自体がB由来である点に注意してください。主判断はC区間と2026/3-5固定適用で見ています。",
        "- EVに使うなら、C区間でlogloss/ECE/brierが改善し、2026/3-5でも同方向だった設定を採用候補にしてください。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    past = load_featured(args.past_csv, require_registrations=False, name="dataset_1_past")
    recent = load_featured(args.recent_csv, require_registrations=True, name="dataset_2_recent")
    _, b, c = split_past_by_row_order(past, args.split_a_size, args.split_b_size)

    final_cache = out / "final_fixed_correction_cache"
    sensitivity_cache = out / "correction_sensitivity_cache"
    b_raw_direct = np.load(final_cache / "direct120_A_train_B_proba.npy")
    c_raw_direct = np.load(sensitivity_cache / "direct120_AB_train_C_proba.npy")
    recent_raw_direct = np.load(final_cache / "direct120_past55_recent_proba.npy")

    direct_jcdr = fit_log_ticket_correction(b, b_raw_direct, "jcd_r_correction", k=500, min_count=300)
    b_models = model_probabilities(df=b, raw_direct=b_raw_direct, direct_jcdr=direct_jcdr)
    c_models = model_probabilities(df=c, raw_direct=c_raw_direct, direct_jcdr=direct_jcdr)
    recent_models = model_probabilities(df=recent, raw_direct=recent_raw_direct, direct_jcdr=direct_jcdr)

    calibration_rows = []
    b_calibration_models: dict[str, list[CalibrationModel]] = {}
    bc_calibration_models: dict[str, list[CalibrationModel]] = {}
    for model_name in MODEL_NAMES:
        print(f"Fitting B calibration for {model_name}...")
        models, table = build_calibration_models(
            model_name=model_name,
            calibration_period="B_only",
            proba=b_models[model_name],
            df=b,
        )
        b_calibration_models[model_name] = models
        calibration_rows.append(table)

        print(f"Fitting B+C calibration for {model_name}...")
        bc_proba = np.vstack([b_models[model_name], c_models[model_name]])
        bc_df = pd.concat([b, c], ignore_index=True)
        models_bc, table_bc = build_calibration_models(
            model_name=model_name,
            calibration_period="B_C_out_of_sample",
            proba=bc_proba,
            df=bc_df,
        )
        bc_calibration_models[model_name] = models_bc
        calibration_rows.append(table_bc)

    calibration_table = pd.concat(calibration_rows, ignore_index=True)
    calibration_table.to_csv(out / "full_prob_calibration_table.csv", index=False, encoding="utf-8-sig")

    print("Evaluating C future check...")
    c_eval, c_bucket = evaluate_models(
        dataset="C_future_check",
        df=c,
        raw_models=c_models,
        calibration_models=b_calibration_models,
    )
    c_eval.to_csv(out / "full_prob_calibration_C_validation.csv", index=False, encoding="utf-8-sig")

    print("Evaluating 2026/3-5 final check...")
    recent_eval, recent_bucket = evaluate_models(
        dataset="validation_202603_202605",
        df=recent,
        raw_models=recent_models,
        calibration_models=bc_calibration_models,
    )
    recent_eval.to_csv(out / "full_prob_calibration_202603_202605_validation.csv", index=False, encoding="utf-8-sig")

    bucket_eval = pd.concat([c_bucket, recent_bucket], ignore_index=True)
    bucket_eval.to_csv(out / "full_prob_calibration_by_bucket.csv", index=False, encoding="utf-8-sig")

    best = choose_best_configs(c_eval)
    best_models = []
    for _, row in best.iterrows():
        matches = [
            m
            for m in bc_calibration_models[row["model_name"]]
            if m.config_name == row["config_name"]
        ]
        if matches:
            best_models.append(matches[0])

    print("Writing 2026/3-5 calibrated Top20 EV candidates...")
    top20 = build_top20_output(
        df=recent,
        raw_models=recent_models,
        best_models=best_models,
        calibration_table=calibration_table,
    )
    top20.to_csv(out / "full_prob_calibrated_top20_202603_202605.csv", index=False, encoding="utf-8-sig")

    write_summary(out / "full_prob_calibration_summary.md", c_eval, recent_eval, best, bucket_eval)
    print(f"Wrote full probability calibration outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
