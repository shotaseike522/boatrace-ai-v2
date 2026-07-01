from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from boat_model.data_loader import load_race_data
from boat_model.features import (
    INDEX_TO_TRIFECTA,
    TRIFECTA_PERMUTATIONS,
    WINRATE_COLS,
    add_basic_features,
    normalize_probability_matrix,
    target_indices,
)


DEFAULT_PAST = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_1_past_201307_202602.csv"
DEFAULT_RECENT = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_2_recent_202603_202605.csv"

N_TICKETS = len(TRIFECTA_PERMUTATIONS)
TICKET_BOATS = np.asarray(TRIFECTA_PERMUTATIONS, dtype=np.int16)
TICKETS = [INDEX_TO_TRIFECTA[i] for i in range(N_TICKETS)]
EPS = 1e-12

SEGMENTS: dict[str, list[int] | None] = {
    "all_races": None,
    **{f"r{i:02d}": [i] for i in range(1, 13)},
    "early": [1, 2, 3, 4],
    "middle": [5, 6, 7, 8],
    "late": [9, 10],
    "final": [11, 12],
    "r11": [11],
    "r12": [12],
}
SEGMENT_ORDER = ["all_races", "early", "middle", "late", "final", "r11", "r12"]
RACE_NO_ORDER = [f"r{i:02d}" for i in range(1, 13)]

BOOST_SCORES = [0.05, 0.10, 0.15, 0.20, 0.30]
CLIPS = [0.10, 0.20, 0.30]
TOP_NS = [10, 20, 30]
BOOST_PATTERNS = [
    "outer_in_23_boost_all",
    "outer_in_23_boost_by_outer_winrate",
    "outer_in_23_boost_by_outer_rank",
    "outer_in_23_boost_when_inner_weak",
    "combined_outer_partner_boost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional 1-head outer partner boost analysis.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST)
    parser.add_argument("--recent-csv", default=DEFAULT_RECENT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split-a-size", type=int, default=200_000)
    parser.add_argument("--split-b-size", type=int, default=200_000)
    return parser.parse_args()


def load_featured(path: str, *, require_registrations: bool, name: str) -> pd.DataFrame:
    raw, _ = load_race_data(path, require_registrations=require_registrations, name=name)
    return add_basic_features(raw).reset_index(drop=True)


def split_past_by_row_order(past: pd.DataFrame, a_size: int, b_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = past.iloc[:a_size].copy().reset_index(drop=True)
    b = past.iloc[a_size : a_size + b_size].copy().reset_index(drop=True)
    c = past.iloc[a_size + b_size :].copy().reset_index(drop=True)
    return a, b, c


def key_series(df: pd.DataFrame, correction_type: str) -> pd.Series:
    jcd = df["jcd"].astype(str).str.zfill(2)
    r = pd.to_numeric(df["r"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
    if correction_type == "jcd_r_correction":
        return jcd + "_" + r
    if correction_type == "global":
        return pd.Series("global", index=df.index)
    raise ValueError(correction_type)


@dataclass
class LogTicketCorrection:
    correction_type: str
    global_log_corr: np.ndarray
    local_log_corr: dict[str, np.ndarray]
    k: int
    min_count: int

    def apply(self, df: pd.DataFrame, proba: np.ndarray, *, strength: float, clip: float, top_n: int | None = None) -> np.ndarray:
        keys = key_series(df, self.correction_type).to_numpy()
        raw = np.log(np.clip(proba.astype(np.float64), EPS, 1.0))
        ranks = None
        if top_n is not None and top_n < N_TICKETS:
            order = np.argsort(-proba, axis=1)
            ranks = np.empty_like(order, dtype=np.int16)
            ranks[np.arange(len(df))[:, None], order] = np.arange(1, N_TICKETS + 1, dtype=np.int16)
        for key in pd.unique(keys):
            mask = keys == key
            adj = np.clip(strength * self.local_log_corr.get(str(key), self.global_log_corr), -clip, clip)
            if ranks is None:
                raw[mask] += adj.reshape(1, -1)
            else:
                block = raw[mask]
                block += (ranks[mask] <= top_n).astype(float) * adj.reshape(1, -1)
                raw[mask] = block
        raw -= raw.max(axis=1, keepdims=True)
        return normalize_probability_matrix(np.exp(raw)).astype(np.float32)


def fit_log_ticket_correction(
    df: pd.DataFrame,
    proba: np.ndarray,
    correction_type: str,
    *,
    k: int = 500,
    min_count: int = 300,
    smoothing: float = 1.0,
) -> LogTicketCorrection:
    actual_idx = target_indices(df)
    expected_global = proba.sum(axis=0).astype(np.float64) + smoothing
    hits_global = np.bincount(actual_idx, minlength=N_TICKETS).astype(np.float64) + smoothing / N_TICKETS
    global_log = np.log(np.clip(hits_global / np.maximum(expected_global, EPS), 0.05, 20.0))
    keys = key_series(df, correction_type).to_numpy()
    local: dict[str, np.ndarray] = {}
    for key in pd.unique(keys):
        mask = keys == key
        expected = proba[mask].sum(axis=0).astype(np.float64) + smoothing
        hits = np.bincount(actual_idx[mask], minlength=N_TICKETS).astype(np.float64) + smoothing / N_TICKETS
        local_log = np.log(np.clip(hits / np.maximum(expected, EPS), 0.05, 20.0))
        n = int(mask.sum())
        shrink = n / (n + k) if n >= min_count else 0.0
        local[str(key)] = (shrink * local_log + (1.0 - shrink) * global_log).astype(np.float32)
    return LogTicketCorrection(correction_type, global_log.astype(np.float32), local, k, min_count)


@dataclass
class PositionBucketCorrection:
    correction_type: str
    global_log_corr: np.ndarray
    local_log_corr: dict[str, np.ndarray]
    k: int
    min_count: int

    def apply(self, df: pd.DataFrame, proba: np.ndarray, *, strengths: tuple[float, float, float], clip: float, top_n: int | None = None) -> np.ndarray:
        keys = key_series(df, self.correction_type).to_numpy()
        raw = np.log(np.clip(proba.astype(np.float64), EPS, 1.0))
        ranks = None
        if top_n is not None and top_n < N_TICKETS:
            order = np.argsort(-proba, axis=1)
            ranks = np.empty_like(order, dtype=np.int16)
            ranks[np.arange(len(df))[:, None], order] = np.arange(1, N_TICKETS + 1, dtype=np.int16)
        for key in pd.unique(keys):
            mask = keys == key
            corr = self.local_log_corr.get(str(key), self.global_log_corr)
            ticket_adj = np.zeros(N_TICKETS, dtype=np.float64)
            for pos, strength in enumerate(strengths):
                if abs(strength) < 1e-12:
                    continue
                boats = TICKET_BOATS[:, pos] - 1
                ticket_adj += float(strength) * corr[pos, boats]
            ticket_adj = np.clip(ticket_adj, -clip, clip)
            if ranks is None:
                raw[mask] += ticket_adj.reshape(1, -1)
            else:
                block = raw[mask]
                block += (ranks[mask] <= top_n).astype(float) * ticket_adj.reshape(1, -1)
                raw[mask] = block
        raw -= raw.max(axis=1, keepdims=True)
        return normalize_probability_matrix(np.exp(raw)).astype(np.float32)


def fit_position_bucket_correction(
    df: pd.DataFrame,
    proba: np.ndarray,
    correction_type: str,
    *,
    k: int = 500,
    min_count: int = 300,
    smoothing: float = 1.0,
) -> PositionBucketCorrection:
    actual_boats = TICKET_BOATS[target_indices(df)]

    def calc(mask: np.ndarray) -> np.ndarray:
        out = np.zeros((3, 6), dtype=np.float64)
        local_proba = proba[mask]
        local_actual = actual_boats[mask]
        for pos in range(3):
            for boat in range(1, 7):
                ticket_mask = TICKET_BOATS[:, pos] == boat
                expected = float(local_proba[:, ticket_mask].sum()) + smoothing
                hits = float(np.sum(local_actual[:, pos] == boat)) + smoothing / 6.0
                out[pos, boat - 1] = np.log(np.clip(hits / max(expected, EPS), 0.05, 20.0))
        return out

    global_log = calc(np.ones(len(df), dtype=bool)).astype(np.float32)
    local: dict[str, np.ndarray] = {}
    keys = key_series(df, correction_type).to_numpy()
    for key in pd.unique(keys):
        mask = keys == key
        n = int(mask.sum())
        local_log = calc(mask)
        shrink = n / (n + k) if n >= min_count else 0.0
        local[str(key)] = (shrink * local_log + (1.0 - shrink) * global_log).astype(np.float32)
    return PositionBucketCorrection(correction_type, global_log, local, k, min_count)


def segment_mask(df: pd.DataFrame, segment: str) -> np.ndarray:
    races = SEGMENTS[segment]
    if races is None:
        return np.ones(len(df), dtype=bool)
    r = pd.to_numeric(df["r"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.isin(r, races)


def actual_rank_fast(proba: np.ndarray, actual_idx: np.ndarray) -> np.ndarray:
    actual_prob = proba[np.arange(len(actual_idx)), actual_idx]
    return (proba > actual_prob[:, None]).sum(axis=1).astype(np.int16) + 1


def brier_multiclass(proba: np.ndarray, actual_idx: np.ndarray) -> np.ndarray:
    actual_prob = proba[np.arange(len(actual_idx)), actual_idx]
    return np.sum(proba * proba, axis=1) + 1.0 - 2.0 * actual_prob


def topn_indices(proba: np.ndarray, n: int) -> np.ndarray:
    part = np.argpartition(-proba, kth=n - 1, axis=1)[:, :n]
    part_scores = proba[np.arange(len(proba))[:, None], part]
    order = np.argsort(-part_scores, axis=1)
    return part[np.arange(len(proba))[:, None], order]


def rank_mask(proba: np.ndarray, top_n: int) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(len(proba))[:, None], order] = np.arange(1, N_TICKETS + 1, dtype=np.int16)
    return ranks <= top_n


def ticket_pattern_1_head_outer(ticket_indices: np.ndarray) -> np.ndarray:
    boats = TICKET_BOATS[ticket_indices]
    return (boats[..., 0] == 1) & np.isin(boats[..., 1], [4, 5, 6]) | ((boats[..., 0] == 1) & np.isin(boats[..., 2], [4, 5, 6]))


def base_outer_ticket_mask() -> np.ndarray:
    return (TICKET_BOATS[:, 0] == 1) & (np.isin(TICKET_BOATS[:, 1], [4, 5, 6]) | np.isin(TICKET_BOATS[:, 2], [4, 5, 6]))


def winrate_and_rank_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rates = df[WINRATE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rates = np.where(np.isfinite(rates), rates, np.nanmedian(rates, axis=0))
    ranks = pd.DataFrame(rates).rank(axis=1, ascending=False, method="min").to_numpy(dtype=float)
    field_avg = rates.mean(axis=1)
    inner_weak = (
        (rates[:, 1] < field_avg)
        | (rates[:, 2] < field_avg)
        | (ranks[:, 1] >= 4)
        | (ranks[:, 2] >= 4)
    )
    return rates, ranks, field_avg, inner_weak


def outer_condition_matrix(df: pd.DataFrame, pattern: str) -> np.ndarray:
    rates, ranks, field_avg, inner_weak = winrate_and_rank_arrays(df)
    ticket_mask = base_outer_ticket_mask()
    cond = np.tile(ticket_mask.reshape(1, -1), (len(df), 1))
    if pattern == "outer_in_23_boost_all":
        return cond

    second = TICKET_BOATS[:, 1] - 1
    third = TICKET_BOATS[:, 2] - 1
    second_outer = second >= 3
    third_outer = third >= 3

    outer_win = np.zeros((len(df), N_TICKETS), dtype=bool)
    outer_rank = np.zeros((len(df), N_TICKETS), dtype=bool)
    for ticket_idx in np.where(ticket_mask)[0]:
        parts = []
        rank_parts = []
        if second_outer[ticket_idx]:
            parts.append(rates[:, second[ticket_idx]] >= field_avg)
            rank_parts.append(ranks[:, second[ticket_idx]] <= 4)
        if third_outer[ticket_idx]:
            parts.append(rates[:, third[ticket_idx]] >= field_avg)
            rank_parts.append(ranks[:, third[ticket_idx]] <= 4)
        outer_win[:, ticket_idx] = np.logical_or.reduce(parts) if parts else False
        outer_rank[:, ticket_idx] = np.logical_or.reduce(rank_parts) if rank_parts else False

    inner = inner_weak.reshape(-1, 1)
    if pattern == "outer_in_23_boost_by_outer_winrate":
        return cond & outer_win
    if pattern == "outer_in_23_boost_by_outer_rank":
        return cond & outer_rank
    if pattern == "outer_in_23_boost_when_inner_weak":
        return cond & inner
    if pattern == "combined_outer_partner_boost":
        return cond & (outer_win | outer_rank) & inner
    raise ValueError(pattern)


def apply_outer_boost(
    df: pd.DataFrame,
    base_proba: np.ndarray,
    *,
    pattern: str,
    boost_score: float,
    clip: float,
    top_n: int,
) -> np.ndarray:
    cond = outer_condition_matrix(df, pattern)
    cond &= rank_mask(base_proba, top_n)
    add = min(float(boost_score), float(clip))
    raw = np.log(np.clip(base_proba.astype(np.float64), EPS, 1.0))
    raw += cond.astype(np.float64) * add
    raw -= raw.max(axis=1, keepdims=True)
    return normalize_probability_matrix(np.exp(raw)).astype(np.float32)


def apply_outer_boost_with_masks(
    base_proba: np.ndarray,
    *,
    condition_mask: np.ndarray,
    top_mask: np.ndarray,
    boost_score: float,
    clip: float,
) -> np.ndarray:
    add = min(float(boost_score), float(clip))
    raw = np.log(np.clip(base_proba.astype(np.float64), EPS, 1.0))
    raw += (condition_mask & top_mask).astype(np.float64) * add
    raw -= raw.max(axis=1, keepdims=True)
    return normalize_probability_matrix(np.exp(raw)).astype(np.float32)


def actual_1_head_outer_mask(df: pd.DataFrame) -> np.ndarray:
    values = df[["r1", "r2", "r3"]].astype(int).to_numpy()
    return (values[:, 0] == 1) & (np.isin(values[:, 1], [4, 5, 6]) | np.isin(values[:, 2], [4, 5, 6]))


def top5_outer_metrics(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top5 = topn_indices(proba, 5)
    pattern = ticket_pattern_1_head_outer(top5)
    return pattern.any(axis=1), pattern.mean(axis=1)


def eval_rows(
    *,
    dataset: str,
    df: pd.DataFrame,
    model_name: str,
    proba: np.ndarray,
    baseline_ranks: np.ndarray,
    segments: list[str],
    extra: dict[str, object],
) -> pd.DataFrame:
    actual_idx = target_indices(df)
    ranks = actual_rank_fast(proba, actual_idx)
    actual_prob = np.clip(proba[np.arange(len(df)), actual_idx], EPS, 1.0)
    brier = brier_multiclass(proba, actual_idx)
    top5_has_outer, top5_outer_share = top5_outer_metrics(proba)
    top1 = np.argmax(proba, axis=1)
    top1_outer = base_outer_ticket_mask()[top1]
    actual_outer = actual_1_head_outer_mask(df)
    payouts = pd.to_numeric(df["3rt"], errors="coerce").fillna(0).to_numpy(dtype=float)
    low = payouts < 1500
    rows = []
    for segment in segments:
        mask = segment_mask(df, segment)
        if not mask.any():
            continue
        low_mask = mask & low
        outer_actual_mask = mask & actual_outer
        low_outer_actual_mask = low_mask & actual_outer
        row = {
            "dataset": dataset,
            "model_name": model_name,
            "race_segment": segment,
            "n_races": int(mask.sum()),
            "top1_hit_rate": float(np.mean(ranks[mask] == 1)),
            "top3_contains_rate": float(np.mean(ranks[mask] <= 3)),
            "top5_contains_rate": float(np.mean(ranks[mask] <= 5)),
            "top10_contains_rate": float(np.mean(ranks[mask] <= 10)),
            "mean_actual_rank": float(np.mean(ranks[mask])),
            "logloss": float(np.mean(-np.log(actual_prob[mask]))),
            "brier_score": float(np.mean(brier[mask])),
            "pattern_1_head_outer_actual_rate": float(actual_outer[mask].mean()),
            "top1_1_head_outer_rate": float(top1_outer[mask].mean()),
            "top5_has_1_head_outer_rate": float(top5_has_outer[mask].mean()),
            "top5_1_head_outer_ticket_share": float(top5_outer_share[mask].mean()),
            "actual_1_head_outer_top5_contains_rate": float(np.mean(ranks[outer_actual_mask] <= 5)) if outer_actual_mask.any() else np.nan,
            "low_under1500_n": int(low_mask.sum()),
            "low_under1500_top1_hit_rate": float(np.mean(ranks[low_mask] == 1)) if low_mask.any() else np.nan,
            "low_under1500_top5_contains_rate": float(np.mean(ranks[low_mask] <= 5)) if low_mask.any() else np.nan,
            "low_under1500_mean_actual_rank": float(np.mean(ranks[low_mask])) if low_mask.any() else np.nan,
            "low_under1500_top1_miss_to_hit_vs_baseline": int(np.sum(low_mask & (baseline_ranks != 1) & (ranks == 1))),
            "low_under1500_top5_miss_to_hit_vs_baseline": int(np.sum(low_mask & (baseline_ranks > 5) & (ranks <= 5))),
            "low_under1500_top5_hit_to_miss_vs_baseline": int(np.sum(low_mask & (baseline_ranks <= 5) & (ranks > 5))),
            "low_1_head_outer_actual_n": int(low_outer_actual_mask.sum()),
            "low_1_head_outer_actual_top5_contains_rate": float(np.mean(ranks[low_outer_actual_mask] <= 5)) if low_outer_actual_mask.any() else np.nan,
            "low_1_head_outer_actual_top5_miss_to_hit_vs_baseline": int(np.sum(low_outer_actual_mask & (baseline_ranks > 5) & (ranks <= 5))),
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_all(
    *,
    dataset: str,
    df: pd.DataFrame,
    candidates: list[tuple[str, np.ndarray, dict[str, object]]],
    baseline_proba: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_ranks = actual_rank_fast(baseline_proba, target_indices(df))
    segment_frames = []
    race_frames = []
    low_frames = []
    for name, proba, extra in candidates:
        segment_rows = eval_rows(dataset=dataset, df=df, model_name=name, proba=proba, baseline_ranks=baseline_ranks, segments=SEGMENT_ORDER, extra=extra)
        race_rows = eval_rows(dataset=dataset, df=df, model_name=name, proba=proba, baseline_ranks=baseline_ranks, segments=RACE_NO_ORDER, extra=extra)
        segment_frames.append(segment_rows)
        race_frames.append(race_rows)
        low_frames.append(
            pd.concat(
                [
                    segment_rows[
                        [
                            "dataset",
                            "model_name",
                            "race_segment",
                            "n_races",
                            "low_under1500_n",
                            "low_under1500_top1_hit_rate",
                            "low_under1500_top5_contains_rate",
                            "low_under1500_mean_actual_rank",
                            "low_under1500_top1_miss_to_hit_vs_baseline",
                            "low_under1500_top5_miss_to_hit_vs_baseline",
                            "low_under1500_top5_hit_to_miss_vs_baseline",
                            "low_1_head_outer_actual_n",
                            "low_1_head_outer_actual_top5_contains_rate",
                            "low_1_head_outer_actual_top5_miss_to_hit_vs_baseline",
                            "pattern_1_head_outer_actual_rate",
                            "top5_has_1_head_outer_rate",
                            "top5_1_head_outer_ticket_share",
                        ]
                    ],
                    segment_rows[[c for c in extra.keys() if c in segment_rows.columns]],
                ],
                axis=1,
            )
        )
    return pd.concat(segment_frames, ignore_index=True), pd.concat(race_frames, ignore_index=True), pd.concat(low_frames, ignore_index=True)


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


def write_summary(path: Path, comp: pd.DataFrame, low: pd.DataFrame) -> None:
    all_rows = comp[comp["race_segment"].eq("all_races")].copy()
    focus_cols = [
        "dataset",
        "model_name",
        "top1_hit_rate",
        "top3_contains_rate",
        "top5_contains_rate",
        "top10_contains_rate",
        "mean_actual_rank",
        "logloss",
        "brier_score",
        "top5_1_head_outer_ticket_share",
        "low_under1500_top5_miss_to_hit_vs_baseline",
        "low_under1500_top5_hit_to_miss_vs_baseline",
    ]
    baseline_name = "direct120_jcd_r_top20_rerank"
    baseline_rows = all_rows[all_rows["model_name"].eq(baseline_name)].set_index("dataset")
    for metric in ["top1_hit_rate", "top3_contains_rate", "top5_contains_rate", "top10_contains_rate", "mean_actual_rank", "logloss", "brier_score"]:
        all_rows[f"delta_{metric}_vs_base"] = [
            float(value) - float(baseline_rows.loc[dataset, metric])
            if dataset in baseline_rows.index
            else np.nan
            for dataset, value in zip(all_rows["dataset"], all_rows[metric])
        ]
    c_focus_cols = focus_cols + [
        "delta_top1_hit_rate_vs_base",
        "delta_top5_contains_rate_vs_base",
        "delta_mean_actual_rank_vs_base",
        "delta_logloss_vs_base",
        "boost_pattern",
        "boost_score",
        "clip",
        "top_n",
    ]
    c_focus = all_rows[all_rows["dataset"].eq("C_future_check")].sort_values(["top5_contains_rate", "logloss"], ascending=[False, True])
    final_focus = all_rows[all_rows["dataset"].eq("validation_202603_202605")].sort_values(["top5_contains_rate", "logloss"], ascending=[False, True])
    boost_only = all_rows[all_rows["model_name"].str.contains("outer_in_23", regex=False)].copy()
    c_boost = boost_only[boost_only["dataset"].eq("C_future_check")].sort_values(["top5_contains_rate", "logloss"], ascending=[False, True])
    final_boost = boost_only[boost_only["dataset"].eq("validation_202603_202605")].sort_values(["top5_contains_rate", "logloss"], ascending=[False, True])
    candidate_pairs = c_boost[["model_name"]].drop_duplicates().merge(
        final_boost[["model_name"]].drop_duplicates(),
        on="model_name",
        how="inner",
    )
    adoption_rows = []
    for model_name in candidate_pairs["model_name"].head(40):
        c_row = c_boost[c_boost["model_name"].eq(model_name)].iloc[0]
        f_row = final_boost[final_boost["model_name"].eq(model_name)].iloc[0]
        c_ok = (
            c_row["delta_top1_hit_rate_vs_base"] >= -1e-12
            and c_row["delta_top3_contains_rate_vs_base"] >= -1e-12
            and c_row["delta_top5_contains_rate_vs_base"] > 0
            and c_row["delta_top10_contains_rate_vs_base"] >= -1e-12
            and c_row["delta_mean_actual_rank_vs_base"] <= 0
            and c_row["delta_logloss_vs_base"] <= 0
            and c_row["delta_brier_score_vs_base"] <= 0
        )
        final_not_bad = (
            f_row["delta_top5_contains_rate_vs_base"] >= -1e-12
            and f_row["delta_logloss_vs_base"] <= 0.005
            and f_row["delta_brier_score_vs_base"] <= 0.001
        )
        if c_ok and final_not_bad:
            judgment = "adoption_candidate"
        elif not c_ok:
            judgment = "reject_c_not_improved"
        else:
            judgment = "watch_final_weak"
        adoption_rows.append(
            {
                "judgment": judgment,
                "model_name": model_name,
                "C_top1_delta": c_row["delta_top1_hit_rate_vs_base"],
                "C_top5_delta": c_row["delta_top5_contains_rate_vs_base"],
                "C_mean_rank_delta": c_row["delta_mean_actual_rank_vs_base"],
                "C_logloss_delta": c_row["delta_logloss_vs_base"],
                "C_brier_delta": c_row["delta_brier_score_vs_base"],
                "Final_top5_delta": f_row["delta_top5_contains_rate_vs_base"],
                "Final_logloss_delta": f_row["delta_logloss_vs_base"],
                "top5_outer_share_C": c_row["top5_1_head_outer_ticket_share"],
                "boost_pattern": c_row["boost_pattern"],
                "boost_score": c_row["boost_score"],
                "clip": c_row["clip"],
                "top_n": c_row["top_n"],
            }
        )
    adoption = pd.DataFrame(adoption_rows).sort_values(["judgment", "C_top5_delta", "C_logloss_delta"], ascending=[True, False, True])
    low_final = low[(low["dataset"].eq("validation_202603_202605")) & (low["race_segment"].eq("all_races"))].copy()
    low_final = low_final.sort_values(["low_under1500_top5_miss_to_hit_vs_baseline", "low_under1500_top5_contains_rate"], ascending=[False, False])
    low_c = low[(low["dataset"].eq("C_future_check")) & (low["race_segment"].eq("all_races"))].copy()
    low_c = low_c.sort_values(["low_under1500_top5_miss_to_hit_vs_baseline", "low_under1500_top5_contains_rate"], ascending=[False, False])
    dist_risk = boost_only.sort_values("top5_1_head_outer_ticket_share", ascending=False)

    lines = [
        "# Outer In 2/3 Partner Boost Summary",
        "",
        "## Main Judgment: C区間を主評価",
        "この検証では、2026/3-5ではなくwalk-forward C区間を主評価にします。C区間で改善し、2026/3-5でも悪化しない候補だけを採用候補とします。",
        "",
        "判断ルール:",
        "- C区間でTop1/Top3/Top5/Top10が改善または横ばい、mean_actual_rank/logloss/brierが悪化しないものを重視",
        "- 2026/3-5だけ良い候補は上振れ疑いとして本線候補から外す",
        "- C区間で悪化する候補は、2026/3-5で良くても採用しない",
        "",
        "direct120_jcd_r_top20_rerankをベースに、TopN内の `1号艇頭かつ4-6号艇が2/3着` の買い目だけをログスコア加算しました。1着艇は直接崩さず、2着・3着の外枠絡みを少し上げる補正です。",
        "",
        "## Adoption Candidates By C区間",
        markdown_table(adoption[["judgment", "model_name", "C_top1_delta", "C_top5_delta", "C_mean_rank_delta", "C_logloss_delta", "C_brier_delta", "Final_top5_delta", "Final_logloss_delta", "top5_outer_share_C", "boost_pattern", "boost_score", "clip", "top_n"]], 16),
        "",
        "## C Future Check: All Races",
        markdown_table(c_focus[c_focus_cols], 14),
        "",
        "## 2026/3-5: All Races Reference",
        markdown_table(final_focus[c_focus_cols], 14),
        "",
        "## Best Boost Candidates on C",
        markdown_table(c_boost[c_focus_cols], 12),
        "",
        "## Best Boost Candidates on 2026/3-5 Reference",
        markdown_table(final_boost[c_focus_cols], 12),
        "",
        "## Low Payout Improvement Focus: C区間",
        markdown_table(low_c[["model_name", "race_segment", "low_under1500_n", "low_under1500_top1_hit_rate", "low_under1500_top5_contains_rate", "low_under1500_top5_miss_to_hit_vs_baseline", "low_under1500_top5_hit_to_miss_vs_baseline", "low_1_head_outer_actual_top5_miss_to_hit_vs_baseline"]], 14),
        "",
        "## Low Payout Improvement Focus: 2026/3-5 Reference",
        markdown_table(low_final[["model_name", "race_segment", "low_under1500_n", "low_under1500_top1_hit_rate", "low_under1500_top5_contains_rate", "low_under1500_top5_miss_to_hit_vs_baseline", "low_under1500_top5_hit_to_miss_vs_baseline", "low_1_head_outer_actual_top5_miss_to_hit_vs_baseline"]], 14),
        "",
        "## Distribution Risk",
        "Top5内の1頭外枠絡みチケット比率が高すぎる候補は、外枠絡みを増やしすぎている可能性があります。C区間で改善していても、この比率が極端に上がるものは別列・補助候補として扱います。",
        markdown_table(dist_risk[["dataset", "model_name", "top5_1_head_outer_ticket_share", "top5_has_1_head_outer_rate", "top1_1_head_outer_rate", "boost_pattern", "boost_score", "clip", "top_n"]], 12),
        "",
        "## Reading Notes",
        "- `low_under1500_top5_miss_to_hit_vs_baseline` は、ベースのdirect120_jcd_r_top20でTop5外だった低配当レースが、候補でTop5内に入った件数です。",
        "- 同時に `low_under1500_top5_hit_to_miss_vs_baseline` が増える場合、改善分と入れ替わりの悪化が起きています。",
        "- C区間はB由来補正の未来側確認、2026/3-5は最新環境の参考確認として扱います。",
        "- 採用判断は、C区間のTop1/Top3/Top5/Top10、mean_actual_rank、logloss、brierを優先し、2026/3-5は方向性確認に使います。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def iter_candidates(
    *,
    direct_raw: np.ndarray,
    position_raw: np.ndarray,
    direct_base: np.ndarray,
    pos_weight: np.ndarray,
    position_third: np.ndarray,
    condition_masks: dict[str, np.ndarray],
    top_masks: dict[int, np.ndarray],
):
    yield ("raw_direct120", direct_raw, {"candidate_family": "baseline", "boost_pattern": "", "boost_score": np.nan, "clip": np.nan, "top_n": np.nan})
    yield ("direct120_jcd_r_top20_rerank", direct_base, {"candidate_family": "baseline", "boost_pattern": "", "boost_score": np.nan, "clip": np.nan, "top_n": np.nan})
    yield ("direct120_top20_pos_weight_f0.0_s0.5_t2.0_clip0.1", pos_weight, {"candidate_family": "baseline", "boost_pattern": "", "boost_score": np.nan, "clip": np.nan, "top_n": np.nan})
    yield ("raw_position6", position_raw, {"candidate_family": "baseline", "boost_pattern": "", "boost_score": np.nan, "clip": np.nan, "top_n": np.nan})
    yield ("position6_third_only_correction", position_third, {"candidate_family": "baseline", "boost_pattern": "", "boost_score": np.nan, "clip": np.nan, "top_n": np.nan})
    for pattern in BOOST_PATTERNS:
        for boost_score in BOOST_SCORES:
            for clip in CLIPS:
                for top_n in TOP_NS:
                    name = f"{pattern}_score{boost_score:.2f}_clip{clip:.2f}_top{top_n}"
                    boosted = apply_outer_boost_with_masks(
                        direct_base,
                        condition_mask=condition_masks[pattern],
                        top_mask=top_masks[top_n],
                        boost_score=boost_score,
                        clip=clip,
                    )
                    yield (
                        name,
                        boosted,
                        {
                            "candidate_family": "outer_boost",
                            "boost_pattern": pattern,
                            "boost_score": boost_score,
                            "clip": clip,
                            "top_n": top_n,
                        },
                    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    past = load_featured(args.past_csv, require_registrations=False, name="dataset_1_past")
    recent = load_featured(args.recent_csv, require_registrations=True, name="dataset_2_recent")
    _, b, c = split_past_by_row_order(past, args.split_a_size, args.split_b_size)

    final_cache = out / "final_fixed_correction_cache"
    sensitivity_cache = out / "correction_sensitivity_cache"
    b_direct = np.load(final_cache / "direct120_A_train_B_proba.npy")
    b_position = np.load(final_cache / "position6_A_train_B_proba.npy")

    direct_jcdr = fit_log_ticket_correction(b, b_direct, "jcd_r_correction", k=500, min_count=300)
    direct_pos_jcdr = fit_position_bucket_correction(b, b_direct, "jcd_r_correction", k=500, min_count=300)
    position_global = fit_position_bucket_correction(b, b_position, "global", k=500, min_count=300)

    datasets = [
        (
            "C_future_check",
            c,
            np.load(sensitivity_cache / "direct120_AB_train_C_proba.npy"),
            np.load(sensitivity_cache / "position6_AB_train_C_proba.npy"),
        ),
        (
            "validation_202603_202605",
            recent,
            np.load(final_cache / "direct120_past55_recent_proba.npy"),
            np.load(final_cache / "position6_past55_recent_proba.npy"),
        ),
    ]

    comp_frames = []
    race_frames = []
    low_frames = []
    for dataset_name, df, direct_raw, position_raw in datasets:
        print(f"Evaluating {dataset_name}...")
        direct_base = direct_jcdr.apply(df, direct_raw, strength=1.0, clip=0.60, top_n=20)
        pos_weight = direct_pos_jcdr.apply(df, direct_raw, strengths=(0.0, 0.5, 2.0), clip=0.10, top_n=20)
        position_third = position_global.apply(df, position_raw, strengths=(0.0, 0.0, 1.0), clip=0.60)
        condition_masks = {pattern: outer_condition_matrix(df, pattern) for pattern in BOOST_PATTERNS}
        top_masks = {top_n: rank_mask(direct_base, top_n) for top_n in TOP_NS}
        candidates = iter_candidates(
            direct_raw=direct_raw,
            position_raw=position_raw,
            direct_base=direct_base,
            pos_weight=pos_weight,
            position_third=position_third,
            condition_masks=condition_masks,
            top_masks=top_masks,
        )
        comp, race, low = evaluate_all(dataset=dataset_name, df=df, candidates=candidates, baseline_proba=direct_base)
        comp_frames.append(comp)
        race_frames.append(race)
        low_frames.append(low)

    comp_all = pd.concat(comp_frames, ignore_index=True)
    race_all = pd.concat(race_frames, ignore_index=True)
    low_all = pd.concat(low_frames, ignore_index=True)

    comp_all.to_csv(out / "outer_in_23_boost_comparison.csv", index=False, encoding="utf-8-sig")
    race_all.to_csv(out / "outer_in_23_boost_by_race_no.csv", index=False, encoding="utf-8-sig")
    low_all.to_csv(out / "outer_in_23_boost_low_payout.csv", index=False, encoding="utf-8-sig")
    write_summary(out / "outer_in_23_boost_summary.md", comp_all, low_all)
    print(f"Wrote outer boost outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
