from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd


BOATS = [1, 2, 3, 4, 5, 6]
WINRATE_COLS = [f"勝率{i}" for i in BOATS]
REGISTRATION_COLS = [f"登番{i}" for i in BOATS]

TRIFECTA_PERMUTATIONS = list(permutations(BOATS, 3))
TRIFECTA_TO_INDEX = {perm: idx for idx, perm in enumerate(TRIFECTA_PERMUTATIONS)}
INDEX_TO_TRIFECTA = {
    idx: f"{perm[0]}-{perm[1]}-{perm[2]}"
    for idx, perm in enumerate(TRIFECTA_PERMUTATIONS)
}
TRIFECTA_STR_TO_INDEX = {v: k for k, v in INDEX_TO_TRIFECTA.items()}


def normalize_registration_series(series: pd.Series) -> pd.Series:
    """登録番号をCSV由来の 4638.0 / 4638 / 空値 の揺れに強い文字列へそろえる。"""
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.mask(out.isin(["", "<NA>", "nan", "None"]), pd.NA)
    return out


def trifecta_string_from_row(row: pd.Series) -> str:
    return f"{int(row['r1'])}-{int(row['r2'])}-{int(row['r3'])}"


def actual_trifecta_strings(df: pd.DataFrame) -> pd.Series:
    return df[["r1", "r2", "r3"]].astype(int).astype(str).agg("-".join, axis=1)


def target_indices(df: pd.DataFrame) -> np.ndarray:
    values = []
    for r1, r2, r3 in df[["r1", "r2", "r3"]].itertuples(index=False, name=None):
        values.append(TRIFECTA_TO_INDEX[(int(r1), int(r2), int(r3))])
    return np.asarray(values, dtype=np.int64)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """勝率構成、場、レース番号から使う基礎特徴量を作る。"""
    out = df.copy()
    rates = out[WINRATE_COLS].apply(pd.to_numeric, errors="coerce")

    # 欠損勝率は特徴量計算時だけ列中央値で補完する。元列は補完済み値にそろえる。
    medians = rates.median()
    rates = rates.fillna(medians).fillna(0.0)
    out[WINRATE_COLS] = rates

    out["勝率平均"] = rates.mean(axis=1)
    out["勝率最大"] = rates.max(axis=1)
    out["勝率最小"] = rates.min(axis=1)
    out["勝率標準偏差"] = rates.std(axis=1, ddof=0)
    out["勝率レンジ"] = out["勝率最大"] - out["勝率最小"]

    rank_values = rates.rank(axis=1, ascending=False, method="min").astype(int)
    for boat in BOATS:
        out[f"勝率順位{boat}"] = rank_values[f"勝率{boat}"]

    out["1号艇が勝率1位か"] = (out["勝率順位1"] == 1).astype(int)
    out["1号艇の勝率順位"] = out["勝率順位1"]

    for boat in BOATS[1:]:
        out[f"勝率1_minus_勝率{boat}"] = out["勝率1"] - out[f"勝率{boat}"]

    for left, right in zip(BOATS[:-1], BOATS[1:]):
        out[f"adjacent_diff_勝率{left}_minus_勝率{right}"] = (
            out[f"勝率{left}"] - out[f"勝率{right}"]
        )

    # 同点は艇番の若い順に安定化する。
    rate_array = rates.to_numpy(dtype=float)
    boat_array = np.asarray(BOATS)
    row_order = np.argsort(-rate_array + boat_array.reshape(1, -1) * 1e-12, axis=1)
    ordered_boats = boat_array[row_order]
    out["勝率上位3艇"] = ["-".join(map(str, row[:3])) for row in ordered_boats]
    out["勝率順位パターン"] = ["-".join(map(str, row)) for row in ordered_boats]

    out["jcd"] = out["jcd"].astype("string").str.zfill(2)
    out["r"] = pd.to_numeric(out["r"], errors="coerce").astype("Int64")
    return out


def add_racer_master_features(
    df: pd.DataFrame,
    racer_master: pd.DataFrame,
) -> pd.DataFrame:
    """登番から各艇のコース別成績を結合する。

    racer_master は2026年6月更新の可能性があるため、この関数を使うモデルは
    評価表で leakage risk ありとして扱う。
    """
    out = df.copy()
    master = racer_master.copy()
    master["登録番号"] = normalize_registration_series(master["登録番号"])
    master = master.dropna(subset=["登録番号"]).drop_duplicates("登録番号", keep="last")
    master = master.set_index("登録番号")

    metric_map = {
        "course_win_rate": "1着率",
        "course_second_rate": "2着率",
        "course_third_rate": "3着率",
        "course_avg_st": "平均ST",
        "course_st_rank": "ST順",
    }

    for boat in BOATS:
        reg_col = f"登番{boat}"
        if reg_col in out.columns:
            registrations = normalize_registration_series(out[reg_col])
        else:
            registrations = pd.Series(pd.NA, index=out.index, dtype="string")

        for dest_suffix, src_suffix in metric_map.items():
            src_col = f"{boat}コース_{src_suffix}"
            dest_col = f"boat{boat}_{dest_suffix}"
            if src_col not in master.columns:
                out[dest_col] = 0.0
                continue
            values = registrations.map(master[src_col])
            fill_value = pd.to_numeric(master[src_col], errors="coerce").mean()
            out[dest_col] = pd.to_numeric(values, errors="coerce").fillna(fill_value).fillna(0.0)

    win_rate_cols = [f"boat{i}_course_win_rate" for i in BOATS]
    avg_st_cols = [f"boat{i}_course_avg_st" for i in BOATS]

    win_rank = out[win_rate_cols].rank(axis=1, ascending=False, method="min").astype(int)
    st_rank = out[avg_st_cols].rank(axis=1, ascending=True, method="min").astype(int)
    for boat in BOATS:
        out[f"boat{boat}_course_win_rate_rank"] = win_rank[f"boat{boat}_course_win_rate"]
        out[f"boat{boat}_avg_st_rank"] = st_rank[f"boat{boat}_course_avg_st"]

    for boat in BOATS[1:]:
        out[f"boat1_course_win_rate_minus_boat{boat}"] = (
            out["boat1_course_win_rate"] - out[f"boat{boat}_course_win_rate"]
        )

    out["boat1_avg_st_minus_boat2"] = out["boat1_course_avg_st"] - out["boat2_course_avg_st"]
    out["boat1_avg_st_minus_boat3"] = out["boat1_course_avg_st"] - out["boat3_course_avg_st"]
    out["boat1_st_rank_minus_boat2"] = out["boat1_course_st_rank"] - out["boat2_course_st_rank"]
    out["boat1_st_rank_minus_boat3"] = out["boat1_course_st_rank"] - out["boat3_course_st_rank"]
    return out


BASIC_NUMERIC_FEATURES = (
    WINRATE_COLS
    + ["勝率平均", "勝率最大", "勝率最小", "勝率標準偏差", "勝率レンジ"]
    + [f"勝率順位{i}" for i in BOATS]
    + ["1号艇が勝率1位か", "1号艇の勝率順位"]
    + [f"勝率1_minus_勝率{i}" for i in BOATS[1:]]
    + [f"adjacent_diff_勝率{i}_minus_勝率{i + 1}" for i in BOATS[:-1]]
)

BASIC_CATEGORICAL_FEATURES = ["jcd", "r", "勝率上位3艇", "勝率順位パターン"]

RACER_MASTER_FEATURES = (
    [
        f"boat{boat}_{metric}"
        for boat in BOATS
        for metric in [
            "course_win_rate",
            "course_second_rate",
            "course_third_rate",
            "course_avg_st",
            "course_st_rank",
            "course_win_rate_rank",
            "avg_st_rank",
        ]
    ]
    + [f"boat1_course_win_rate_minus_boat{i}" for i in BOATS[1:]]
    + [
        "boat1_avg_st_minus_boat2",
        "boat1_avg_st_minus_boat3",
        "boat1_st_rank_minus_boat2",
        "boat1_st_rank_minus_boat3",
    ]
)

MODEL_C_FEATURES = BASIC_NUMERIC_FEATURES + BASIC_CATEGORICAL_FEATURES
KNN_FEATURES = BASIC_NUMERIC_FEATURES


@dataclass
class FeatureEncoder:
    """pandas DataFrame を純numpyモデルでも扱える数値行列に変換する。"""

    feature_columns: list[str]
    categorical_columns: list[str]
    standardize: bool = True

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self.category_maps_: dict[str, dict[str, int]] = {}
        for col in self.categorical_columns:
            values = df[col].astype("string").fillna("__MISSING__")
            categories = sorted(values.unique().tolist())
            self.category_maps_[col] = {value: idx for idx, value in enumerate(categories)}

        matrix = self._raw_matrix(df)
        self.medians_ = np.nanmedian(matrix, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians_)
        self.means_ = matrix.mean(axis=0)
        self.stds_ = matrix.std(axis=0)
        self.stds_ = np.where(self.stds_ > 1e-12, self.stds_, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._raw_matrix(df)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians_)
        if self.standardize:
            matrix = (matrix - self.means_) / self.stds_
        return matrix.astype(np.float64, copy=False)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def _raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.feature_columns:
            if col in self.categorical_columns:
                mapping = getattr(self, "category_maps_", {}).get(col, {})
                values = (
                    df[col]
                    .astype("string")
                    .fillna("__MISSING__")
                    .map(mapping)
                    .fillna(-1)
                    .astype(float)
                )
            else:
                values = pd.to_numeric(df[col], errors="coerce")
            cols.append(values.to_numpy(dtype=float))
        return np.column_stack(cols)


def normalize_probability_matrix(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.where(np.isfinite(proba), proba, 0.0)
    proba = np.maximum(proba, eps)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return proba / row_sum


def scores_to_trifecta_proba(boat_scores: np.ndarray) -> np.ndarray:
    """6艇スコアを Plackett-Luce 風に120通りの3連単確率へ変換する。"""
    scores = np.asarray(boat_scores, dtype=float)
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    total = exp_scores.sum(axis=1)

    out = np.zeros((scores.shape[0], len(TRIFECTA_PERMUTATIONS)), dtype=float)
    for idx, (first, second, third) in enumerate(TRIFECTA_PERMUTATIONS):
        a = exp_scores[:, first - 1]
        b = exp_scores[:, second - 1]
        c = exp_scores[:, third - 1]
        p1 = a / total
        p2 = b / np.maximum(total - a, 1e-12)
        p3 = c / np.maximum(total - a - b, 1e-12)
        out[:, idx] = p1 * p2 * p3
    return normalize_probability_matrix(out)


def top_prediction_strings(proba: np.ndarray, top_n: int = 5, with_probability: bool = True) -> list[str]:
    order = np.argsort(-proba, axis=1)[:, :top_n]
    rows = []
    for row_idx, top_indices in enumerate(order):
        values = []
        for class_idx in top_indices:
            label = INDEX_TO_TRIFECTA[int(class_idx)]
            if with_probability:
                values.append(f"{label}:{proba[row_idx, class_idx]:.6f}")
            else:
                values.append(label)
        rows.append("|".join(values))
    return rows


def feature_list(existing_df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in existing_df.columns]
