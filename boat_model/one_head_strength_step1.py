# -*- coding: utf-8 -*-
"""Codex側all_head_hierarchicalモデルの推論に必要な基礎定義(FeatureEncoder等)の
サイト用移植版。

Codexプロジェクト側`boat_model/one_head_strength_step1.py`から、推論(predict)に
必要な部分だけを抜き出した縮小版。artifacts/all_head_hierarchical/配下の
joblibファイルはこのモジュールの`FeatureEncoder`クラスをpickle参照している
ため、モジュールパス(boat_model.one_head_strength_step1)とクラス定義を
Codex側と完全に一致させる必要がある(学習用コードは意図的に含めていない)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

BOATS = tuple(range(1, 7))
CANDIDATE_BOATS = tuple(range(2, 7))

BASE_COLUMNS = [
    "date",
    "jcd",
    "r",
    "race_距離",
    "race_天候",
    "race_風向",
    "race_風速",
    "race_波高",
    "3連単_組",
]

BOAT_COLUMNS = [
    "年齢",
    "支部",
    "体重",
    "級別",
    "全国勝率",
    "全国2率",
    "当地勝率",
    "当地2率",
    "モーター2率",
    "ボート2率",
    "着順",
]

METRICS = {
    "national_winrate": "全国勝率",
    "national_2rate": "全国2率",
    "local_winrate": "当地勝率",
    "local_2rate": "当地2率",
    "motor_2rate": "モーター2率",
    "boat_2rate": "ボート2率",
    "age": "年齢",
    "weight": "体重",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.float32)
    medians = np.nanmedian(work, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(work), work, medians[:, None])
    order = np.argsort(-filled, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int8)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, 7, dtype=np.int8)
    return ranks


@dataclass
class FeatureEncoder:
    feature_columns: list[str]
    categorical_columns: list[str]
    category_maps: dict[str, dict[str, int]]
    medians: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: Sequence[str], categorical_columns: Sequence[str]) -> "FeatureEncoder":
        categorical = [column for column in categorical_columns if column in feature_columns]
        maps: dict[str, dict[str, int]] = {}
        medians: dict[str, float] = {}
        for column in feature_columns:
            if column in categorical:
                values = frame[column].astype("string").fillna("__MISSING__")
                maps[column] = {value: index for index, value in enumerate(sorted(values.unique().tolist()))}
            else:
                values = pd.to_numeric(frame[column], errors="coerce")
                median = values.median()
                medians[column] = (
                    float(median)
                    if pd.notna(median) and np.isfinite(float(median))
                    else 0.0
                )
        return cls(list(feature_columns), categorical, maps, medians)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.empty((len(frame), len(self.feature_columns)), dtype=np.float32)
        for index, column in enumerate(self.feature_columns):
            if column in self.category_maps:
                values = frame[column].astype("string").fillna("__MISSING__")
                matrix[:, index] = values.map(self.category_maps[column]).fillna(-1).to_numpy(dtype=np.float32)
            else:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column])
                matrix[:, index] = values.to_numpy(dtype=np.float32)
        return matrix
