# -*- coding: utf-8 -*-
"""Codex側all_head_hierarchicalモデルの推論に必要な特徴量構築(build_race_features)の
サイト用移植版。

Codexプロジェクト側`boat_model/boat1_win_stage.py`から、推論に必要な
`build_race_features`/`feature_columns`/`categorical_columns`だけを移植した
縮小版(学習・データ読み込み用コードは含めていない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from boat_model.one_head_strength_step1 import BASE_COLUMNS, BOATS, BOAT_COLUMNS, METRICS, _numeric, _rank_desc

RACE_METADATA = ["race_id", "date", "year", "month", "boat1_win"]


def build_race_features(races: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(races["date"])
    frame = pd.DataFrame({
        "race_id": races["race_id"].astype("string"),
        "date": date,
        "year": date.dt.year.astype(np.int16),
        "month": date.dt.month.astype(np.int8),
        "boat1_win": races["boat1_win"].astype(np.int8),
        "jcd": races["jcd"].astype("string").str.zfill(2),
        "race_no": pd.to_numeric(races["r"], errors="coerce").astype("Int64").astype("string"),
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12).astype(np.float32),
        "weather": races[BASE_COLUMNS[4]].astype("string").str.strip(),
        "wind_direction": races[BASE_COLUMNS[5]].astype("string").str.strip(),
        "race_distance": _numeric(races[BASE_COLUMNS[3]]),
        "wind_speed": _numeric(races[BASE_COLUMNS[6]]),
        "wave_height": _numeric(races[BASE_COLUMNS[7]]),
    })

    for boat in BOATS:
        frame[f"boat{boat}_class"] = races[f"boat{boat}_{BOAT_COLUMNS[3]}"].astype("string").str.strip()
        frame[f"boat{boat}_branch"] = races[f"boat{boat}_{BOAT_COLUMNS[1]}"].astype("string").str.strip()

    for metric, suffix in METRICS.items():
        values = np.column_stack([_numeric(races[f"boat{boat}_{suffix}"]).to_numpy() for boat in BOATS])
        ranks = _rank_desc(values)
        row_mean = np.nanmean(values, axis=1)
        row_std = np.nanstd(values, axis=1)
        row_range = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        for boat in BOATS:
            frame[f"boat{boat}_{metric}"] = values[:, boat - 1]
            frame[f"boat{boat}_{metric}_rank"] = ranks[:, boat - 1].astype(np.float32)
            if boat > 1:
                frame[f"boat1_minus_boat{boat}_{metric}"] = values[:, 0] - values[:, boat - 1]
        frame[f"field_{metric}_mean"] = row_mean
        frame[f"field_{metric}_std"] = row_std
        frame[f"field_{metric}_range"] = row_range
        frame[f"inner_{metric}_mean"] = np.nanmean(values[:, :3], axis=1)
        frame[f"outer_{metric}_mean"] = np.nanmean(values[:, 3:], axis=1)
        frame[f"inner_minus_outer_{metric}"] = frame[f"inner_{metric}_mean"] - frame[f"outer_{metric}_mean"]
        frame[f"boat1_{metric}_is_top"] = (ranks[:, 0] == 1).astype(np.float32)
        frame[f"boat1_{metric}_margin_best_other"] = values[:, 0] - np.nanmax(values[:, 1:], axis=1)
    return frame.copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in RACE_METADATA]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if column in {"jcd", "race_no", "weather", "wind_direction"}
        or column.endswith("_class")
        or column.endswith("_branch")
    ]
