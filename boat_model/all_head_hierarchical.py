# -*- coding: utf-8 -*-
"""Codex側`boat_model/all_head_hierarchical.py`のサイト用移植版。

推論(predict)に必要な部分だけを抜き出した縮小版で、学習用コード
(train_first_model/train_choice_model/run_development等)は含めていない。
artifacts/all_head_hierarchical/配下のfirst6_model.joblib・
conditional_second_model.joblib・conditional_third_model.joblibは
このモジュールの`EncodedClassifier`クラスをpickle参照しているため、
モジュールパス(boat_model.all_head_hierarchical)とクラス定義を
Codex側と完全に一致させる必要がある。

【Codex側との唯一の意図的な差分】
`build_choice_frame`は元々、学習・過去期間の評価専用に書かれており、
races側に`actual_second_boat`/`actual_third_boat`(=正解ラベル)が
必ず存在する前提で`choice_label`列を計算していた。サイトの実運用では
未来のレース(結果が存在しない)を予測する必要があるため、これらの列が
存在しない場合はchoice_label計算をスキップするよう変更している
(choice_label自体は学習用の目的変数であり、推論時にはモデルへの入力
特徴量として一切使われない列なので、この変更は確率計算の結果に
影響しない)。
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from boat_model.one_head_strength_step1 import BOATS, METRICS, FeatureEncoder

warnings.filterwarnings("ignore", message="X does not have valid feature names")


TICKETS = tuple(itertools.permutations(BOATS, 3))
TICKET_TO_CLASS = {ticket: index for index, ticket in enumerate(TICKETS)}


@dataclass
class EncodedClassifier:
    encoder: FeatureEncoder
    classifier: object
    features: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.classifier.predict_proba(self.encoder.transform(frame)), dtype=np.float64)
        return np.clip(probability, 1e-12, 1.0)


CHOICE_METADATA = {
    "race_id", "date", "year", "race_pos", "candidate_boat", "choice_label",
}


def _choice_common(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "race_id", "date", "year", "month", "jcd", "race_no", "month_sin", "month_cos",
        "weather", "wind_direction", "race_distance", "wind_speed", "wave_height",
    ]
    for metric in METRICS:
        columns.extend([
            f"field_{metric}_mean", f"field_{metric}_std", f"field_{metric}_range",
            f"inner_{metric}_mean", f"outer_{metric}_mean", f"inner_minus_outer_{metric}",
        ])
    output = features[columns].copy().reset_index(drop=True)
    output["race_pos"] = np.arange(len(output), dtype=np.int32)
    return output


def _feature_matrices(features: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    values = {
        metric: np.column_stack([features[f"boat{boat}_{metric}"].to_numpy() for boat in BOATS])
        for metric in METRICS
    }
    ranks = {
        metric: np.column_stack([features[f"boat{boat}_{metric}_rank"].to_numpy() for boat in BOATS])
        for metric in METRICS
    }
    classes = np.column_stack([
        features[f"boat{boat}_class"].astype("string").to_numpy() for boat in BOATS
    ])
    return values, ranks, classes


@dataclass
class ChoiceCache:
    common: pd.DataFrame
    values: dict[str, np.ndarray]
    ranks: dict[str, np.ndarray]
    classes: np.ndarray


def prepare_choice_cache(features: pd.DataFrame) -> ChoiceCache:
    values, ranks, classes = _feature_matrices(features)
    return ChoiceCache(_choice_common(features), values, ranks, classes)


def build_choice_frame(
    races: pd.DataFrame,
    features: pd.DataFrame,
    *,
    stage: str,
    fixed_first: int | None = None,
    fixed_second: int | None = None,
    cache: ChoiceCache | None = None,
) -> pd.DataFrame:
    if stage not in {"second", "third"}:
        raise ValueError(stage)
    n = len(features)
    cache = cache or prepare_choice_cache(features)
    common = cache.common
    values, ranks, classes = cache.values, cache.ranks, cache.classes
    if fixed_first is None:
        first = races["actual_first_boat"].to_numpy(dtype=np.int8)
    else:
        first = np.full(n, fixed_first, dtype=np.int8)
    if stage == "third":
        if fixed_second is None:
            second = races["actual_second_boat"].to_numpy(dtype=np.int8)
        else:
            second = np.full(n, fixed_second, dtype=np.int8)
    else:
        second = np.zeros(n, dtype=np.int8)

    actual_col = "actual_second_boat" if stage == "second" else "actual_third_boat"
    has_actual = actual_col in races.columns

    row = np.arange(n)
    parts = []
    for candidate in BOATS:
        mask = candidate != first
        if stage == "third":
            mask &= candidate != second
        positions = np.flatnonzero(mask)
        part = common.iloc[positions].copy()
        first_idx = first[positions] - 1
        candidate_idx = candidate - 1
        part["first_boat_cat"] = pd.Series(first[positions], index=part.index).astype("string")
        part["candidate_boat"] = np.int8(candidate)
        part["candidate_boat_cat"] = str(candidate)
        part["first_class"] = classes[positions, first_idx]
        part["candidate_class"] = classes[positions, candidate_idx]
        part["candidate_is_outer"] = np.float32(candidate >= 4)
        part["candidate_lane_distance_from_first"] = np.abs(candidate - first[positions]).astype(np.float32)
        if stage == "third":
            second_idx = second[positions] - 1
            part["second_boat_cat"] = pd.Series(second[positions], index=part.index).astype("string")
            part["second_class"] = classes[positions, second_idx]
        for metric in METRICS:
            matrix = values[metric]
            rank_matrix = ranks[metric]
            first_value = matrix[positions, first_idx]
            candidate_value = matrix[positions, candidate_idx]
            part[f"first_{metric}"] = first_value
            part[f"candidate_{metric}"] = candidate_value
            part[f"candidate_minus_first_{metric}"] = candidate_value - first_value
            part[f"first_{metric}_rank"] = rank_matrix[positions, first_idx]
            part[f"candidate_{metric}_rank"] = rank_matrix[positions, candidate_idx]
            if stage == "third":
                second_value = matrix[positions, second_idx]
                part[f"second_{metric}"] = second_value
                part[f"candidate_minus_second_{metric}"] = candidate_value - second_value
                part[f"second_{metric}_rank"] = rank_matrix[positions, second_idx]
        if has_actual:
            actual = races.iloc[positions][actual_col].to_numpy(dtype=np.int8)
            part["choice_label"] = (actual == candidate).astype(np.int8)
        else:
            # 未来レース予測(結果不明)向けの拡張。choice_labelは学習専用の
            # 目的変数でモデル入力には使われないため、ダミー値でも
            # predict_proba()の結果には一切影響しない。
            part["choice_label"] = np.int8(-1)
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    return output.sort_values(["race_pos", "candidate_boat"], kind="stable").reset_index(drop=True)


def predict_second_tensor(
    model: EncodedClassifier,
    races: pd.DataFrame,
    features: pd.DataFrame,
) -> np.ndarray:
    n = len(features)
    tensor = np.zeros((n, 6, 6), dtype=np.float64)
    cache = prepare_choice_cache(features)
    for first in BOATS:
        choice = build_choice_frame(races, features, stage="second", fixed_first=first, cache=cache)
        raw = model.predict_proba(choice)[:, 1]
        tensor[
            choice["race_pos"].to_numpy(dtype=np.int64),
            first - 1,
            choice["candidate_boat"].to_numpy(dtype=np.int64) - 1,
        ] = raw
    tensor /= np.maximum(tensor.sum(axis=2, keepdims=True), 1e-15)
    return tensor


def predict_third_tensor(
    model: EncodedClassifier,
    races: pd.DataFrame,
    features: pd.DataFrame,
) -> np.ndarray:
    n = len(features)
    tensor = np.zeros((n, 6, 6, 6), dtype=np.float64)
    cache = prepare_choice_cache(features)
    for first in BOATS:
        for second in BOATS:
            if second == first:
                continue
            choice = build_choice_frame(
                races, features, stage="third", fixed_first=first, fixed_second=second, cache=cache
            )
            raw = model.predict_proba(choice)[:, 1]
            tensor[
                choice["race_pos"].to_numpy(dtype=np.int64),
                first - 1,
                second - 1,
                choice["candidate_boat"].to_numpy(dtype=np.int64) - 1,
            ] = raw
    tensor /= np.maximum(tensor.sum(axis=3, keepdims=True), 1e-15)
    return tensor


def temperature_scale(probability: np.ndarray, temperature: float, axis: int) -> np.ndarray:
    powered = np.power(np.clip(probability, 1e-15, 1.0), 1.0 / temperature)
    powered = np.where(probability > 0, powered, 0.0)
    return powered / np.maximum(powered.sum(axis=axis, keepdims=True), 1e-15)


def compose_ticket_probabilities(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(first), len(TICKETS)), dtype=np.float64)
    for class_id, (a, b, c) in enumerate(TICKETS):
        output[:, class_id] = first[:, a - 1] * second[:, a - 1, b - 1] * third[:, a - 1, b - 1, c - 1]
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-15)
    return output
