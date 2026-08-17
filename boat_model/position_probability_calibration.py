# -*- coding: utf-8 -*-
"""Codex側`boat_model/position_probability_calibration.py`のサイト用移植版(内容は
学習・推論の両方で共通のため、変更なくそのままコピーしている)。
artifacts/all_head_hierarchical/position_boat_isotonic_calibrators.joblibと
outer_head_ticket_residual_calibrator.joblibはこのモジュールのクラスを
pickle参照しているため、モジュールパスとクラス定義をCodex側と完全に
一致させる必要がある。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class ClasswiseIsotonicNormalizer:
    """Calibrate each boat monotonically, then restore a valid race distribution."""

    calibrators: list[IsotonicRegression]
    n_samples_by_boat: list[int]

    @classmethod
    def fit(
        cls,
        probability: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> "ClasswiseIsotonicNormalizer":
        probability = np.asarray(probability, dtype=np.float64)
        target = np.asarray(target, dtype=np.int64)
        if probability.ndim != 2 or probability.shape[1] != 6:
            raise ValueError(f"Expected an (n, 6) probability matrix, got {probability.shape}")
        if valid_mask is None:
            valid_mask = np.ones_like(probability, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        calibrators: list[IsotonicRegression] = []
        counts: list[int] = []
        for boat_index in range(6):
            mask = valid_mask[:, boat_index]
            if not np.any(mask):
                raise ValueError(f"Boat {boat_index + 1} has no valid calibration samples")
            model = IsotonicRegression(y_min=1e-8, y_max=1.0, out_of_bounds="clip")
            model.fit(
                probability[mask, boat_index],
                (target[mask] == boat_index).astype(np.float64),
            )
            calibrators.append(model)
            counts.append(int(mask.sum()))
        return cls(calibrators=calibrators, n_samples_by_boat=counts)

    def predict(
        self,
        probability: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        probability = np.asarray(probability, dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] != 6:
            raise ValueError(f"Expected an (n, 6) probability matrix, got {probability.shape}")
        if valid_mask is None:
            valid_mask = np.ones_like(probability, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        calibrated = np.zeros_like(probability, dtype=np.float64)
        for boat_index, model in enumerate(self.calibrators):
            mask = valid_mask[:, boat_index]
            calibrated[mask, boat_index] = model.predict(probability[mask, boat_index])
        row_sum = calibrated.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise RuntimeError("Classwise calibration produced an empty probability row")
        calibrated /= row_sum
        calibrated[~valid_mask] = 0.0
        if not np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError("Calibrated position probabilities do not sum to one")
        return calibrated


def apply_calibrator_along_last_axis(
    probability: np.ndarray,
    calibrator: ClasswiseIsotonicNormalizer,
) -> np.ndarray:
    """Apply a six-boat calibrator to every non-empty distribution in a tensor."""

    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape[-1] != 6:
        raise ValueError(f"Last axis must contain six boats, got {probability.shape}")
    flat = probability.reshape(-1, 6)
    valid = flat > 0
    usable = valid.sum(axis=1) > 0
    output = np.zeros_like(flat)
    output[usable] = calibrator.predict(flat[usable], valid[usable])
    return output.reshape(probability.shape)


@dataclass
class OuterHeadResidualCalibrator:
    """Shrunken monotone residual calibration for tickets headed by boats 4-6."""

    model: IsotonicRegression
    bucket_edges: np.ndarray
    fit_table: list[dict[str, float | int]]
    shrink_k: float

    @classmethod
    def fit(
        cls,
        probability: np.ndarray,
        target_class: np.ndarray,
        tickets: tuple[tuple[int, int, int], ...],
        *,
        shrink_k: float = 5000.0,
    ) -> "OuterHeadResidualCalibrator":
        probability = np.asarray(probability, dtype=np.float64)
        target_class = np.asarray(target_class, dtype=np.int64)
        outer_classes = np.asarray([index for index, ticket in enumerate(tickets) if ticket[0] >= 4])
        values = probability[:, outer_classes].reshape(-1)
        edges = np.asarray([
            0.0, .0005, .001, .0025, .005, .0075, .01, .015, .02, .03,
            .05, .07, .10, .15, 1.000001,
        ])
        bins = np.clip(np.digitize(values, edges, right=False) - 1, 0, len(edges) - 2)
        class_to_local = {int(class_id): position for position, class_id in enumerate(outer_classes)}
        hit_bins = np.full(len(target_class), -1, dtype=np.int16)
        for row, class_id in enumerate(target_class):
            local = class_to_local.get(int(class_id))
            if local is not None:
                hit_bins[row] = bins[row * len(outer_classes) + local]

        points = []
        targets = []
        weights = []
        table: list[dict[str, float | int]] = []
        for index in range(len(edges) - 1):
            mask = bins == index
            n = int(mask.sum())
            if not n:
                continue
            hits = int(np.sum(hit_bins == index))
            predicted = float(values[mask].mean())
            actual = hits / n
            shrink_weight = n / (n + shrink_k)
            shrunk = shrink_weight * actual + (1.0 - shrink_weight) * predicted
            points.append(predicted)
            targets.append(shrunk)
            weights.append(n)
            table.append({
                "bucket_lower": float(edges[index]),
                "bucket_upper": float(edges[index + 1]),
                "n_predictions": n,
                "hit_count": hits,
                "avg_predicted_probability": predicted,
                "actual_hit_rate": actual,
                "shrink_weight": shrink_weight,
                "shrunk_target": shrunk,
            })
        model = IsotonicRegression(y_min=1e-12, y_max=1.0, out_of_bounds="clip")
        model.fit(np.asarray(points), np.asarray(targets), sample_weight=np.asarray(weights))
        return cls(model=model, bucket_edges=edges, fit_table=table, shrink_k=float(shrink_k))

    def predict(
        self,
        probability: np.ndarray,
        tickets: tuple[tuple[int, int, int], ...],
    ) -> np.ndarray:
        probability = np.asarray(probability, dtype=np.float64)
        output = probability.copy()
        outer_classes = np.asarray([index for index, ticket in enumerate(tickets) if ticket[0] >= 4])
        output[:, outer_classes] = self.model.predict(output[:, outer_classes].reshape(-1)).reshape(
            len(output), len(outer_classes)
        )
        output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-15)
        if not np.allclose(output.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError("Outer-head residual probabilities do not sum to one")
        return output
