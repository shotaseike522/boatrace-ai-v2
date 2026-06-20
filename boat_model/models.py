from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    BASIC_CATEGORICAL_FEATURES,
    BASIC_NUMERIC_FEATURES,
    BOATS,
    INDEX_TO_TRIFECTA,
    KNN_FEATURES,
    MODEL_C_FEATURES,
    RACER_MASTER_FEATURES,
    TRIFECTA_PERMUTATIONS,
    TRIFECTA_TO_INDEX,
    FeatureEncoder,
    actual_trifecta_strings,
    feature_list,
    normalize_probability_matrix,
    scores_to_trifecta_proba,
    target_indices,
)


@dataclass
class ModelPrediction:
    model_name: str
    proba: np.ndarray
    leakage_risk: bool
    model_type: str
    notes: str = ""


def blend_probabilities(knn_proba: np.ndarray, pairwise_proba: np.ndarray, knn_weight: float) -> np.ndarray:
    """KNNとpairwiseの3連単120通り確率を重み付き平均する。"""
    knn_weight = float(knn_weight)
    pairwise_weight = 1.0 - knn_weight
    blended = knn_weight * knn_proba + pairwise_weight * pairwise_proba
    return normalize_probability_matrix(blended)


class RuleWinrateModel:
    def __init__(self, top_probability: float = 0.35) -> None:
        self.model_name = "A_winrate_rule"
        self.leakage_risk = False
        self.top_probability = top_probability

    def fit(self, df: pd.DataFrame) -> "RuleWinrateModel":
        labels = target_indices(df)
        counts = np.bincount(labels, minlength=len(TRIFECTA_PERMUTATIONS)).astype(float) + 1.0
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        rates = df[[f"勝率{i}" for i in BOATS]].to_numpy(dtype=float)
        boat_array = np.asarray(BOATS)
        order = np.argsort(-rates + boat_array.reshape(1, -1) * -1e-12, axis=1)
        ordered_boats = boat_array[order]
        top_indices = [
            TRIFECTA_TO_INDEX[tuple(int(x) for x in row[:3])]
            for row in ordered_boats
        ]
        proba = np.tile(self.prior_ * (1.0 - self.top_probability), (len(df), 1))
        proba[np.arange(len(df)), top_indices] += self.top_probability
        return normalize_probability_matrix(proba)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame()


class ApproxRaceKNNModel:
    def __init__(self, k: int, weighted: bool) -> None:
        self.k = int(k)
        self.weighted = bool(weighted)
        suffix = "weighted" if weighted else "count"
        self.model_name = f"B_knn_k{self.k}_{suffix}"
        self.leakage_risk = False

    def fit(self, df: pd.DataFrame) -> "ApproxRaceKNNModel":
        self.feature_columns_ = feature_list(df, KNN_FEATURES)
        encoder = FeatureEncoder(
            feature_columns=self.feature_columns_,
            categorical_columns=[],
            standardize=True,
        )
        self.encoder_ = encoder
        self.X_ = encoder.fit_transform(df).astype(np.float32)
        self.labels_ = target_indices(df)
        self.all_indices_ = np.arange(len(df), dtype=np.int64)

        group_jcd_r = df.groupby(["jcd", "r"], dropna=False).indices
        self.group_jcd_r_ = {(str(k[0]), int(k[1])): np.asarray(v, dtype=np.int64) for k, v in group_jcd_r.items()}
        self.group_jcd_ = {
            str(k): np.asarray(v, dtype=np.int64)
            for k, v in df.groupby("jcd", dropna=False).indices.items()
        }
        self.group_r_ = {
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in df.groupby("r", dropna=False).indices.items()
        }
        return self

    def _candidate_indices(self, row: pd.Series) -> np.ndarray:
        key = (str(row["jcd"]).zfill(2), int(row["r"]))
        candidates = self.group_jcd_r_.get(key)
        if candidates is not None and len(candidates) >= self.k:
            return candidates

        pieces = []
        if candidates is not None:
            pieces.append(candidates)
        jcd_candidates = self.group_jcd_.get(str(row["jcd"]).zfill(2))
        if jcd_candidates is not None:
            pieces.append(jcd_candidates)
        r_candidates = self.group_r_.get(int(row["r"]))
        if r_candidates is not None:
            pieces.append(r_candidates)

        if pieces:
            merged = np.unique(np.concatenate(pieces))
            if len(merged) >= self.k:
                return merged
        return self.all_indices_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_eval = self.encoder_.transform(df).astype(np.float32)
        out = np.zeros((len(df), len(TRIFECTA_PERMUTATIONS)), dtype=float)
        alpha = 0.2

        for row_idx, (_, row) in enumerate(df.iterrows()):
            candidates = self._candidate_indices(row)
            X_cand = self.X_[candidates]
            diff = X_cand - X_eval[row_idx]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            if len(candidates) > self.k:
                local = np.argpartition(dist, self.k - 1)[: self.k]
            else:
                local = np.arange(len(candidates))
            nearest = candidates[local]
            nearest_dist = dist[local]

            if self.weighted:
                weights = 1.0 / np.maximum(nearest_dist, 1e-6)
                weights = np.minimum(weights, 1e6)
            else:
                weights = np.ones_like(nearest_dist, dtype=float)

            counts = np.bincount(
                self.labels_[nearest],
                weights=weights,
                minlength=len(TRIFECTA_PERMUTATIONS),
            ).astype(float)
            counts += alpha
            out[row_idx] = counts / counts.sum()
        return normalize_probability_matrix(out)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame()


class CentroidSoftmaxClassifier:
    """LightGBMが使えない環境でも検証を完走させるための軽量フォールバック。"""

    def __init__(self, n_classes: int) -> None:
        self.n_classes = int(n_classes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CentroidSoftmaxClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        global_mean = X.mean(axis=0)
        centroids = np.tile(global_mean, (self.n_classes, 1))
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        for cls in range(self.n_classes):
            mask = y == cls
            if mask.any():
                centroids[cls] = X[mask].mean(axis=0)
        priors = (counts + 1.0) / (counts.sum() + self.n_classes)
        self.centroids_ = centroids
        self.log_priors_ = np.log(priors)
        self.feature_importances_ = np.var(centroids, axis=0)
        self.temperature_ = max(float(X.shape[1]), 1.0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.centroids_[None, :, :]
        dist2 = np.einsum("ncf,ncf->nc", diff, diff)
        scores = -dist2 / self.temperature_ + self.log_priors_[None, :]
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        return normalize_probability_matrix(proba)


class OptionalMulticlassClassifier:
    def __init__(
        self,
        n_classes: int,
        *,
        random_state: int = 42,
        prefer_lightgbm: bool = True,
    ) -> None:
        self.n_classes = int(n_classes)
        self.random_state = random_state
        self.prefer_lightgbm = prefer_lightgbm
        self.backend_name = "lightgbm"
        self.model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptionalMulticlassClassifier":
        if not self.prefer_lightgbm:
            raise RuntimeError("Model C requires LightGBM. Remove --no-lightgbm and install requirements.txt.")

        try:
            import lightgbm as lgb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local optional deps
            raise RuntimeError(
                "LightGBM is required for Model C. Install it with: pip install -r requirements.txt"
            ) from exc

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        train_set = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "objective": "multiclass",
            "num_class": self.n_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.045,
            "num_leaves": 63,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": self.random_state,
            "feature_fraction_seed": self.random_state,
            "bagging_seed": self.random_state,
            "verbosity": -1,
            "num_threads": -1,
        }
        self.model = lgb.train(params, train_set, num_boost_round=280)
        self.backend_name = "lightgbm"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        raw = self.model.predict(np.asarray(X, dtype=np.float32))
        return normalize_probability_matrix(raw)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        values = np.asarray(self.model.feature_importance(importance_type="gain"), dtype=float)
        return pd.DataFrame({"feature": feature_names, "importance": values})


class ModelCBasicDirect:
    def __init__(self, prefer_lightgbm: bool = True) -> None:
        self.prefer_lightgbm = prefer_lightgbm
        self.leakage_risk = False

    def fit(self, df: pd.DataFrame) -> "ModelCBasicDirect":
        self.feature_columns_ = feature_list(df, MODEL_C_FEATURES)
        categorical = [col for col in BASIC_CATEGORICAL_FEATURES if col in self.feature_columns_]
        self.encoder_ = FeatureEncoder(self.feature_columns_, categorical_columns=categorical)
        X = self.encoder_.fit_transform(df)
        y = target_indices(df)
        self.classifier_ = OptionalMulticlassClassifier(
            len(TRIFECTA_PERMUTATIONS),
            prefer_lightgbm=self.prefer_lightgbm,
        ).fit(X, y)
        self.model_name = f"C_{self.classifier_.backend_name}_direct120"
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.encoder_.transform(df)
        return self.classifier_.predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        fi = self.classifier_.feature_importance(self.feature_columns_)
        if fi.empty:
            return fi
        fi["submodel"] = "direct120"
        return fi


class ModelCBasicPosition:
    def __init__(self, prefer_lightgbm: bool = True) -> None:
        self.prefer_lightgbm = prefer_lightgbm
        self.leakage_risk = False

    def fit(self, df: pd.DataFrame) -> "ModelCBasicPosition":
        self.feature_columns_ = feature_list(df, MODEL_C_FEATURES)
        categorical = [col for col in BASIC_CATEGORICAL_FEATURES if col in self.feature_columns_]
        self.encoder_ = FeatureEncoder(self.feature_columns_, categorical_columns=categorical)
        X = self.encoder_.fit_transform(df)

        self.classifiers_ = []
        backends = []
        for target_col in ["r1", "r2", "r3"]:
            y = pd.to_numeric(df[target_col], errors="coerce").astype(int).to_numpy() - 1
            clf = OptionalMulticlassClassifier(6, prefer_lightgbm=self.prefer_lightgbm).fit(X, y)
            self.classifiers_.append(clf)
            backends.append(clf.backend_name)
        backend = "lightgbm" if all(name == "lightgbm" for name in backends) else "centroid_fallback"
        self.model_name = f"C_{backend}_position6"
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.encoder_.transform(df)
        p1, p2, p3 = [clf.predict_proba(X) for clf in self.classifiers_]
        out = np.zeros((len(df), len(TRIFECTA_PERMUTATIONS)), dtype=float)
        for idx, (first, second, third) in enumerate(TRIFECTA_PERMUTATIONS):
            out[:, idx] = p1[:, first - 1] * p2[:, second - 1] * p3[:, third - 1]
        return normalize_probability_matrix(out)

    def feature_importance(self) -> pd.DataFrame:
        frames = []
        for label, clf in zip(["r1", "r2", "r3"], self.classifiers_):
            fi = clf.feature_importance(self.feature_columns_)
            if not fi.empty:
                fi["submodel"] = label
                frames.append(fi)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class NumpyBinaryLogisticRegression:
    def __init__(self, *, learning_rate: float = 0.08, n_iter: int = 140, l2: float = 1e-4) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "NumpyBinaryLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        sample_weight = sample_weight / np.maximum(sample_weight.mean(), 1e-12)

        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = 0.0
        n = float(len(y))

        for step in range(self.n_iter):
            z = X @ self.coef_ + self.intercept_
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
            error = (p - y) * sample_weight
            grad_w = (X.T @ error) / n + self.l2 * self.coef_
            grad_b = error.mean()
            lr = self.learning_rate / np.sqrt(1.0 + step / 25.0)
            self.coef_ -= lr * grad_w
            self.intercept_ -= lr * grad_b
        return self

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


PAIRWISE_BASE_FEATURES = [
    "left_boat_no",
    "right_boat_no",
    "left_winrate",
    "right_winrate",
    "winrate_diff",
    "left_winrate_rank",
    "right_winrate_rank",
    "left_course_win_rate",
    "right_course_win_rate",
    "course_win_rate_diff",
    "left_avg_st",
    "right_avg_st",
    "avg_st_diff",
    "left_st_rank",
    "right_st_rank",
    "st_rank_diff",
    "jcd",
    "r",
]
PAIRWISE_CATEGORICAL = ["jcd", "r"]


def _boat_finish_positions(row: pd.Series) -> dict[int, int]:
    return {
        int(row["r1"]): 1,
        int(row["r2"]): 2,
        int(row["r3"]): 3,
    }


def build_pairwise_frame(
    df: pd.DataFrame,
    *,
    include_racer_master: bool,
    include_target: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray, list[tuple[int, int]]]:
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    weights: list[float] = []
    race_indices: list[int] = []
    pair_keys: list[tuple[int, int]] = []

    for race_idx, (_, row) in enumerate(df.iterrows()):
        positions = _boat_finish_positions(row) if include_target else {}
        for left in BOATS:
            for right in range(left + 1, 7):
                left_win = float(row[f"勝率{left}"])
                right_win = float(row[f"勝率{right}"])
                left_rank = float(row[f"勝率順位{left}"])
                right_rank = float(row[f"勝率順位{right}"])

                if include_racer_master:
                    left_course = float(row.get(f"boat{left}_course_win_rate", 0.0))
                    right_course = float(row.get(f"boat{right}_course_win_rate", 0.0))
                    left_st = float(row.get(f"boat{left}_course_avg_st", 0.0))
                    right_st = float(row.get(f"boat{right}_course_avg_st", 0.0))
                    left_st_rank = float(row.get(f"boat{left}_course_st_rank", 0.0))
                    right_st_rank = float(row.get(f"boat{right}_course_st_rank", 0.0))
                else:
                    left_course = right_course = 0.0
                    left_st = right_st = 0.0
                    left_st_rank = right_st_rank = 0.0

                rows.append(
                    {
                        "left_boat_no": left,
                        "right_boat_no": right,
                        "left_winrate": left_win,
                        "right_winrate": right_win,
                        "winrate_diff": left_win - right_win,
                        "left_winrate_rank": left_rank,
                        "right_winrate_rank": right_rank,
                        "left_course_win_rate": left_course,
                        "right_course_win_rate": right_course,
                        "course_win_rate_diff": left_course - right_course,
                        "left_avg_st": left_st,
                        "right_avg_st": right_st,
                        "avg_st_diff": left_st - right_st,
                        "left_st_rank": left_st_rank,
                        "right_st_rank": right_st_rank,
                        "st_rank_diff": left_st_rank - right_st_rank,
                        "jcd": str(row["jcd"]).zfill(2),
                        "r": int(row["r"]),
                    }
                )
                race_indices.append(race_idx)
                pair_keys.append((left, right))

                if include_target:
                    left_pos = positions.get(left, 4)
                    right_pos = positions.get(right, 4)
                    labels.append(1 if left_pos < right_pos else 0)
                    # 4着以下同士は実順位が不明。15ペアは作りつつ、学習上の重みを弱める。
                    weights.append(0.25 if left_pos == right_pos == 4 else 1.0)

    X = pd.DataFrame(rows)
    y = np.asarray(labels, dtype=int) if include_target else None
    w = np.asarray(weights, dtype=float) if include_target else None
    return X, y, w, np.asarray(race_indices, dtype=int), pair_keys


class PairwiseRankModel:
    def __init__(self, *, include_racer_master: bool) -> None:
        self.include_racer_master = include_racer_master
        self.leakage_risk = include_racer_master
        suffix = "with_master_LEAKAGE_RISK" if include_racer_master else "no_master"
        self.model_name = f"D_pairwise_recent_{suffix}"

    def fit(self, df: pd.DataFrame) -> "PairwiseRankModel":
        X_df, y, sample_weight, _, _ = build_pairwise_frame(
            df,
            include_racer_master=self.include_racer_master,
            include_target=True,
        )
        self.feature_columns_ = PAIRWISE_BASE_FEATURES
        self.encoder_ = FeatureEncoder(
            self.feature_columns_,
            categorical_columns=PAIRWISE_CATEGORICAL,
            standardize=True,
        )
        X = self.encoder_.fit_transform(X_df)
        self.model_ = NumpyBinaryLogisticRegression().fit(X, y, sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_df, _, _, race_indices, pair_keys = build_pairwise_frame(
            df,
            include_racer_master=self.include_racer_master,
            include_target=False,
        )
        X = self.encoder_.transform(X_df)
        pair_p = self.model_.predict_proba_positive(X)

        boat_scores = np.zeros((len(df), 6), dtype=float)
        for idx, (race_idx, (left, right)) in enumerate(zip(race_indices, pair_keys)):
            p_left = pair_p[idx]
            boat_scores[race_idx, left - 1] += p_left
            boat_scores[race_idx, right - 1] += 1.0 - p_left
        return scores_to_trifecta_proba(boat_scores)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_columns_,
                "importance": np.abs(self.model_.coef_),
                "submodel": "pairwise_logistic",
            }
        )


def fit_predict(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, model_type: str) -> ModelPrediction:
    model.fit(train_df)
    proba = model.predict_proba(test_df)
    notes = ""
    if getattr(model, "leakage_risk", False):
        notes = "racer_master uses possibly June-2026 information; leakage riskあり"
    return ModelPrediction(
        model_name=model.model_name,
        proba=proba,
        leakage_risk=getattr(model, "leakage_risk", False),
        model_type=model_type,
        notes=notes,
    )


def model_feature_importance(model: Any) -> pd.DataFrame:
    if not hasattr(model, "feature_importance"):
        return pd.DataFrame()
    fi = model.feature_importance()
    if fi.empty:
        return fi
    fi.insert(0, "model", model.model_name)
    fi["leakage_risk"] = getattr(model, "leakage_risk", False)
    return fi
