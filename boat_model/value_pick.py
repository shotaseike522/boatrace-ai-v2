"""boat_model/value_pick.py

11R・12R専用の「回収率寄り妙味候補」を計算するモジュール。

ChatGPT/Codexで検証済みの contribution_adjusted_input / top1 / gamma=0.5 を
サイト予測（run_site_predictions.py）に組み込む。

設計思想:
- 通常AI予想（direct120 + position6 + KNN + Pairwise のブレンド）とは別枠
- 全国勝率をコース別成績・進入率で小さく補正（gamma=0.5）してBasePositionModelで予測
- 11R・12Rのみ value_pick_ticket 列に出力、それ以外は空欄
- 通常予想に混ぜず、表示上も「11・12R 妙味候補」として扱う
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from boat_model.features import (
    BOATS,
    BASIC_CATEGORICAL_FEATURES,
    BASIC_NUMERIC_FEATURES,
    INDEX_TO_TRIFECTA,
    REGISTRATION_COLS,
    TRIFECTA_PERMUTATIONS,
    WINRATE_COLS,
    FeatureEncoder,
    add_basic_features,
    feature_list,
    normalize_probability_matrix,
    normalize_registration_series,
)
from boat_model.models import CentroidSoftmaxClassifier

GAMMA_FIXED: float = 0.5
VALUE_PICK_MODEL: str = "contribution_adjusted_input"
VALUE_PICK_REASON: str = "11R・12R専用の回収率寄り候補。全国勝率をコース別寄与で補正。"
N_TICKETS: int = len(TRIFECTA_PERMUTATIONS)


# ============================================================
# racer_master → コーステーブル変換
# ============================================================

def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _master_lookup(master: pd.DataFrame) -> pd.DataFrame:
    """登録番号をインデックスにした参照テーブルを作る。"""
    work = master.copy()
    work["登録番号"] = normalize_registration_series(work["登録番号"])
    return (
        work.dropna(subset=["登録番号"])
        .drop_duplicates("登録番号", keep="last")
        .set_index("登録番号")
    )


def prepare_master_course_table(master: pd.DataFrame) -> pd.DataFrame:
    """racer_master を受け取り、各選手のコーススコアと進入率加重平均スコアを計算する。

    コーススコア（1〜6コースそれぞれ）:
        course_score_c = 1着率 + 0.60×2着率 + 0.35×3着率

    進入率加重平均スコア:
        player_avg_course_score = Σ(正規化済み進入率_c × course_score_c)
        ※ 進入率は合計が1.0になるよう正規化してから使う（%表記・絶対値の影響を除去）
    """
    work = _master_lookup(master)
    for course in BOATS:
        first  = _safe_numeric(work.get(f"{course}コース_1着率", pd.Series(index=work.index, dtype=float)))
        second = _safe_numeric(work.get(f"{course}コース_2着率", pd.Series(index=work.index, dtype=float)))
        third  = _safe_numeric(work.get(f"{course}コース_3着率", pd.Series(index=work.index, dtype=float)))
        work[f"course_score_{course}"] = first + 0.60 * second + 0.35 * third

    score_cols = [f"course_score_{c}" for c in BOATS]
    entry_cols = [f"{c}コース_進入率" for c in BOATS]
    scores = work[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if all(col in work.columns for col in entry_cols):
        entries = work[entry_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # 進入率を合計1.0に正規化（%表記でも絶対値でも影響を受けないようにする）
        entry_sum = entries.sum(axis=1, keepdims=True)
        fallback = np.full_like(entries, 1.0 / len(BOATS))
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized_entries = np.where(
                entry_sum > 1e-12,
                entries / np.where(entry_sum > 1e-12, entry_sum, 1.0),
                fallback,
            )
        weighted = (normalized_entries * scores).sum(axis=1)
        work["player_avg_course_score"] = weighted
    else:
        work["player_avg_course_score"] = scores.mean(axis=1)

    return work


def _map_master_value(
    registrations: pd.Series,
    master_table: pd.DataFrame,
    source_col: str,
    *,
    fill_value: float | None = None,
) -> pd.Series:
    if source_col not in master_table.columns:
        return pd.Series(0.0 if fill_value is None else fill_value, index=registrations.index)
    values = registrations.map(master_table[source_col])
    if fill_value is None:
        fill_value = float(pd.to_numeric(master_table[source_col], errors="coerce").mean())
    return pd.to_numeric(values, errors="coerce").fillna(fill_value).fillna(0.0)


def contribution_adjust_winrates(
    df: pd.DataFrame,
    contribution_table: pd.DataFrame,
    *,
    gamma: float,
) -> pd.DataFrame:
    """全国勝率をコース別寄与で補正した出走表DataFrameを返す。

    adjusted_winrate = 全国勝率 × (target_course_score / avg_course_score) ^ gamma

    欠損選手（racer_masterに未登録）は lift=1.0（補正なし）として扱う。
    """
    out = df.copy()
    for boat in BOATS:
        reg_col = REGISTRATION_COLS[boat - 1]
        registrations = (
            normalize_registration_series(out[reg_col])
            if reg_col in out.columns
            else pd.Series(pd.NA, index=out.index)
        )
        target_score = _map_master_value(registrations, contribution_table, f"course_score_{boat}")
        avg_score    = _map_master_value(registrations, contribution_table, "player_avg_course_score")

        target_values = target_score.to_numpy(dtype=float)
        avg_values    = avg_score.to_numpy(dtype=float)

        lift = np.ones_like(target_values, dtype=float)
        np.divide(target_values, avg_values, out=lift, where=avg_values > 1e-12)
        lift = np.where(np.isfinite(lift) & (lift > 1e-6), lift, 1.0)

        base = pd.to_numeric(out[WINRATE_COLS[boat - 1]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[WINRATE_COLS[boat - 1]] = base * np.power(lift, float(gamma))

    return add_basic_features(out)


# ============================================================
# BasePositionModel：1着/2着/3着を6クラス分類して3連単に変換
# ============================================================

class BasePositionModel:
    """全国勝率（補正後も可）から3連単120通りの確率を出す軽量モデル。

    ModelCBasicPositionと同じ仕組みだが、CentroidSoftmaxClassifier のみ使用する
    （LightGBMなし）ため軽量で、検証コードと同じ実装を維持する。
    """

    def fit(self, df: pd.DataFrame) -> "BasePositionModel":
        self.feature_columns_ = feature_list(
            df,
            list(BASIC_NUMERIC_FEATURES) + list(BASIC_CATEGORICAL_FEATURES),
        )
        categorical = [c for c in BASIC_CATEGORICAL_FEATURES if c in self.feature_columns_]
        self.encoder_ = FeatureEncoder(self.feature_columns_, categorical_columns=categorical)
        x = self.encoder_.fit_transform(df).astype(np.float32)
        self.classifiers_: list[CentroidSoftmaxClassifier] = []
        for target_col in ["r1", "r2", "r3"]:
            y = pd.to_numeric(df[target_col], errors="coerce").astype(int).to_numpy() - 1
            self.classifiers_.append(CentroidSoftmaxClassifier(6).fit(x, y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self.encoder_.transform(df).astype(np.float32)
        p1, p2, p3 = [clf.predict_proba(x) for clf in self.classifiers_]
        out = np.zeros((len(df), N_TICKETS), dtype=np.float32)
        for idx, (first, second, third) in enumerate(TRIFECTA_PERMUTATIONS):
            out[:, idx] = p1[:, first - 1] * p2[:, second - 1] * p3[:, third - 1]
        return normalize_probability_matrix(out)


# ============================================================
# artifacts への保存・読み込み
# ============================================================

def save_base_position_model(model: BasePositionModel, models_dir: Path) -> Path:
    """学習済み BasePositionModel を artifacts/models/base_position_model.joblib に保存する。"""
    import joblib
    path = models_dir / "base_position_model.joblib"
    joblib.dump({"model_object": model, "model_type": "BasePositionModel"}, path)
    return path


def load_base_position_model(models_dir: Path) -> BasePositionModel | None:
    """保存済み BasePositionModel を読み込む。ファイルがなければ None を返す。"""
    import joblib
    path = models_dir / "base_position_model.joblib"
    if not path.exists():
        return None
    artifact = joblib.load(path)
    return artifact["model_object"]


# ============================================================
# サイト予測への組み込み：value_pick 列の計算
# ============================================================

def compute_value_pick(
    races: pd.DataFrame,
    base_model: BasePositionModel,
    master: pd.DataFrame,
    *,
    gamma: float = GAMMA_FIXED,
    target_races: tuple[int, ...] = (11, 12),
) -> pd.DataFrame:
    """11R・12Rに対して value_pick_* 列を計算して返す。

    対象外レースは全列が空文字列または NaN になる。

    Returns
    -------
    pd.DataFrame
        入力と同じ行数を持つ DataFrame。列は以下の通り。
        - value_pick_ticket   : 妙味候補の3連単チケット（例: "1-2-3"）
        - value_pick_prob     : そのチケットの予測確率（0〜1）
        - value_pick_model    : モデル名（固定文字列）
        - value_pick_reason   : 理由（固定文字列）
        - value_pick_gamma    : 使用したgamma値
        - value_pick_target   : 対象レース群名（固定文字列）
    """
    n = len(races)
    result = pd.DataFrame(
        {
            "value_pick_ticket": [""] * n,
            "value_pick_prob":   [float("nan")] * n,
            "value_pick_model":  [""] * n,
            "value_pick_reason": [""] * n,
            "value_pick_gamma":  [float("nan")] * n,
            "value_pick_target": [""] * n,
        }
    )

    # 11R・12Rのみ処理
    r_values = pd.to_numeric(races["r"], errors="coerce")
    target_mask = r_values.isin(target_races).to_numpy()
    if not target_mask.any():
        return result

    # racer_master → contribution_table
    master_table = prepare_master_course_table(master)

    # 補正後勝率でadd_basic_featuresを通し直す
    adjusted = contribution_adjust_winrates(races, master_table, gamma=gamma)

    # BasePositionModelでpredict_proba
    proba = base_model.predict_proba(adjusted)  # (n_races, 120)

    # 11R・12Rのみ結果を書き込む
    for i in np.where(target_mask)[0]:
        top1_idx = int(np.argmax(proba[i]))
        ticket = INDEX_TO_TRIFECTA[top1_idx]
        prob   = float(proba[i, top1_idx])
        result.at[i, "value_pick_ticket"]  = ticket
        result.at[i, "value_pick_prob"]    = prob
        result.at[i, "value_pick_model"]   = VALUE_PICK_MODEL
        result.at[i, "value_pick_reason"]  = VALUE_PICK_REASON
        result.at[i, "value_pick_gamma"]   = gamma
        result.at[i, "value_pick_target"]  = "r11_12"

    return result
