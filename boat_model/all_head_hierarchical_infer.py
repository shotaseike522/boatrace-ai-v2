# -*- coding: utf-8 -*-
"""Codex側all_head_hierarchicalモデル(Codex_all_head_hierarchical_outer_corrected)を
サイトの日次予測に組み込むための推論専用モジュール。

- artifacts/all_head_hierarchical/配下のartifactをそのまま読み込む(再学習しない)。
- サイトの出走表CSV(races_YYYYMMDD.csv、結果列なし)から、Codex側モデルが
  期待する特徴量フレームを組み立てる(build_site_features)。
- docs/claude_code_codex_model_site_integration_prompt.txtで指定された
  13ステップの確率計算パイプラインをpredict_race_probabilities()として実装する。
- 出力前に7項目の実行時検証を行い、違反時は例外を送出して停止する(fallbackしない)。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from boat_model.all_head_hierarchical import (
    TICKETS,
    compose_ticket_probabilities,
    predict_second_tensor,
    predict_third_tensor,
    temperature_scale,
)
from boat_model.boat1_win_stage import build_race_features
from boat_model.position_probability_calibration import apply_calibrator_along_last_axis

ARTIFACT_FILES = [
    "first6_model.joblib",
    "conditional_second_model.joblib",
    "conditional_third_model.joblib",
    "position_boat_isotonic_calibrators.joblib",
    "outer_head_ticket_residual_calibrator.joblib",
    "development_config.json",
    "position_calibration_config.json",
]

_LABELS = np.array(["-".join(map(str, t)) for t in TICKETS])
_LABEL_RANK = np.empty(120, dtype=np.int64)
_LABEL_RANK[np.argsort(_LABELS)] = np.arange(120)
_FIRST_BOAT_OF_TICKET = np.array([t[0] for t in TICKETS], dtype=np.int8)

BASE_FIELD_SITE_COLUMNS = {
    "全国勝率": "勝率{w}",
    "全国2率": "全国2率{w}",
    "当地勝率": "当地勝率{w}",
    "当地2率": "当地2率{w}",
    "モーター2率": "モーター2率{w}",
    "ボート2率": "ボート2率{w}",
    "年齢": "年齢{w}",
    "体重": "体重{w}",
}


class ArtifactLoadError(RuntimeError):
    """artifact読み込み失敗時に送出する。fallbackは行わず、常にここで停止する。"""


class PredictionValidationError(RuntimeError):
    """13ステップ後の確率が7項目の実行時検証のいずれかに違反した場合に送出する。"""


def load_artifacts(artifact_dir: str | Path) -> dict:
    artifact_dir = Path(artifact_dir)
    for name in ARTIFACT_FILES:
        if not (artifact_dir / name).exists():
            raise ArtifactLoadError(f"必須artifactが見つかりません: {artifact_dir / name}")
    try:
        config = json.loads((artifact_dir / "development_config.json").read_text(encoding="utf-8"))
        calibrators = joblib.load(artifact_dir / "position_boat_isotonic_calibrators.joblib")
        outer_calibrator = joblib.load(artifact_dir / "outer_head_ticket_residual_calibrator.joblib")
        first_model = joblib.load(artifact_dir / "first6_model.joblib")
        second_model = joblib.load(artifact_dir / "conditional_second_model.joblib")
        third_model = joblib.load(artifact_dir / "conditional_third_model.joblib")
    except Exception as exc:  # noqa: BLE001 - artifact読み込み失敗は必ずここで停止させる
        raise ArtifactLoadError(f"artifactの読み込みに失敗しました: {exc}") from exc
    if config.get("fallback_used"):
        raise ArtifactLoadError("development_config.jsonがfallback_used=trueを記録しています")
    return {
        "temperatures": config["temperatures"],
        "calibrators": calibrators,
        "outer_calibrator": outer_calibrator,
        "first_model": first_model,
        "second_model": second_model,
        "third_model": third_model,
    }


def build_site_features(races_csv_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """サイトのdata/races_YYYYMMDD.csv(結果列なし)から、build_race_features()が
    期待する形式のracesフレームと、そのfeaturesフレームを組み立てる。

    天候・風向・風速・波高は、サイトの日次スクレイピング(mbrace.or.jp公式B形式)に
    元々含まれていないため(K形式=レース結果にしか存在しない構造的制約)、欠損
    (NaN)のまま渡す。FeatureEncoderは学習時の中央値/"__MISSING__"カテゴリへ
    自動フォールバックする設計になっているため、モデルは動作するが、これらの
    特徴量に基づく精度改善効果は得られない。
    レース距離(race_距離)はB形式のレース見出し行(「Ｈ１８００ｍ」等)に確定値
    として記載されており、天候と異なり購入前に取得可能なため、daily_prep.pyが
    書き出す"距離"列をそのまま使う(列が無い古い出走表CSVとの後方互換のため
    存在しない場合はNaNへフォールバックする)。
    """
    df = races_csv_df.copy()
    df["date"] = df["date"].astype(str)
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("Int64")
    df["race_id"] = (
        df["date"] + "_" + df["jcd"] + "_" + df["r"].astype(str).str.zfill(2)
    )

    df["race_距離"] = pd.to_numeric(df["距離"], errors="coerce") if "距離" in df.columns else np.nan
    df["race_天候"] = np.nan
    df["race_風向"] = np.nan
    df["race_風速"] = np.nan
    df["race_波高"] = np.nan

    for w in range(1, 7):
        df[f"boat{w}_級別"] = df[f"級別{w}"] if f"級別{w}" in df.columns else None
        df[f"boat{w}_支部"] = df[f"支部{w}"] if f"支部{w}" in df.columns else None
        for codex_field, site_pattern in BASE_FIELD_SITE_COLUMNS.items():
            site_col = site_pattern.format(w=w)
            df[f"boat{w}_{codex_field}"] = df[site_col] if site_col in df.columns else None

    df["boat1_win"] = 0  # ダミー。build_race_features()の出力からは除外される(未使用)。
    features = build_race_features(df)
    return df, features


def predict_race_probabilities(races: pd.DataFrame, features: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """指示書の13ステップの確率計算パイプライン(ステップ13のランキングはCSV出力側で行う)。

    1. first6_modelで1着6分類確率を生成
    2. 1着確率へtemperature=0.9を適用
    3. 1着確率へposition_boat_isotonic_calibratorsのfirst補正を適用
    4. conditional_second_modelで条件付き2着確率を生成
    5. 2着確率へtemperature=0.9を適用
    6. 2着確率へposition_boat_isotonic_calibratorsのsecond補正を適用
    7. conditional_third_modelで条件付き3着確率を生成
    8. 3着確率へtemperature=0.9のみを適用(isotonic補正は使わない)
    9. P(first)×P(second|first)×P(third|first,second)で3連単120通りを生成
    10. 各レース内で120通りの合計を1へ正規化
    11. outer_head_ticket_residual_calibratorを4〜6号艇頭の買い目へ適用
    12. 各レース内で再度合計1へ正規化
    """
    temperatures = artifacts["temperatures"]

    first = artifacts["first_model"].predict_proba(features)
    first = first / first.sum(axis=1, keepdims=True)
    first = temperature_scale(first, float(temperatures["first"]), axis=1)
    first = artifacts["calibrators"]["first"].predict(first)

    second = predict_second_tensor(artifacts["second_model"], races, features)
    second = temperature_scale(second, float(temperatures["second"]), axis=2)
    second = apply_calibrator_along_last_axis(second, artifacts["calibrators"]["second"])

    third = predict_third_tensor(artifacts["third_model"], races, features)
    third = temperature_scale(third, float(temperatures["third"]), axis=3)  # isotonicは使わない(指示通り)

    composed = compose_ticket_probabilities(first, second, third)  # 9-10 (内部で正規化済み)

    corrected = artifacts["outer_calibrator"].predict(composed, TICKETS)  # 11 (内部で12の正規化も実施)
    corrected = corrected / corrected.sum(axis=1, keepdims=True)  # 12 念のための再正規化
    return corrected


def top20_from_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各レースについて、確率降順(同率時はticket文字列昇順で固定タイブレーク)で
    Top20のticket文字列と確率を返す。"""
    neg_prob = -probabilities
    rank_key = np.broadcast_to(_LABEL_RANK, probabilities.shape)
    order = np.lexsort((rank_key, neg_prob), axis=1)  # 主キー: neg_prob昇順(=確率降順)、副キー: ticket文字列昇順
    top20_idx = order[:, :20]
    top20_tickets = _LABELS[top20_idx]
    top20_probs = np.take_along_axis(probabilities, top20_idx, axis=1)
    return top20_tickets, top20_probs


def favorite_challenger_tickets(probabilities: np.ndarray) -> dict:
    """サイト表示用: 1着確率(最終補正後の周辺確率)が最も高い号艇(本命)のTop5と、
    2番目に高い号艇(対抗)のTop3の3連単を、確率降順(同率時はticket文字列昇順)で返す。

    1着確率の周辺分布は、120通りの最終補正後確率のうち「そのticketの1着艇」が
    一致する列を合計して求める(=raw/中間確率ではなく最終確率ベース)。
    """
    n = probabilities.shape[0]
    boat_marginal = np.zeros((n, 6), dtype=np.float64)
    for boat in range(1, 7):
        boat_marginal[:, boat - 1] = probabilities[:, _FIRST_BOAT_OF_TICKET == boat].sum(axis=1)
    boat_rank = np.argsort(-boat_marginal, axis=1, kind="stable")
    favorite_boat = (boat_rank[:, 0] + 1).astype(np.int8)
    challenger_boat = (boat_rank[:, 1] + 1).astype(np.int8)

    favorite_tickets = np.empty((n, 5), dtype=object)
    favorite_probs = np.zeros((n, 5), dtype=np.float64)
    challenger_tickets = np.empty((n, 3), dtype=object)
    challenger_probs = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        fav_idx = np.flatnonzero(_FIRST_BOAT_OF_TICKET == favorite_boat[i])
        chal_idx = np.flatnonzero(_FIRST_BOAT_OF_TICKET == challenger_boat[i])
        fav_order = fav_idx[np.lexsort((_LABEL_RANK[fav_idx], -probabilities[i, fav_idx]))]
        chal_order = chal_idx[np.lexsort((_LABEL_RANK[chal_idx], -probabilities[i, chal_idx]))]
        top5 = fav_order[:5]
        top3 = chal_order[:3]
        favorite_tickets[i] = _LABELS[top5]
        favorite_probs[i] = probabilities[i, top5]
        challenger_tickets[i] = _LABELS[top3]
        challenger_probs[i] = probabilities[i, top3]

    if not np.all(np.diff(favorite_probs, axis=1) <= 1e-15):
        raise PredictionValidationError("本命Top5が確率降順になっていないレースがあります")
    if not np.all(np.diff(challenger_probs, axis=1) <= 1e-15):
        raise PredictionValidationError("対抗Top3が確率降順になっていないレースがあります")
    if np.any(favorite_boat == challenger_boat):
        raise PredictionValidationError("本命艇と対抗艇が同じレースがあります")

    return {
        "favorite_boat": favorite_boat,
        "favorite_tickets": favorite_tickets,
        "favorite_probs": favorite_probs,
        "challenger_boat": challenger_boat,
        "challenger_tickets": challenger_tickets,
        "challenger_probs": challenger_probs,
    }


def validate_predictions(
    probabilities: np.ndarray,
    top20_tickets: np.ndarray,
    top20_probs: np.ndarray,
    n_input_rows: int,
) -> None:
    """指示書が要求する7項目の実行時検証。違反時はPredictionValidationErrorを送出する。"""
    n = probabilities.shape[0]

    # 1. 3連単120通りがすべて存在する
    if probabilities.shape[1] != 120:
        raise PredictionValidationError(f"3連単の列数が120ではありません: {probabilities.shape[1]}")
    if len(TICKETS) != 120 or len(set(TICKETS)) != 120:
        raise PredictionValidationError("TICKETSの定義に120通り・重複なしの前提が崩れています")

    # 2. ticket重複がない(TICKETS自体の構造チェック、念のため実行時にも確認)
    if len(set(_LABELS.tolist())) != 120:
        raise PredictionValidationError("ticketラベルに重複があります")

    # 3. 確率がすべて有限で0以上
    if not np.isfinite(probabilities).all():
        raise PredictionValidationError("確率に非有限値(NaN/inf)が含まれています")
    if (probabilities < 0).any():
        raise PredictionValidationError("負の確率が含まれています")

    # 4. 120確率の合計が1(許容誤差1e-10)
    row_sums = probabilities.sum(axis=1)
    max_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    if max_sum_error > 1e-10:
        raise PredictionValidationError(f"120確率の合計が1から外れています(最大誤差={max_sum_error:.3e})")

    # 5. Top1〜Top20が確率降順
    if not np.all(np.diff(top20_probs, axis=1) <= 1e-15):
        raise PredictionValidationError("Top1〜Top20が確率降順になっていないレースがあります")

    # 6. Top20のticket重複がない
    for i in range(n):
        if len(set(top20_tickets[i].tolist())) != 20:
            raise PredictionValidationError(f"レース{i}のTop20にticket重複があります")

    # 7. 入力行数と出力行数が一致する
    if n_input_rows != n:
        raise PredictionValidationError(f"入力行数({n_input_rows})と出力行数({n})が一致しません")
