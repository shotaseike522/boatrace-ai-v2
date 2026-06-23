from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pytz

LOCAL_DEPS = Path(__file__).resolve().parent / "work" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boat_model.artifacts import (  # noqa: E402
    assign_roughness_bins,
    build_ai_top10_table,
    compute_quinella_top3,
    compute_roughness_score,
    knn_predict_proba_and_neighbors,
    load_joblib_artifact,
    load_knn_artifacts,
    position_artifacts_predict_proba,
    similar_race_diagnostics,
)
from boat_model.features import REGISTRATION_COLS, WINRATE_COLS, add_basic_features, normalize_probability_matrix  # noqa: E402
from boat_model.data_loader import load_racer_master  # noqa: E402
from boat_model.value_pick import (  # noqa: E402
    compute_value_pick,
    load_base_position_model,
    GAMMA_FIXED,
)

# 仕様書(model_inventory.md)記載の重み。
#   AI_score = 0.50*direct120 + 0.35*position6 + 0.15*blend
#   blend    = 0.5*knn500 + 0.5*pairwise
WEIGHT_DIRECT120 = 0.50
WEIGHT_POSITION6 = 0.35
WEIGHT_BLEND = 0.15
WEIGHT_KNN_IN_BLEND = 0.5
WEIGHT_PAIRWISE_IN_BLEND = 0.5

# 荒れやすさスコア(0-100点)の表示ラベル区分。仕様書の境界値に合わせている。
ROUGHNESS_LABEL_BINS = [0, 20, 40, 60, 80, 100]
ROUGHNESS_LABELS_JP = ["超堅め", "堅め", "普通", "荒れ注意", "波乱含み"]

REQUIRED_RESULT_COLS = ["r1", "r2", "r3", "3rt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create site prediction CSV from trained artifacts.")
    parser.add_argument("--input", required=True, help="July race entry CSV, e.g. data/races_202607.csv")
    parser.add_argument("--output", required=True, help="Output CSV, e.g. outputs/site_predictions_202607.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def read_site_races(path: str | Path) -> pd.DataFrame:
    """サイト予測用の出走表CSVを読み込む。

    7月出走表は未来予測用のため、結果列(r1, r2, r3, 3rt)が無いことを前提に
    必須列チェックは jcd, r, 勝率1-6 のみとする。結果列は近似100レースの
    集計(過去データ側)でのみ使い、ここでは要求しない。
    """
    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(path)

    required = ["jcd", "r"] + WINRATE_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"入力レースCSVに必須列がありません: {missing}\n"
            f"必要な列: {required}"
        )

    out = df.copy()
    out["jcd"] = out["jcd"].astype("string").str.strip().str.zfill(2)
    out["r"] = pd.to_numeric(out["r"], errors="coerce").astype("Int64")
    for col in WINRATE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in REGISTRATION_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    missing_result_cols = [c for c in REQUIRED_RESULT_COLS if c not in out.columns]
    if missing_result_cols:
        print(
            f"[INFO] 結果列が見つかりません {missing_result_cols} -> "
            "未来レース(結果未確定)として予測のみ実行します。"
        )
    return out.reset_index(drop=True)


def load_roughness_thresholds(artifacts_dir: str | Path) -> list[float]:
    path = Path(artifacts_dir) / "roughness_thresholds.json"
    if not path.exists():
        print(f"[WARN] {path} が見つかりません。レース内相対分位での荒れやすさ区分にフォールバックします。")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(x) for x in data.get("thresholds", [])]


def roughness_score_to_label(score_0_100: pd.Series) -> pd.Series:
    """0-100点の荒れやすさスコアを仕様書の5段階ラベルに変換する。"""
    return pd.cut(
        score_0_100,
        bins=ROUGHNESS_LABEL_BINS,
        labels=ROUGHNESS_LABELS_JP,
        include_lowest=True,
    ).astype(str)


def require_artifact(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"artifactが見つかりません: {path}\n"
            "先に `python run_train_models.py` を実行してartifactsを作成してください。"
        )
    return path


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    models_dir = artifacts_dir / "models"
    knn_dir = artifacts_dir / "knn"

    race_raw = read_site_races(args.input)
    races = add_basic_features(race_raw)

    direct_artifact = load_joblib_artifact(require_artifact(models_dir / "lightgbm_direct120.joblib"))
    position_artifacts = [
        load_joblib_artifact(require_artifact(models_dir / "lightgbm_position1.joblib")),
        load_joblib_artifact(require_artifact(models_dir / "lightgbm_position2.joblib")),
        load_joblib_artifact(require_artifact(models_dir / "lightgbm_position3.joblib")),
    ]
    pairwise_artifact = load_joblib_artifact(require_artifact(models_dir / "pairwise_model.joblib"))
    if not knn_dir.exists():
        raise FileNotFoundError(
            f"KNN artifactsディレクトリが見つかりません: {knn_dir}\n"
            "先に `python run_train_models.py` を実行してartifactsを作成してください。"
        )
    knn_artifact = load_knn_artifacts(knn_dir)

    direct_proba = direct_artifact["model_object"].predict_proba(races)
    position_proba = position_artifacts_predict_proba(races, position_artifacts)
    knn_proba, neighbors100 = knn_predict_proba_and_neighbors(races, knn_artifact, k=500, diagnostic_k=100)
    pairwise_proba = pairwise_artifact["model_object"].predict_proba(races)

    # P_blend = 0.5*KNN500 + 0.5*Pairwise (仕様書通り)
    blend_proba = normalize_probability_matrix(
        WEIGHT_KNN_IN_BLEND * knn_proba + WEIGHT_PAIRWISE_IN_BLEND * pairwise_proba
    )

    # AI_score = 0.50*direct120 + 0.35*position6 + 0.15*blend (仕様書通り)
    ai_proba = normalize_probability_matrix(
        WEIGHT_DIRECT120 * direct_proba
        + WEIGHT_POSITION6 * position_proba
        + WEIGHT_BLEND * blend_proba
    )

    # --- 荒れやすさ (0-100点 + 5段階ラベル) ---
    raw_roughness = compute_roughness_score(races)  # 0.0-1.0のpercentile rank相当
    thresholds = load_roughness_thresholds(artifacts_dir)
    if thresholds:
        roughness_bin = assign_roughness_bins(raw_roughness, thresholds)
    else:
        roughness_bin = pd.qcut(
            raw_roughness.rank(method="first"),
            q=5,
            labels=["Q1_low_roughness", "Q2", "Q3", "Q4", "Q5_high_roughness"],
        )
    roughness_score_100 = (raw_roughness.rank(pct=True) * 100).round(1)
    roughness_label = roughness_score_to_label(roughness_score_100)

    jst = pytz.timezone("Asia/Tokyo")
    today_str = datetime.now(jst).strftime("%Y%m%d")

    output = pd.DataFrame(
        {
            "date": today_str,
            "jcd": races["jcd"].astype(str),
            "r": races["r"].astype(int),
            "rough_raw": raw_roughness,
            "roughness_score": roughness_score_100,
            "roughness_label": roughness_label,
            "roughness_bin": roughness_bin.astype(str),
        }
    )

    # --- AI予想Top10 (チケット・確率・印・各モデル順位・一致数) ---
    ai_top10 = build_ai_top10_table(
        ai_proba, direct_proba, position_proba, blend_proba, top_n=10, agreement_top_n=5
    )
    output = pd.concat([output, ai_top10], axis=1)

    # --- 2連複 ベスト3 ---
    quinella = compute_quinella_top3(ai_proba, top_n=3)
    output = pd.concat([output, quinella], axis=1)

    # --- 近似100レース (過去事実: 配当分布 + 着順発生率Top10) ---
    diagnostics = similar_race_diagnostics(knn_artifact, neighbors100)
    output = pd.concat([output, diagnostics], axis=1)

    # --- 11・12R 妙味候補 (value_pick) ---
    # BasePositionModelがartifactsにあれば計算、なければ全行空欄で通常予想には影響しない。
    base_position_model = load_base_position_model(models_dir)
    master_path = Path("racer_master.csv")
    if base_position_model is not None and master_path.exists():
        try:
            master = load_racer_master(str(master_path))
            value_pick = compute_value_pick(
                races, base_position_model, master, gamma=GAMMA_FIXED
            )
            output = pd.concat([output, value_pick], axis=1)
            n_picks = (output["value_pick_ticket"] != "").sum()
            print(f"  value_pick: {n_picks} races (11R・12Rのみ)")
        except Exception as e:
            print(f"[WARN] value_pick の計算中にエラーが発生しました（通常予想には影響しません）: {e}")
            for col in ["value_pick_ticket", "value_pick_prob", "value_pick_model",
                        "value_pick_reason", "value_pick_gamma", "value_pick_target"]:
                output[col] = ""
    else:
        if base_position_model is None:
            print("[INFO] base_position_model.joblib が見つかりません。value_pick をスキップします。")
            print("       run_train_models.py を実行して再学習してください。")
        if not master_path.exists():
            print("[INFO] racer_master.csv が見つかりません。value_pick をスキップします。")
        for col in ["value_pick_ticket", "value_pick_prob", "value_pick_model",
                    "value_pick_reason", "value_pick_gamma", "value_pick_target"]:
            output[col] = ""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path} ({len(output)} races)")


if __name__ == "__main__":
    main()
