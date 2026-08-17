# -*- coding: utf-8 -*-
"""日次予測: Codex製all_head_hierarchicalモデル(Codex_all_head_hierarchical_outer_corrected)
による3連単予測をoutputs/all_head_predictions_YYYYMMDD.csvへ書き出す。

サイト表示は「1着可能性が最も高い号艇(本命)のTop5」「2番目に高い号艇(対抗)のTop3」の
2グループ構成(favorite_top1〜5 / challenger_top1〜3)。

既存のrun_scenario_predictions.py(旧モデル)とは完全に独立した経路であり、
このスクリプトは旧モデルの出力・artifactには一切触れない。artifactは
artifacts/all_head_hierarchical/配下のものをそのまま読み込み、再学習・
上書きは行わない。読み込みに失敗した場合はfallbackせず例外で停止する。
"""
from __future__ import annotations

import argparse

import pandas as pd

from boat_model.all_head_hierarchical_infer import (
    ArtifactLoadError,
    PredictionValidationError,
    build_site_features,
    favorite_challenger_tickets,
    load_artifacts,
    predict_race_probabilities,
    top20_from_probabilities,
    validate_predictions,
)

ARTIFACT_DIR = "artifacts/all_head_hierarchical"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("出走表を読み込み中...")
    races_csv_df = pd.read_csv(args.input, dtype={"jcd": str})
    n_input_rows = len(races_csv_df)
    print(f"  入力レース数: {n_input_rows}")

    print("Codexモデルのartifactを読み込み中(再学習・fallbackなし)...")
    try:
        artifacts = load_artifacts(ARTIFACT_DIR)
    except ArtifactLoadError as exc:
        print(f"⚠️ artifact読み込みに失敗しました。処理を停止します: {exc}")
        raise

    print("特徴量を構築中(結果列は使用しない)...")
    races, features = build_site_features(races_csv_df)

    print("13ステップの確率計算パイプラインを実行中...")
    probabilities = predict_race_probabilities(races, features, artifacts)

    top20_tickets, top20_probs = top20_from_probabilities(probabilities)

    print("7項目の実行時検証を実施中(120通り全体の健全性チェック)...")
    try:
        validate_predictions(probabilities, top20_tickets, top20_probs, n_input_rows)
    except PredictionValidationError as exc:
        print(f"⚠️ 実行時検証に失敗しました。処理を停止します: {exc}")
        raise
    print("  検証すべて合格")

    print("表示用(本命Top5・対抗Top3)を抽出中...")
    try:
        grouped = favorite_challenger_tickets(probabilities)
    except PredictionValidationError as exc:
        print(f"⚠️ 表示用データの検証に失敗しました。処理を停止します: {exc}")
        raise

    rows = []
    for i in range(len(races)):
        row = {
            "date": races.iloc[i]["date"],
            "jcd": races.iloc[i]["jcd"],
            "r": races.iloc[i]["r"],
            "favorite_boat": int(grouped["favorite_boat"][i]),
            "challenger_boat": int(grouped["challenger_boat"][i]),
        }
        for k in range(5):
            row[f"favorite_top{k + 1}_ticket"] = grouped["favorite_tickets"][i, k]
            row[f"favorite_top{k + 1}_prob"] = round(float(grouped["favorite_probs"][i, k]), 6)
        for k in range(3):
            row[f"challenger_top{k + 1}_ticket"] = grouped["challenger_tickets"][i, k]
            row[f"challenger_top{k + 1}_prob"] = round(float(grouped["challenger_probs"][i, k]), 6)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"完了: {len(out_df)}レース分を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
