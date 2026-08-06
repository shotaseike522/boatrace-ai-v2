"""既存のniren-wide-experimentアーカイブ(2005-2026、ワイド形式)から、直近3年分
(2023/7〜2026/6)を site_archive_rolling3y.parquet と同じロング形式(1艇1行)に
変換し、初期シードとして書き出す。

これが無いと、サイト自前のローリングアーカイブは今日(スクレイピング開始日)から
ゼロで積み上がることになり、1年複勝率等のローリング特徴量が計算できる
ようになるまで1年待つ必要が生じてしまう。過去3年分を最初から入れておくことで、
初日から強さモデルの再学習・1年複勝率計算が可能になる。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_ARCHIVE = Path(r"C:\Users\trium\OneDrive\Desktop\boat\lzh_archive\boatrace_bk_dataset_2005-2026_with_1y.csv")
OUT_PATH = Path(__file__).resolve().parent / "data" / "site_archive_rolling3y.parquet"
SEED_START = "2023-07-01"

# rolling_archive.ALL_COLSと対応させる
ENTRY_FIELDS = {
    "登番": "登番", "級別": "級別", "年齢": "年齢", "体重": "体重",
    "全国勝率": "全国勝率", "全国2率": "全国2率", "当地勝率": "当地勝率", "当地2率": "当地2率",
    "モーター2率": "モーター2率", "ボート2率": "ボート2率",
}
RESULT_FIELDS = {"着順": "着順", "決まり手": None, "ST": "ST"}  # 決まり手は別ソース(kimarite seed)にあるため空でよい


def main() -> None:
    usecols = ["date", "jcd", "r"]
    for n in range(1, 7):
        for f in ["登番", "級別", "年齢", "体重", "全国勝率", "全国2率", "当地勝率", "当地2率",
                  "モーター2率", "ボート2率", "着順", "ST"]:
            usecols.append(f"boat{n}_{f}")

    print("既存アーカイブ読み込み中(列を絞って読み込み)...")
    df = pd.read_csv(SOURCE_ARCHIVE, dtype=str, usecols=usecols)
    df = df[df["date"] >= SEED_START].copy()
    print(f"対象期間({SEED_START}〜): {len(df)}レース")

    frames = []
    for n in range(1, 7):
        sub = pd.DataFrame({
            "date": df["date"], "jcd": df["jcd"], "r": df["r"], "boat_num": str(n),
        })
        sub["登番"] = df[f"boat{n}_登番"]
        sub["級別"] = df[f"boat{n}_級別"]
        sub["年齢"] = pd.to_numeric(df[f"boat{n}_年齢"], errors="coerce")
        sub["体重"] = pd.to_numeric(df[f"boat{n}_体重"], errors="coerce")
        sub["全国勝率"] = pd.to_numeric(df[f"boat{n}_全国勝率"], errors="coerce")
        sub["全国2率"] = pd.to_numeric(df[f"boat{n}_全国2率"], errors="coerce")
        sub["当地勝率"] = pd.to_numeric(df[f"boat{n}_当地勝率"], errors="coerce")
        sub["当地2率"] = pd.to_numeric(df[f"boat{n}_当地2率"], errors="coerce")
        sub["モーター番号"] = None
        sub["モーター2率"] = pd.to_numeric(df[f"boat{n}_モーター2率"], errors="coerce")
        sub["ボート番号"] = None
        sub["ボート2率"] = pd.to_numeric(df[f"boat{n}_ボート2率"], errors="coerce")
        sub["着順"] = df[f"boat{n}_着順"]
        sub["決まり手"] = None  # kimarite_payout_seedの方に別途ある(date+jcd+rで結合可能)
        sub["天候"] = None
        sub["風速"] = None
        sub["波高"] = None
        sub["水温"] = None
        sub["気温"] = None
        sub["ST"] = df[f"boat{n}_ST"]
        sub["単勝_払戻"] = None
        sub["複勝1_払戻"] = None
        sub["3連単_払戻"] = None
        sub["3連複_払戻"] = None
        sub["2連単_払戻"] = None
        sub["2連複_払戻"] = None
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)

    # 既にスクレイピングで蓄積済みの分(直近日付)があればそれを優先してマージする
    # (シードは2026-06-30までなので日付が重複することは通常ないが、念のため
    # date+jcd+r+boat_numで重複排除し、後勝ち=既存データ優先にする)
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        for c in existing.columns:
            existing[c] = existing[c].astype(str)
        combined = pd.concat([long_df, existing], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "jcd", "r", "boat_num"], keep="last")
    else:
        combined = long_df

    combined = combined.sort_values(["date", "jcd", "r", "boat_num"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"完了: {len(combined)}行 x {len(combined.columns)}列 を書き出しました: {OUT_PATH}")


if __name__ == "__main__":
    main()
