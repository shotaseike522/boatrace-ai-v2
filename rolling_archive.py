"""シナリオ予想モデル用のローリング3年アーカイブ(data/site_archive_rolling3y.csv)を
管理する。1行=1艇×1レースの長形式で、以下の2段階で更新する:

  1. 当日の出走表取得後(entryフェーズ): 登番・級別・全国勝率・全国2率・当地勝率・
     当地2率・モーター番号/2率・ボート番号/2率・年齢・体重を新規行として追記する。
     この時点では着順・決まり手・天候・払戻はまだ分からないため空欄。
  2. 翌日の結果取得後(resultフェーズ): 前日にentryフェーズで追記した行を
     date+jcd+r+boat_numで突き合わせ、着順・決まり手・天候・ST・払戻を追記(UPDATE)する。

日々のトリムで直近3年より古い行を削除し、アーカイブのサイズを一定に保つ。
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import pytz

ARCHIVE_PATH = "data/site_archive_rolling3y.csv"
TRIM_YEARS = 3


def _to_iso_date(date_val) -> str:
    """daily_prep.pyはdateを"YYYYMMDD"形式(mbrace.or.jpのファイル名慣習)で渡すが、
    アーカイブのシードデータ(build_archive_seed.py由来)は"YYYY-MM-DD"形式のため、
    混在するとpd.to_datetime()が本日分の行だけNaTにしてしまう(trim_archive()の
    文字列比較も壊れる)。書き込み時に必ずISO形式へ正規化して統一する。
    """
    s = str(date_val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s

ENTRY_COLS = [
    "登番", "級別", "年齢", "体重",
    "全国勝率", "全国2率", "当地勝率", "当地2率",
    "モーター番号", "モーター2率", "ボート番号", "ボート2率",
]
RESULT_COLS = [
    "着順", "決まり手", "天候", "風速", "波高", "水温", "気温", "ST",
    "単勝_払戻", "複勝1_払戻", "3連単_払戻", "3連複_払戻", "2連単_払戻", "2連複_払戻",
]
ALL_COLS = ["date", "jcd", "r", "boat_num"] + ENTRY_COLS + RESULT_COLS


def _load_archive() -> pd.DataFrame:
    if os.path.exists(ARCHIVE_PATH):
        df = pd.read_csv(ARCHIVE_PATH, dtype={"date": str, "jcd": str, "r": str, "boat_num": str, "登番": str})
        return df
    return pd.DataFrame(columns=ALL_COLS)


def _save_archive(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
    df.to_csv(ARCHIVE_PATH, index=False, encoding="utf-8-sig")


def append_entries_to_archive(races_csv_path: str | None) -> None:
    """当日の出走表(拡張済みraces_YYYYMMDD.csv)をローリングアーカイブに新規行として追記する。
    同じdate+jcd+rが既に存在する場合は二重追記しない(daily_prep.pyの再実行対策)。
    """
    if races_csv_path is None or not os.path.exists(races_csv_path):
        print("⚠️ 出走表が無いため、アーカイブへの追記をスキップします。")
        return

    races = pd.read_csv(races_csv_path, dtype={"jcd": str})
    archive = _load_archive()

    existing_keys = set(zip(archive["date"].astype(str), archive["jcd"].astype(str), archive["r"].astype(str))) \
        if not archive.empty else set()

    new_rows = []
    for _, row in races.iterrows():
        date_s, jcd_s, r_s = _to_iso_date(row["date"]), str(row["jcd"]), str(row["r"])
        if (date_s, jcd_s, r_s) in existing_keys:
            continue
        for w in range(1, 7):
            new_row = {"date": date_s, "jcd": jcd_s, "r": r_s, "boat_num": str(w)}
            new_row["登番"] = row.get(f"登番{w}", "")
            new_row["級別"] = row.get(f"級別{w}", "")
            new_row["年齢"] = row.get(f"年齢{w}", None)
            new_row["体重"] = row.get(f"体重{w}", None)
            new_row["全国勝率"] = row.get(f"勝率{w}", None)
            new_row["全国2率"] = row.get(f"全国2率{w}", None)
            new_row["当地勝率"] = row.get(f"当地勝率{w}", None)
            new_row["当地2率"] = row.get(f"当地2率{w}", None)
            new_row["モーター番号"] = row.get(f"モーター番号{w}", "")
            new_row["モーター2率"] = row.get(f"モーター2率{w}", None)
            new_row["ボート番号"] = row.get(f"ボート番号{w}", "")
            new_row["ボート2率"] = row.get(f"ボート2率{w}", None)
            for c in RESULT_COLS:
                new_row[c] = None
            new_rows.append(new_row)

    if not new_rows:
        print("✅ 追記対象の新規レースはありませんでした(既に追記済み)。")
        return

    combined = pd.concat([archive, pd.DataFrame(new_rows)], ignore_index=True)
    _save_archive(combined)
    print(f"💾 ローリングアーカイブに{len(new_rows)}行(艇)を追記しました: {ARCHIVE_PATH}")


def update_archive_with_results(results_csv_path: str | None) -> None:
    """前日分の結果(拡張済みresults_YYYYMMDD.csv)を、entryフェーズで追記済みの行に
    date+jcd+r+boat_numでマッチさせてUPDATEする(新規行は作らない)。
    """
    if results_csv_path is None or not os.path.exists(results_csv_path):
        print("⚠️ 結果データが無いため、アーカイブの更新をスキップします。")
        return

    results = pd.read_csv(results_csv_path, dtype={"jcd": str})
    archive = _load_archive()
    if archive.empty:
        print("⚠️ アーカイブが空のため、結果の反映をスキップします。")
        return

    archive = archive.set_index(["date", "jcd", "r", "boat_num"])
    archive[RESULT_COLS] = archive[RESULT_COLS].astype(object)  # 全てNoneで型推論されるのを防ぐ
    updated = 0

    for _, row in results.iterrows():
        date_s, jcd_s, r_s = _to_iso_date(row["date"]), str(row["jcd"]), str(row["r"])
        for w in range(1, 7):
            key = (date_s, jcd_s, r_s, str(w))
            if key not in archive.index:
                continue
            rank = row.get(f"着順{w}")
            if pd.notna(rank):
                archive.loc[key, "着順"] = rank
            for col, src in [
                ("決まり手", "決まり手"), ("天候", "天候"), ("風速", "風速"), ("波高", "波高"),
                ("水温", "水温"), ("気温", "気温"),
                ("単勝_払戻", "単勝_払戻"), ("複勝1_払戻", "複勝1_払戻"),
                ("3連単_払戻", "3rt"), ("3連複_払戻", "3連複_払戻"),
                ("2連単_払戻", "2連単_払戻"), ("2連複_払戻", "2連複_払戻"),
            ]:
                if src in row and pd.notna(row[src]):
                    archive.loc[key, col] = row[src]
            st_val = row.get(f"ST{w}")
            if pd.notna(st_val):
                archive.loc[key, "ST"] = st_val
            updated += 1

    archive = archive.reset_index()
    _save_archive(archive)
    print(f"💾 ローリングアーカイブに結果を反映しました({updated}行更新): {ARCHIVE_PATH}")


def trim_archive() -> None:
    """直近3年より古い行をアーカイブから削除する。"""
    archive = _load_archive()
    if archive.empty:
        return
    jst = pytz.timezone("Asia/Tokyo")
    cutoff = (datetime.now(jst) - pd.DateOffset(years=TRIM_YEARS)).strftime("%Y-%m-%d")
    before = len(archive)
    archive = archive[archive["date"].astype(str) >= cutoff]
    after = len(archive)
    _save_archive(archive)
    if before != after:
        print(f"🗑️ ローリングアーカイブをトリムしました: {before}行 → {after}行(cutoff={cutoff})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "trim":
        trim_archive()
