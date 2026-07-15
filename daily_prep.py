"""日次バッチ: 出走表取得 -> 新AIモデル(boat_model)での予測実行 -> 選手マスタ更新。

旧システム(エンジンA/B/Cのロジスティック回帰・パターンマスタ・類似100レース)は
boat_model配下の新システム(LightGBM direct120 / position6 / KNN500 / Pairwise)に
置き換えられた。このファイルはスクレイピングと選手マスタ更新のみを担当し、
予測計算そのものは run_site_predictions.py に委譲する。

出力される出走表CSVの列名は、新システム(boat_model.features)が要求する形式に
そろえている:
  - jcd: ゼロ埋め2桁文字列 (例: "09")
  - r: レース番号 (int)
  - 勝率1〜勝率6: 全国勝率の生値 (0〜10程度のレンジ。レース内平均差し引きは行わない)
  - 登番1〜登番6: 選手登録番号

旧システムの `相対勝率_1〜6`(平均差し引き済み)・`登番_1〜6`(アンダースコア区切り)
という列名は使わない。相対化は新システム側の add_basic_features() が内部で行う。
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pytz
import os
import subprocess
import sys
import time
import random

from boat_model.features import add_full_course_profile_features

boats = [1, 2, 3, 4, 5, 6]
venues_map = {
    "01": "01_桐生", "02": "02_戸田", "03": "03_江戸川", "04": "04_平和島", "05": "05_多摩川", "06": "06_浜名湖",
    "07": "07_蒲郡", "08": "08_常滑", "09": "09_津", "10": "10_三国", "11": "11_びわこ", "12": "12_住之江",
    "13": "13_尼崎", "14": "14_鳴門", "15": "15_丸亀", "16": "16_児島", "17": "17_宮島", "18": "18_徳山",
    "19": "19_下関", "20": "20_若松", "21": "21_芦屋", "22": "22_福岡", "23": "23_唐津", "24": "24_大村"
}

RACES_DIR = "data"
OUTPUTS_DIR = "outputs"
ARTIFACTS_DIR = "artifacts"


def safe_float(val):
    if not val:
        return 0.0
    val = str(val).replace('%', '').strip()
    if val in ['-', '- -', '']:
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def get_active_jcds(session, target_date_str):
    index_url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={target_date_str}"
    active_jcds = []
    place_dict = {v.split('_')[1]: int(k) for k, v in venues_map.items()}
    try:
        res = session.get(index_url, timeout=30)
        soup = BeautifulSoup(res.content, "html.parser")
        for td in soup.find_all("td", class_="is-arrow1 is-fBold is-fs15"):
            img = td.find("img", alt=True)
            if img and img["alt"] in place_dict:
                active_jcds.append(place_dict[img["alt"]])
    except Exception:
        active_jcds = list(range(1, 25))
    return active_jcds


def fetch_today_race_entries(session):
    """本日の出走表を取得し、新システム形式のCSV(data/races_YYYYMMDD.csv)に保存する。

    戻り値: (出走表CSVのパス または None, 取得した選手登録番号のリスト)
    """
    jst = pytz.timezone('Asia/Tokyo')
    hd_str = datetime.now(jst).strftime("%Y%m%d")
    print(f"\n--- [1] 出走表取得 ({hd_str}) を開始 ---")

    os.makedirs(RACES_DIR, exist_ok=True)
    out_file = os.path.join(RACES_DIR, f"races_{hd_str}.csv")

    # 💡 スマートスキップ: 今日分が既に存在する場合は通信処理をスキップ
    if os.path.exists(out_file):
        try:
            df_check = pd.read_csv(out_file, dtype={"jcd": str})
            if "date" in df_check.columns and str(df_check['date'].iloc[0]) == hd_str:
                print(f"✅ 本日 ({hd_str}) の出走表は取得済みのため、通信処理をスキップします。")
                tobans = []
                for w in range(1, 7):
                    col = f"登番{w}"
                    if col in df_check.columns:
                        tobans.extend(df_check[col].astype(str).tolist())
                return out_file, sorted({t for t in tobans if t.isdigit() and len(t) == 4})
        except Exception:
            pass

    active_jcds = get_active_jcds(session, hd_str)
    all_rows = []
    unique_tobans = set()

    for jcd_int in active_jcds:
        jcd_str = f"{jcd_int:02d}"
        for rno in range(1, 13):
            url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd_str}&hd={hd_str}"
            try:
                res = session.get(url, timeout=15)
                soup = BeautifulSoup(res.content, "html.parser")
                tbodies = soup.find_all("tbody", class_="is-fs12")
                if len(tbodies) != 6:
                    continue
                rates, tobans = [], []
                for tbody in tbodies:
                    tds = tbody.find("tr").find_all("td", recursive=False) if tbody.find("tr") else []
                    toban = (
                        tds[2].find("div", class_="is-fs11").get_text().split("/")[0].strip()
                        if len(tds) > 2 and tds[2].find("div", class_="is-fs11")
                        else ""
                    )
                    rate_txt = tds[4].get_text(separator="\n").strip().split('\n')[0] if len(tds) > 4 else ""
                    tobans.append(toban)
                    # 💡 生の全国勝率(0〜10程度)をそのまま使う。平均差し引きはしない。
                    rates.append(float(rate_txt) if rate_txt and rate_txt != "-.--" else 0.0)
                    if toban.isdigit() and len(toban) == 4:
                        unique_tobans.add(toban)

                if len(rates) == 6:
                    row = {"date": hd_str, "jcd": jcd_str, "r": rno}
                    for w in range(1, 7):
                        row[f"登番{w}"] = tobans[w - 1]
                        row[f"勝率{w}"] = rates[w - 1]
                    all_rows.append(row)
            except Exception:
                continue

    if not all_rows:
        print("⚠️ 本日の出走表が取得できませんでした。")
        return None, []

    df_today = pd.DataFrame(all_rows)
    df_today.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"💾 出走表を保存しました: {out_file} ({len(df_today)}レース)")
    return out_file, sorted(unique_tobans)


def run_site_predictions(races_csv):
    """run_site_predictions.py をサブプロセスとして実行し、予測CSVを生成する。"""
    if races_csv is None:
        print("⚠️ 出走表が無いため、予測処理をスキップします。")
        return None

    jst = pytz.timezone('Asia/Tokyo')
    hd_str = datetime.now(jst).strftime("%Y%m%d")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUTS_DIR, f"site_predictions_{hd_str}.csv")

    # 💡 二重実行防止: GAS(0:00〜1:00頃)が既に成功していれば、
    # GitHub Actions(6:15の保険実行)はこの予測処理をスキップする。
    if os.path.exists(output_csv):
        print(f"✅ 本日 ({hd_str}) の予測は既に作成済みのため、AI予測処理をスキップします: {output_csv}")
        return output_csv

    print(f"\n--- [2] AI予測の実行 ({hd_str}) を開始 ---")
    cmd = [
        sys.executable,
        "run_site_predictions_calibrated.py",
        "--input", races_csv,
        "--output", output_csv,
        "--artifacts-dir", ARTIFACTS_DIR,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"✅ AI予測が完了しました: {output_csv}")
        return output_csv
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ run_site_predictions.py の実行に失敗しました: {exc}")
        print(exc.stdout)
        print(exc.stderr)
        return None
    except FileNotFoundError:
        print("⚠️ run_site_predictions.py が見つかりません。リポジトリ直下で実行してください。")
        return None


def fetch_results_and_archive(session):
    """昨日の結果を取得し、過去データとして蓄積する。

    NOTE: 新システム(boat_model)は dataset_1_past / dataset_2_recent という
    まとまったCSVを月次で学習に使う設計のため、この関数は現時点では
    「結果を取得して outputs/results_YYYYMMDD.csv に保存するだけ」に留めている。
    daily_predictions.csv との突き合わせ・learning_data蓄積(旧仕様)は廃止した。
    新システム用の学習データ蓄積パイプラインは別途設計する。
    """
    jst = pytz.timezone('Asia/Tokyo')
    yesterday = datetime.now(jst) - pd.Timedelta(days=1)
    hd_str = yesterday.strftime("%Y%m%d")

    print(f"\n--- [0] 昨日の結果取得 ({hd_str}) を開始 ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUTS_DIR, f"results_{hd_str}.csv")
    if os.path.exists(out_file):
        print(f"✅ {hd_str} の結果は取得済みのため、通信処理をスキップします。")
        return

    active_jcds = get_active_jcds(session, hd_str)
    all_results = []

    for jcd_int in active_jcds:
        jcd_str = f"{jcd_int:02d}"
        for rno in range(1, 13):
            url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd_str}&hd={hd_str}"
            try:
                res = session.get(url, timeout=15)
                soup = BeautifulSoup(res.content, "html.parser")
                r1, r2, r3, p = None, None, None, None
                for tr in soup.find_all("tr"):
                    if "3連単" in tr.get_text():
                        nums = tr.find_all("span", class_="numberSet1_number")
                        payout = tr.find("span", class_="is-payout1")
                        if len(nums) >= 3 and payout:
                            r1, r2, r3 = nums[0].text.strip(), nums[1].text.strip(), nums[2].text.strip()
                            p = payout.text.replace("¥", "").replace(",", "").replace("円", "").strip()
                            break
                if r1 and p:
                    all_results.append(
                        {"date": hd_str, "jcd": jcd_str, "r": rno, "r1": r1, "r2": r2, "r3": r3, "3rt": p}
                    )
            except Exception:
                continue

    if not all_results:
        print("⚠️ 昨日の結果が取得できませんでした。")
        return

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"💾 昨日の結果を保存しました: {out_file} ({len(df_results)}レース)")


def update_racer_master(session, today_tobans):
    print("\n--- [3] 選手マスタの自動更新を開始 ---")
    if not today_tobans:
        print("✅ 新規の選手データがないため、マスタ更新をスキップします。")
        return

    jst = pytz.timezone('Asia/Tokyo')
    master_file = 'racer_master.csv'

    if os.path.exists(master_file):
        df_master = pd.read_csv(master_file)
        df_master['登録番号'] = df_master['登録番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    else:
        df_master = pd.DataFrame(columns=['登録番号', '更新日'])

    existing_tobans = set(df_master['登録番号'].tolist())
    new_racers = [t for t in today_tobans if t not in existing_tobans]
    seven_days_ago = pd.to_datetime(datetime.now(jst).date()) - pd.Timedelta(days=7)

    if not df_master.empty:
        df_master['更新日'] = pd.to_datetime(df_master['更新日'], errors='coerce')
        old_racers = (
            df_master[
                df_master['登録番号'].isin(today_tobans)
                & ((df_master['更新日'] < seven_days_ago) | (df_master['更新日'].isna()))
            ]
            .sort_values('更新日')['登録番号']
            .astype(str)
            .tolist()
        )
    else:
        old_racers = []

    target_racers = (new_racers + old_racers)[:50]
    if not target_racers:
        print("✅ 全選手のデータが最新（7日以内）のため、マスタ更新をスキップします。")
        return

    updated_data = []
    for toban in target_racers:
        time.sleep(random.uniform(1.5, 3.0))
        url = f"https://www.boatrace.jp/owpc/pc/data/racersearch/course?toban={toban}"
        racer_info = {"登録番号": toban, "更新日": datetime.now(jst).strftime("%Y-%m-%d")}
        for c in range(1, 7):
            racer_info[f"{c}コース_進入率"] = 0.0
            racer_info[f"{c}コース_1着率"] = 0.0
            racer_info[f"{c}コース_2着率"] = 0.0
            racer_info[f"{c}コース_3着率"] = 0.0
            racer_info[f"{c}コース_平均ST"] = 0.00
            racer_info[f"{c}コース_ST順"] = 0.0

        try:
            res = session.get(url, timeout=15)
            if res.url != url or "データが存在しないので" in res.text:
                updated_data.append(racer_info)
                continue
            soup = BeautifulSoup(res.content, "html.parser")
            tables = soup.find_all("div", class_="table1")
            if tables and len(tables) >= 4:
                for i, l in enumerate(tables[0].find_all("span", class_="table1_progress2Label")):
                    racer_info[f"{i+1}コース_進入率"] = safe_float(l.text)
                for i, l in enumerate(tables[1].find_all("span", class_="table1_progress2Label")):
                    bars = tables[1].find_all("tr")[i + 1].find_all("span", class_="is-progress")
                    if len(bars) >= 1:
                        racer_info[f"{i+1}コース_1着率"] = safe_float(bars[0]['style'].split(':')[1])
                    if len(bars) >= 2:
                        racer_info[f"{i+1}コース_2着率"] = safe_float(bars[1]['style'].split(':')[1])
                    if len(bars) >= 3:
                        racer_info[f"{i+1}コース_3着率"] = safe_float(bars[2]['style'].split(':')[1])
                for i, l in enumerate(tables[2].find_all("span", class_="table1_progress2Label")):
                    racer_info[f"{i+1}コース_平均ST"] = safe_float(l.text)
                for i, l in enumerate(tables[3].find_all("span", class_="table1_progress2Label")):
                    racer_info[f"{i+1}コース_ST順"] = safe_float(l.text)
            updated_data.append(racer_info)
        except Exception:
            continue

    if updated_data:
        df_new = pd.DataFrame(updated_data)
        df_combined = (
            pd.concat([df_new, df_master]).drop_duplicates(subset=['登録番号'], keep='first')
            if not df_master.empty
            else df_new
        )
        df_combined['更新日'] = pd.to_datetime(df_combined['更新日'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_combined.to_csv(master_file, index=False, encoding='utf-8-sig')
        print(f"💾 選手マスタを更新しました: {len(target_racers)}名")


def enrich_races_with_racer_master(races_csv_path):
    """出走表CSVに、その時点のracer_master.csvから選手のコース別成績を紐づけて追記する。

    racer_master.csvは日々更新されるローリングスナップショットのため、
    ここで追記される値は「その日の出走表取得時点で分かっていた最新の選手データ」であり、
    月次で過去分をまとめてマージするより時系列的にはむしろ正確に近い
    （月次マージだと、後の月に更新された値がその月より前のレースにも
    紐づいてしまい、ずれが大きくなる）。
    """
    if races_csv_path is None:
        return
    master_file = 'racer_master.csv'
    if not os.path.exists(master_file):
        print("⚠️ racer_master.csvが無いため、選手データの紐づけをスキップします。")
        return

    df = pd.read_csv(races_csv_path, dtype={"jcd": str})
    if "1号艇_1コース_1着率" in df.columns:
        print("✅ 出走表には既に選手データが紐づけ済みのため、スキップします。")
        return

    racer_master = pd.read_csv(master_file)
    enriched = add_full_course_profile_features(df, racer_master)
    enriched.to_csv(races_csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 出走表に選手データ（全コース分）を紐づけました: {races_csv_path}")


def run_pattern_alert():
    """AI予想Top5・近似100レースTop5の一致パターンを検出し、LINE配信用JSONを出力する。

    通常AI予想やサイト表示には一切影響しない別枠の処理。ここで例外が起きても
    日次バッチ全体を止めないよう、失敗は警告表示のみに留める。
    """
    jst = pytz.timezone('Asia/Tokyo')
    hd_str = datetime.now(jst).strftime("%Y%m%d")
    print(f"\n--- [4] 一致パターン検出 ({hd_str}) を開始 ---")
    try:
        result = subprocess.run(
            [sys.executable, "run_pattern_alert.py", "--date", hd_str],
            check=True, capture_output=True, text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as exc:
        print(f"⚠️ 一致パターン検出の実行に失敗しました（他の処理には影響しません）: {exc}")


if __name__ == "__main__":
    main_session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    main_session.mount('https://', HTTPAdapter(max_retries=retries))
    main_session.headers.update({'User-Agent': 'Mozilla/5.0'})

    fetch_results_and_archive(main_session)
    races_csv_path, today_racer_tobans = fetch_today_race_entries(main_session)
    run_site_predictions(races_csv_path)
    update_racer_master(main_session, today_racer_tobans)
    enrich_races_with_racer_master(races_csv_path)
    run_pattern_alert()
