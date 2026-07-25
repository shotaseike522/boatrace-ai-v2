"""日次バッチ: 前日結果取得(mbrace.or.jp K形式) -> ローリング3年アーカイブ更新
   -> 当日出走表取得(mbrace.or.jp B形式) -> ローリング3年アーカイブ追記・トリム
   -> シナリオ予測実行(run_scenario_predictions.py) -> 選手マスタ更新。

旧システム(AI Top5・2連複Best3・近似100レースのKNN)は展開シナリオ方式
(本命/対抗/他有力の3連単買い目 + 類似レース分析)に置き換えられた。
出走表・結果の取得は、以前はboatrace.jpのHTML画面をスクレイピングしていたが、
mbrace.or.jp公式のB/K形式LZHアーカイブを直接ダウンロード・解析する方式
(boat_model.parse_bk_format, mbrace_download.py)に切り替えている。

強さモデル・選手スタイル・類似レース表の再学習は月次(retrain_monthly.py、
別ワークフローから実行)で行い、このファイルでは行わない。
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from pathlib import Path
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


def fetch_today_race_entries(session):
    """本日の出走表を取得し、新システム形式のCSV(data/races_YYYYMMDD.csv)に保存する。

    以前はboatrace.jpのHTML画面をスクレイピングしていたが、mbrace.or.jp公式の
    番組表(B形式)LZHアーカイブが当日分もレース開始前に公開されていることを
    確認できたため、そちらを直接ダウンロード・解析する方式に切り替えた
    (HTML構造の変化に弱いスクレイピングより、21年分実績のある固定長パーサー
    `boat_model.parse_bk_format` を使う方が壊れにくい)。sessionパラメータは
    他の関数との呼び出し互換のために残しているが、この関数内では使用しない。

    戻り値: (出走表CSVのパス または None, 取得した選手登録番号のリスト)
    """
    import tempfile

    from boat_model.parse_bk_format import extract_lzh_text, parse_b_file
    from mbrace_download import download_lzh

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

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        lzh_path = download_lzh("B", hd_str, tmp_dir)
        if lzh_path is None:
            print(f"⚠️ 本日 ({hd_str}) の番組表LZHがまだ公開されていません。")
            return None, []
        raw = extract_lzh_text(lzh_path, tmp_dir)
        b_races = parse_b_file(raw)

    all_rows = []
    unique_tobans = set()

    for race in b_races:
        boats = race["boats"]
        if len(boats) != 6:
            continue
        row = {"date": hd_str, "jcd": race["jcd"], "r": race["r"]}
        for w in range(1, 7):
            b = boats[w - 1]
            toban = b["登番"]
            row[f"登番{w}"] = toban
            row[f"級別{w}"] = b["級別"]
            row[f"年齢{w}"] = safe_float(b["年齢"])
            row[f"体重{w}"] = safe_float(b["体重"])
            # 💡 生の全国勝率(0〜10程度)をそのまま使う。平均差し引きはしない。
            row[f"勝率{w}"] = safe_float(b["全国勝率"])
            row[f"全国2率{w}"] = safe_float(b["全国2率"])
            row[f"当地勝率{w}"] = safe_float(b["当地勝率"])
            row[f"当地2率{w}"] = safe_float(b["当地2率"])
            row[f"モーター番号{w}"] = b["モーターNo"]
            row[f"モーター2率{w}"] = safe_float(b["モーター2率"])
            row[f"ボート番号{w}"] = b["ボートNo"]
            row[f"ボート2率{w}"] = safe_float(b["ボート2率"])
            if toban.isdigit() and len(toban) == 4:
                unique_tobans.add(toban)
        all_rows.append(row)

    if not all_rows:
        print("⚠️ 本日の出走表が取得できませんでした。")
        return None, []

    df_today = pd.DataFrame(all_rows)
    df_today.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"💾 出走表を保存しました: {out_file} ({len(df_today)}レース)")
    return out_file, sorted(unique_tobans)


def run_site_predictions(races_csv):
    """run_scenario_predictions.py をサブプロセスとして実行し、シナリオ予測CSVを生成する。

    以前はrun_site_predictions_calibrated.py(AI Top5/2連複Best3/近似100レース系)
    を呼んでいたが、展開シナリオ方式(run_scenario_predictions.py)に置き換えた。
    """
    if races_csv is None:
        print("⚠️ 出走表が無いため、予測処理をスキップします。")
        return None

    jst = pytz.timezone('Asia/Tokyo')
    hd_str = datetime.now(jst).strftime("%Y%m%d")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUTS_DIR, f"scenario_predictions_{hd_str}.csv")

    # 💡 二重実行防止: GAS(0:00〜1:00頃)が既に成功していれば、
    # GitHub Actions(6:15の保険実行)はこの予測処理をスキップする。
    if os.path.exists(output_csv):
        print(f"✅ 本日 ({hd_str}) の予測は既に作成済みのため、シナリオ予測処理をスキップします: {output_csv}")
        return output_csv

    print(f"\n--- [2] シナリオ予測の実行 ({hd_str}) を開始 ---")
    cmd = [
        sys.executable,
        "run_scenario_predictions.py",
        "--input", races_csv,
        "--output", output_csv,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"✅ シナリオ予測が完了しました: {output_csv}")
        return output_csv
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ run_scenario_predictions.py の実行に失敗しました: {exc}")
        print(exc.stdout)
        print(exc.stderr)
        return None
    except FileNotFoundError:
        print("⚠️ run_scenario_predictions.py が見つかりません。リポジトリ直下で実行してください。")
        return None


def fetch_results_and_archive(session):
    """昨日の結果を取得し、過去データとして蓄積する。

    以前はboatrace.jpのHTML画面をスクレイピングしていたが、mbrace.or.jp公式の
    競走成績(K形式)LZHアーカイブを直接ダウンロード・解析する方式に切り替えた
    (fetch_today_race_entriesと同じ理由)。決まり手・天候(風速/波高等)・
    全6艇のST・主要payout種別(単勝を除く2連単/2連複/3連単/3連複)がまとめて
    取得できる。ここでの出力(outputs/results_YYYYMMDD.csv)は
    rolling_archive.update_archive_with_results() が読み込み、ローリング3年
    アーカイブ(data/site_archive_rolling3y.csv)に反映する。
    sessionパラメータは呼び出し互換のために残しているが使用しない。
    """
    import tempfile

    from boat_model.parse_bk_format import extract_lzh_text, parse_k_file
    from mbrace_download import download_lzh

    jst = pytz.timezone('Asia/Tokyo')
    yesterday = datetime.now(jst) - pd.Timedelta(days=1)
    hd_str = yesterday.strftime("%Y%m%d")

    print(f"\n--- [0] 昨日の結果取得 ({hd_str}) を開始 ---")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUTS_DIR, f"results_{hd_str}.csv")
    if os.path.exists(out_file):
        print(f"✅ {hd_str} の結果は取得済みのため、通信処理をスキップします。")
        return out_file

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        lzh_path = download_lzh("K", hd_str, tmp_dir)
        if lzh_path is None:
            print(f"⚠️ 昨日 ({hd_str}) の競走成績LZHがまだ公開されていません。")
            return None
        raw = extract_lzh_text(lzh_path, tmp_dir)
        k_races, k_payouts = parse_k_file(raw)

    all_results = []
    for race in k_races:
        boats = race["boats"]
        by_rank = {b["着順"]: b for b in boats}
        r1 = by_rank.get("01", {}).get("艇番")
        r2 = by_rank.get("02", {}).get("艇番")
        r3 = by_rank.get("03", {}).get("艇番")
        if not r1:
            continue

        payout = k_payouts.get((race["jcd"], race["r"]), {})
        weather = race.get("weather", {})
        row = {
            "date": hd_str, "jcd": race["jcd"], "r": race["r"],
            "r1": r1, "r2": r2, "r3": r3,
            "3rt": payout.get("3連単_払戻"),
            "決まり手": race.get("決まり手"),
            "天候": weather.get("天候"), "風速": weather.get("風速"), "波高": weather.get("波高"),
            "水温": None, "気温": None,
            "単勝_払戻": None, "複勝1_払戻": None,
            "3連単_払戻": payout.get("3連単_払戻"), "3連複_払戻": payout.get("3連複_払戻"),
            "2連単_払戻": payout.get("2連単_払戻"), "2連複_払戻": payout.get("2連複_払戻"),
        }
        for boat in boats:
            w = boat["艇番"]
            if w.isdigit():
                row[f"着順{w}"] = boat["着順"]
                row[f"ST{w}"] = boat["ST"]
        all_results.append(row)

    if not all_results:
        print("⚠️ 昨日の結果が取得できませんでした。")
        return None

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"💾 昨日の結果を保存しました: {out_file} ({len(df_results)}レース)")
    return out_file


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
    import rolling_archive

    main_session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    main_session.mount('https://', HTTPAdapter(max_retries=retries))
    main_session.headers.update({'User-Agent': 'Mozilla/5.0'})

    results_csv_path = fetch_results_and_archive(main_session)
    rolling_archive.update_archive_with_results(results_csv_path)

    races_csv_path, today_racer_tobans = fetch_today_race_entries(main_session)
    rolling_archive.append_entries_to_archive(races_csv_path)
    rolling_archive.trim_archive()

    # TODO(Phase5): run_site_predictions()をrun_scenario_predictions.pyの呼び出しに置き換える
    run_site_predictions(races_csv_path)
    update_racer_master(main_session, today_racer_tobans)
    enrich_races_with_racer_master(races_csv_path)
    run_pattern_alert()
