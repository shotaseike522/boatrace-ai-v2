"""日次予測: 当日の出走表(data/races_YYYYMMDD.csv)と、直近の月次再学習artifacts
(artifacts/scenario/配下)を使って、シナリオ1/2/3・アラインドペア・類似レース分析
を計算し、outputs/scenario_predictions_YYYYMMDD.csvに書き出す。

run_site_predictions_calibrated.py(旧AI Top5/2連複Best3/近似100レース系)の
役割を置き換える、シナリオ予想方式向けの新しい予測スクリプト。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boat_model import scenario_lib as sl
from boat_model.pl_probabilities import trifecta_probabilities, pair_probabilities, triple_label, TRIPLES


def _tickets_to_json(tickets: list[tuple]) -> str:
    """[((1,4,3), 0.034), ...] -> '[["1-4-3", 0.034], ...]' (JSON文字列)。
    app.py側でjson.loads()して表示用に整形する。
    """
    return json.dumps([["-".join(map(str, combo)), round(float(p), 4)] for combo, p in tickets],
                       ensure_ascii=False)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "scenario"
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
BASE_FIELDS = [
    "全国勝率", "全国2率", "当地勝率", "当地2率", "モーター2率", "ボート2率",
    "年齢", "体重", "1年出走数", "1年複勝率", "1年該当コース出走数", "1年該当コース複勝率",
]
ALIGNED_PAIR_THRESHOLD = 0.51


def strength_bin(rate: float) -> str:
    if rate < 5.0:
        return "弱"
    if rate < 6.0:
        return "中"
    return "強"


def load_today_features(races_csv: str) -> pd.DataFrame:
    """当日の出走表CSVを読み込み、強さモデルの特徴量(1年複勝率等含む)を
    ロング形式(1艇1行)で組み立てる。1年複勝率等は当日の出走表には
    含まれないため、ローリングアーカイブから別途計算して結合する。
    """
    from retrain_monthly import ARCHIVE_PATH, load_merged_archive, compute_trailing_year_stats

    races = pd.read_csv(races_csv, dtype={"jcd": str})
    archive = load_merged_archive()
    hist = compute_trailing_year_stats(archive).sort_values("date_dt")
    # 全体成績(1年出走数/1年複勝率)は登番の最新レコードから、該当コース成績
    # (1年該当コース出走数/1年該当コース複勝率)は(登番,boat_num)の最新レコードから
    # 取る。登番だけで最新1件に絞ると、その選手が直近に走った別コースの成績が
    # 今日のboat_numに紐づいてしまうため分けている。
    latest_overall = hist.drop_duplicates(subset=["登番"], keep="last").set_index("登番")[
        ["1年出走数", "1年複勝率"]]
    latest_by_course = hist.drop_duplicates(subset=["登番", "boat_num"], keep="last").set_index(
        ["登番", "boat_num"])[["1年該当コース出走数", "1年該当コース複勝率"]]

    rows = []
    for _, race in races.iterrows():
        for w in range(1, 7):
            toban = str(race[f"登番{w}"])
            row = {
                "date": race["date"], "jcd": race["jcd"], "r": race["r"], "boat_num": w, "登番": toban,
                "級別": race.get(f"級別{w}"),
                "全国勝率": race.get(f"勝率{w}"), "全国2率": race.get(f"全国2率{w}"),
                "当地勝率": race.get(f"当地勝率{w}"), "当地2率": race.get(f"当地2率{w}"),
                "モーター2率": race.get(f"モーター2率{w}"), "ボート2率": race.get(f"ボート2率{w}"),
                "年齢": race.get(f"年齢{w}"), "体重": race.get(f"体重{w}"),
            }
            if toban in latest_overall.index:
                row["1年出走数"] = latest_overall.loc[toban, "1年出走数"]
                row["1年複勝率"] = latest_overall.loc[toban, "1年複勝率"]
            else:
                row["1年出走数"] = np.nan
                row["1年複勝率"] = np.nan
            course_key = (toban, str(w))
            if course_key in latest_by_course.index:
                row["1年該当コース出走数"] = latest_by_course.loc[course_key, "1年該当コース出走数"]
                row["1年該当コース複勝率"] = latest_by_course.loc[course_key, "1年該当コース複勝率"]
            else:
                row["1年該当コース出走数"] = np.nan
                row["1年該当コース複勝率"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def score_races(features: pd.DataFrame, model: lgb.Booster, medians: dict) -> pd.DataFrame:
    df = features.copy()
    df["jcd_cat"] = df["jcd"].astype("category")
    df["級別num"] = df["級別"].map(CLASS_MAP).fillna(0)
    for f in BASE_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        # 欠損値は当日の少数レースの中央値ではなく、学習時(retrain_monthly.py)と
        # 同じ基準(直近3年学習データの中央値)で埋める。基準がズレると
        # 学習時と本番で同じ欠損値が別の値に化けてしまう(train-serve skew)。
        df[f] = df[f].fillna(medians[f])
    feature_cols = BASE_FIELDS + ["級別num", "boat_num", "jcd_cat"]
    df["score"] = model.predict(df[feature_cols])
    race_key = df["date"].astype(str) + "_" + df["jcd"].astype(str) + "_" + df["r"].astype(str)
    df["theta"] = np.exp(df["score"] - df.groupby(race_key)["score"].transform("max"))
    return df


def build_trow(theta_row: np.ndarray) -> pd.Series:
    probs = trifecta_probabilities(theta_row.reshape(1, 6))[0]
    data = {f"prob_{triple_label(t)}": probs[t] for t in range(len(TRIPLES))}
    return pd.Series(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("特徴量準備中...")
    features = load_today_features(args.input)

    print("強さモデル読み込み・スコアリング中...")
    model = lgb.Booster(model_file=str(ARTIFACTS_DIR / "strength_model.txt"))
    medians = pd.read_pickle(ARTIFACTS_DIR / "base_field_medians.pkl")
    scored = score_races(features, model, medians)

    print("選手スタイル・非1着時プロファイル・類似レース表読み込み中...")
    style = sl.load_racer_style()
    subplace = sl.load_subplace_profile()
    similar_race = sl.load_similar_race_table()
    similar_race = similar_race.set_index(["jcd", "strength_bin", "rough_bin"])
    payout_edges = sl.load_payout_overall_edges()
    payout_bucket_labels = []
    for i in range(20):
        lo, hi = payout_edges[i], payout_edges[i + 1]
        payout_bucket_labels.append(f"{lo:.0f}円〜" if np.isinf(hi) else f"{lo:.0f}〜{hi:.0f}円")

    results = []
    for (date, jcd, r), race_df in scored.groupby(["date", "jcd", "r"]):
        race_df = race_df.sort_values("boat_num")
        theta = race_df["theta"].to_numpy()
        win_prob = theta / theta.sum()
        toubans = dict(zip(race_df["boat_num"], race_df["登番"].astype(str)))
        trow = build_trow(theta)

        scenarios = sl.generate_scenarios(win_prob, trow, toubans, style, subplace, weighting="multiply")

        favorite = scenarios["scenario1"]["winner_boat"]
        challenger = scenarios["scenario2"]["winner_boat"]
        ranked1 = sl.rank_candidates((favorite,), trow)
        cand1 = sl.select_candidates(ranked1)
        ranked2 = sl.rank_candidates((challenger,), trow)
        cand2 = sl.select_candidates(ranked2)
        aligned_pair, aligned_prob = None, None
        if cand1 and cand2:
            pair1 = frozenset([favorite, cand1[0]])
            pair2 = frozenset([challenger, cand2[0]])
            if pair1 == pair2 and len(cand1) == 1 and len(cand2) == 1:
                aligned_pair = tuple(sorted(pair1))
                aligned_prob = ranked1[0][1] + ranked2[0][1]

        # 類似レース分析: 当日朝の時点では実測天候(波高/風速)が分からないため、
        # 荒れ度は問わずjcd×強さ帯だけで集計した参考値を使う
        # (直前情報が出た後に荒れ度も反映して再計算する運用を想定)
        fav_rate = float(race_df.loc[race_df["boat_num"] == favorite, "全国勝率"].iloc[0])
        this_bin = strength_bin(fav_rate)
        kimarite_dist, bucket_pct_actual, bucket_pct_deviation = None, None, None
        try:
            matches = similar_race.loc[(jcd, this_bin, slice(None))]
            if isinstance(matches, pd.Series):
                matches = matches.to_frame().T
            total_n = matches["n"].sum()
            if total_n > 0:
                combined_kimarite = {}
                combined_counts = np.zeros(20)
                for _, mrow in matches.iterrows():
                    weight = float(mrow["n"]) / total_n
                    for k, v in (mrow["kimarite_dist"] or {}).items():
                        combined_kimarite[k] = combined_kimarite.get(k, 0.0) + float(v) * weight
                    combined_counts += np.array(mrow["payout_bucket_counts"], dtype=float)
                kimarite_dist = {k: round(float(v), 1) for k, v in combined_kimarite.items()}
                total_payout_n = combined_counts.sum()
                if total_payout_n > 0:
                    bucket_pct = combined_counts / total_payout_n * 100
                    bucket_pct_actual = [round(float(p), 1) for p in bucket_pct]
                    bucket_pct_deviation = [round(float(p) - 5.0, 1) for p in bucket_pct]
        except KeyError:
            pass

        nige_rate = scenarios["scenario1"]["nige_rate"]
        makuri_rate = scenarios["scenario2"]["makuri_rate"]
        row = {
            "date": date, "jcd": jcd, "r": r,
            "favorite_boat": favorite,
            "favorite_nige_rate": round(float(nige_rate), 1) if nige_rate is not None else None,
            "scenario1_tickets": _tickets_to_json(scenarios["scenario1"]["tickets"]),
            "challenger_boat": challenger,
            "challenger_makuri_rate": round(float(makuri_rate), 1) if makuri_rate is not None else None,
            "scenario2_tickets": _tickets_to_json(scenarios["scenario2"]["tickets"]),
            "scenario3_tickets": _tickets_to_json(scenarios["scenario3"]["tickets"]),
            "aligned_pair": "-".join(map(str, aligned_pair)) if aligned_pair else None,
            "aligned_pair_prob": round(float(aligned_prob), 4) if aligned_prob is not None else None,
            "aligned_pair_flag": aligned_prob is not None and aligned_prob >= ALIGNED_PAIR_THRESHOLD,
            "strength_bin": this_bin,
            "kimarite_dist": json.dumps(kimarite_dist, ensure_ascii=False) if kimarite_dist else None,
            "payout_bucket_labels": json.dumps(payout_bucket_labels, ensure_ascii=False),
            "payout_bucket_pct": json.dumps(bucket_pct_actual) if bucket_pct_actual else None,
            "payout_bucket_deviation": json.dumps(bucket_pct_deviation) if bucket_pct_deviation else None,
        }
        results.append(row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"完了: {len(out_df)}レース分を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
