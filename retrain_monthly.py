"""月次再学習: サイト自前のローリングアーカイブ(data/site_archive_rolling3y.csv)
から、強さモデル(3年+jcd)・選手スタイル(5年)・類似レース表(3年)を再構築する。

boatrace-niren-wide-experimentフォルダで検証済みの手法をそのまま踏襲:
  - 強さモデル: LightGBM lambdarank、直近3年(動的)、jcd特徴量追加
  - 選手スタイル(まくり系率・逃げ率): 直近5年(動的)
  - 類似レース(決まり手%・3連単配当分布): 直近3年(動的)、(jcd×強さ帯×荒れ度)

日次(daily_prep.py)ではなく月1回の実行を想定。GitHub Actionsの
weekly_retrain.yml(または monthly_retrain.yml)から呼び出される。
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE_PATH = DATA_DIR / "site_archive_rolling3y.csv"
KIMARITE_SEED_PATH = DATA_DIR / "kimarite_payout_seed_pre202607.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "scenario"

TRAIN_YEARS = 3
VALID_YEARS = 1
STYLE_YEARS = 5
SIMILAR_RACE_YEARS = 3
STYLE_MIN_WINS = 30
KNN_K = 500
KNN_MIN_GROUP_N = 200  # 較正比率を信頼するための最低件数(一致/不一致それぞれ)

BASE_FIELDS = [
    "全国勝率", "全国2率", "当地勝率", "当地2率", "モーター2率", "ボート2率",
    "年齢", "体重", "1年出走数", "1年複勝率", "1年該当コース出走数", "1年該当コース複勝率",
]
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def strength_bin(rate: float) -> str:
    if rate < 5.0:
        return "弱"
    if rate < 6.0:
        return "中"
    return "強"


def rough_bin(wave: float, wind: float) -> str:
    total = wave + wind
    if total <= 2:
        return "穏やか"
    if total <= 6:
        return "普通"
    return "荒れ"


def load_merged_archive() -> pd.DataFrame:
    """ローリングアーカイブに、決まり手シード(古い日付分)をdate+jcd+rで補完してマージする。
    アーカイブ自身の決まり手・払戻列がnull(=まだ結果反映されていない当日分、または
    シード期間で元々埋まっていない分)の行だけ、シード側の値で埋める。
    """
    archive = pd.read_csv(ARCHIVE_PATH, dtype={"jcd": str, "r": str, "boat_num": str, "登番": str})
    if not KIMARITE_SEED_PATH.exists():
        return archive

    seed = pd.read_csv(KIMARITE_SEED_PATH, dtype={"jcd": str, "r": str})
    seed_cols = ["決まり手", "天候", "風向", "風速", "波高",
                 "3連単_払戻", "3連複_払戻", "2連単_払戻", "2連複_払戻"]
    seed_small = seed[["date", "jcd", "r"] + [c for c in seed_cols if c in seed.columns]].drop_duplicates(
        subset=["date", "jcd", "r"])
    archive = archive.merge(seed_small, on=["date", "jcd", "r"], how="left", suffixes=("", "_seed"))

    for c in seed_cols:
        seed_c = f"{c}_seed"
        if seed_c in archive.columns:
            archive[c] = archive[c].where(archive[c].notna(), archive[seed_c])
            archive = archive.drop(columns=[seed_c])

    return archive


def compute_trailing_year_stats(archive: pd.DataFrame) -> pd.DataFrame:
    """各行(登番×レース)について、そのレース日を含まない過去365日の
    全体複勝率・該当コース複勝率をsearchsorted+cumsum方式で計算して追加する。
    """
    df = archive.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_top2"] = df["着順"].isin(["01", "02"]).astype(int)
    df["is_start"] = df["登番"].notna().astype(int)

    def _trailing(group_cols):
        daily = (
            df.groupby(group_cols + ["date_dt"])
            .agg(day_starts=("is_start", "sum"), day_top2=("is_top2", "sum"))
            .reset_index()
            .sort_values(group_cols + ["date_dt"])
        )
        out_frames = []
        for _, g in daily.groupby(group_cols, sort=False):
            g = g.sort_values("date_dt")
            dates = g["date_dt"].to_numpy()
            starts = g["day_starts"].to_numpy(dtype=float)
            top2 = g["day_top2"].to_numpy(dtype=float)
            cum_starts = np.concatenate([[0.0], np.cumsum(starts)])
            cum_top2 = np.concatenate([[0.0], np.cumsum(top2)])
            window_start = dates - np.timedelta64(365, "D")
            idx_hi = np.arange(len(dates))
            idx_lo = np.searchsorted(dates, window_start, side="left")
            g = g.copy()
            g["starts_365"] = cum_starts[idx_hi] - cum_starts[idx_lo]
            g["top2_365"] = cum_top2[idx_hi] - cum_top2[idx_lo]
            out_frames.append(g)
        return pd.concat(out_frames, ignore_index=True)

    overall = _trailing(["登番"]).rename(columns={"starts_365": "ov_starts", "top2_365": "ov_top2"})
    by_course = _trailing(["登番", "boat_num"]).rename(columns={"starts_365": "co_starts", "top2_365": "co_top2"})

    df = df.merge(overall, on=["登番", "date_dt"], how="left")
    df = df.merge(by_course, on=["登番", "boat_num", "date_dt"], how="left")

    with np.errstate(invalid="ignore", divide="ignore"):
        df["1年出走数"] = df["ov_starts"]
        df["1年複勝率"] = np.where(df["ov_starts"] > 0, df["ov_top2"] / df["ov_starts"] * 100, np.nan)
        df["1年該当コース出走数"] = df["co_starts"]
        df["1年該当コース複勝率"] = np.where(df["co_starts"] > 0, df["co_top2"] / df["co_starts"] * 100, np.nan)

    return df


def build_wide_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    """ロング形式(1行=1艇)のアーカイブを、近似100レース(KNN)モデル用のワイド形式
    (date,jcd,r,勝率1..6,r1,r2,r3)に変換する。着順01/02/03の艇番をr1/r2/r3とする。
    6艇分の全国勝率・上位3着順が全て揃っているレースだけを残す。
    """
    d = df.copy()
    d["boat_num_num"] = pd.to_numeric(d["boat_num"], errors="coerce")
    d["全国勝率_num"] = pd.to_numeric(d["全国勝率"], errors="coerce")
    wide_rate = d.pivot_table(index=["date", "jcd", "r"], columns="boat_num_num", values="全国勝率_num", aggfunc="first")
    wide_rate.columns = [f"勝率{int(c)}" for c in wide_rate.columns]
    rate_cols = [f"勝率{i}" for i in range(1, 7)]
    wide_rate = wide_rate.reindex(columns=rate_cols).dropna(subset=rate_cols)

    finish = d[d["着順"].isin(["01", "02", "03"])][["date", "jcd", "r", "着順", "boat_num_num"]]
    finish_wide = finish.pivot_table(index=["date", "jcd", "r"], columns="着順", values="boat_num_num", aggfunc="first")
    finish_wide = finish_wide.rename(columns={"01": "r1", "02": "r2", "03": "r3"})
    finish_wide = finish_wide.reindex(columns=["r1", "r2", "r3"]).dropna()

    out = wide_rate.join(finish_wide, how="inner").reset_index()
    out["jcd"] = out["jcd"].astype(str).str.zfill(2)
    out["r"] = out["r"].astype(int)
    for c in ["r1", "r2", "r3"]:
        out[c] = out[c].astype(int)
    return out


def train_knn_and_calibration(
    df: pd.DataFrame, model: lgb.Booster, medians: dict,
    train_start, train_end, valid_start, valid_end,
) -> tuple:
    """近似100レース(KNN)モデルを学習し、強さモデルとの「一致/不一致」で
    的中確率の較正比率(calibration ratio)を計算する。

    2026年7月の検証(2024/2025/2026年6月までの3年分、それぞれ他の期間で
    比率を決めて未知の1年でテストする方式)で、強さモデルの本命(2連複/3連複)予想が
    KNNモデルの本命予想と一致した場合は的中率が有意に高く(3連複+5.6〜5.9pt、
    2連複+8.6〜9.1pt、3年間ほぼ一定)、この一致状況で確率を較正するとBrierスコアが
    3年とも一貫して改善することを確認済み。詳細はこのコミットの説明を参照。
    """
    from itertools import combinations

    from boat_model.models import ApproxRaceKNNModel
    from boat_model.features import add_basic_features, TRIFECTA_PERMUTATIONS
    from boat_model.pl_probabilities import pair_probabilities, trifecta_probabilities

    pairs15 = list(combinations(range(1, 7), 2))
    pair_index = {p: i for i, p in enumerate(pairs15)}
    trios20 = list(combinations(range(1, 7), 3))
    trio_index20 = {t: i for i, t in enumerate(trios20)}
    perm_to_pair20 = np.array([pair_index[tuple(sorted(p[:2]))] for p in TRIFECTA_PERMUTATIONS])
    perm_to_trio20 = np.array([trio_index20[tuple(sorted(p[:3]))] for p in TRIFECTA_PERMUTATIONS])

    wide = build_wide_outcome_table(df)
    wide_train = wide[(wide["date"] >= str(train_start.date())) & (wide["date"] < str(train_end.date()))]
    wide_valid = wide[(wide["date"] >= str(valid_start.date())) & (wide["date"] < str(valid_end.date()))]
    print(f"KNN較正: 学習プール{len(wide_train)}レース, 較正検証(強さモデルにとってもvalid期間){len(wide_valid)}レース")

    # 較正比率を出すためだけの、train期間のみで学習したKNN(valid期間を漏らさない)
    knn_calib = ApproxRaceKNNModel(k=KNN_K, weighted=False).fit(wide_train)

    # --- valid期間の強さモデル自身のtop pair/trioを計算(全レース一括のベクトル演算) ---
    feature_cols = BASE_FIELDS + ["級別num", "boat_num", "jcd_cat"]
    v = df[(df["date_dt"] >= valid_start) & (df["date_dt"] < valid_end)].copy()
    v["jcd_cat"] = v["jcd"].astype("category")
    v["級別num"] = v["級別"].map(CLASS_MAP).fillna(0)
    for f in BASE_FIELDS:
        v[f] = pd.to_numeric(v[f], errors="coerce").fillna(medians[f])
    v["boat_num"] = pd.to_numeric(v["boat_num"], errors="coerce")
    v["race_id"] = v["date"] + "_" + v["jcd"] + "_" + v["r"]
    v["score"] = model.predict(v[feature_cols])
    v = v.sort_values(["race_id", "boat_num"])
    race_sizes = v.groupby("race_id", sort=False).size()
    v = v[v["race_id"].isin(race_sizes[race_sizes == 6].index)]

    race_ids = v["race_id"].drop_duplicates().to_numpy()
    score_mat = v["score"].to_numpy().reshape(-1, 6)
    theta_mat = np.exp(score_mat - score_mat.max(axis=1, keepdims=True))
    quinella_probs, _ = pair_probabilities(theta_mat)
    tri_probs = trifecta_probabilities(theta_mat)
    n_races = len(race_ids)
    trio_probs = np.zeros((n_races, 20))
    for k in range(120):
        trio_probs[:, perm_to_trio20[k]] += tri_probs[:, k]
    meta = v.drop_duplicates(subset="race_id")[["race_id", "date", "jcd", "r"]].reset_index(drop=True)
    new_tbl = pd.DataFrame({
        "race_id": race_ids,
        "new_top_pair_idx": np.argmax(quinella_probs, axis=1),
        "new_top_pair_prob": quinella_probs.max(axis=1),
        "new_top_trio_idx": np.argmax(trio_probs, axis=1),
        "new_top_trio_prob": trio_probs.max(axis=1),
    }).merge(meta, on="race_id", how="left")

    # --- 同じvalid期間でKNNのtop pair/trioを計算 ---
    valid_feat = wide_valid.copy()
    valid_feat["race_id"] = valid_feat["date"] + "_" + valid_feat["jcd"] + "_" + valid_feat["r"].astype(str)

    eval_df = add_basic_features(valid_feat)
    knn_proba = knn_calib.predict_proba(eval_df)
    n = len(eval_df)
    knn_pair = np.zeros((n, 15))
    knn_trio = np.zeros((n, 20))
    for k in range(120):
        knn_pair[:, perm_to_pair20[k]] += knn_proba[:, k]
        knn_trio[:, perm_to_trio20[k]] += knn_proba[:, k]
    knn_tbl = pd.DataFrame({
        "race_id": valid_feat["race_id"].to_numpy(),
        "knn_top_pair_idx": np.argmax(knn_pair, axis=1),
        "knn_top_trio_idx": np.argmax(knn_trio, axis=1),
    })

    # --- 実際の着順(r1,r2,r3)からactual pair/trio ---
    actual = wide_valid.copy()
    actual["race_id"] = actual["date"] + "_" + actual["jcd"] + "_" + actual["r"].astype(str)
    actual_pair_idx = actual.apply(lambda r: pair_index[tuple(sorted((int(r["r1"]), int(r["r2"]))))], axis=1)
    actual_trio_idx = actual.apply(lambda r: trio_index20[tuple(sorted((int(r["r1"]), int(r["r2"]), int(r["r3"]))))], axis=1)
    actual_tbl = pd.DataFrame({
        "race_id": actual["race_id"].to_numpy(), "actual_pair_idx": actual_pair_idx.to_numpy(),
        "actual_trio_idx": actual_trio_idx.to_numpy(),
    })

    merged = new_tbl.merge(knn_tbl, on="race_id", how="inner").merge(actual_tbl, on="race_id", how="inner")
    merged["pair_agree"] = merged["new_top_pair_idx"] == merged["knn_top_pair_idx"]
    merged["trio_agree"] = merged["new_top_trio_idx"] == merged["knn_top_trio_idx"]
    merged["pair_hit"] = merged["new_top_pair_idx"] == merged["actual_pair_idx"]
    merged["trio_hit"] = merged["new_top_trio_idx"] == merged["actual_trio_idx"]
    print(f"KNN較正: 較正検証で使えたレース数 {len(merged)}")

    def calib_ratio(mask_col, hit_col, prob_col, agree_val):
        g = merged[merged[mask_col] == agree_val]
        if len(g) < KNN_MIN_GROUP_N or g[prob_col].mean() <= 0:
            return 1.0
        return float(g[hit_col].mean() / g[prob_col].mean())

    for grp_val, label in [(True, "match"), (False, "mismatch")]:
        g = merged[merged["trio_agree"] == grp_val]
        print(f"  [診断] trio {label}: n={len(g)}, "
              f"mean_pred_prob={g['new_top_trio_prob'].mean():.4f}, actual_hit_rate={g['trio_hit'].mean():.4f}")
    import os
    if os.environ.get("KNN_CALIB_DEBUG_DIR"):
        merged.to_pickle(Path(os.environ["KNN_CALIB_DEBUG_DIR"]) / "calib_merged_debug.pkl")

    # 較正比率はvalid期間(直近1年)だけから計算するため、月ごとの学習結果次第で
    # ノイズが乗る。2026年7月時点の実データで実際に確認された例: 一致時の実際の
    # 的中率は不一致時より+5.3pt高く(2024/2025/2026年の3年分の検証と一貫して
    # 同水準)効果自体は本物だが、モデル自身の予測確率も一致時に平均して高めに
    # 出る(0.321 vs 0.253)ため、「実的中率÷予測確率」の比率にすると相殺されて
    # ほぼ差が消え、この月はたまたま僅かに逆転していた(match比率0.8139 <
    # mismatch比率0.8239)。
    #
    # 1回のvalid期間だけを見て判断すると、こうしたモデルの再学習ごとの揺れに
    # 引きずられて逆方向の補正を本番に出しかねない。2024/2025/2026年6月の
    # 3年分の独立した検証(各年、他の2年で比率を決めて未知の年でテストする
    # 方式)で得られた比率は一致・不一致で常に明確な差があり安定していたため、
    # これを事前分布(prior)とし、当月の実測比率とブレンド(事前分布の重みを
    # 高めにして、1か月分のノイズで大きく振られないようにする)して使う。
    PRIOR_TRIO_MATCH = 0.819      # 2024:0.8095 / 2025:0.8126 / 2026H1:0.8337 の平均
    PRIOR_TRIO_MISMATCH = 0.766   # 2024:0.7610 / 2025:0.7548 / 2026H1:0.7828 の平均
    PRIOR_PAIR_MATCH = 0.955      # walk-forward検証(他2年で学習)の3fold平均
    PRIOR_PAIR_MISMATCH = 0.859
    PRIOR_WEIGHT = 0.7
    MIN_RATIO_GAP = 0.02

    def blend_ratio_pair(match_raw, mismatch_raw, prior_match, prior_mismatch, label):
        match_blend = PRIOR_WEIGHT * prior_match + (1 - PRIOR_WEIGHT) * match_raw
        mismatch_blend = PRIOR_WEIGHT * prior_mismatch + (1 - PRIOR_WEIGHT) * mismatch_raw
        print(f"  [較正] {label}: 今月実測 match={match_raw:.4f}/mismatch={mismatch_raw:.4f} → "
              f"事前分布とブレンド後 match={match_blend:.4f}/mismatch={mismatch_blend:.4f}")
        if match_blend - mismatch_blend < MIN_RATIO_GAP:
            print(f"  [較正セーフガード] {label}: ブレンド後も差が僅少なため、今月は補正なし(1.0/1.0)にフォールバック")
            return 1.0, 1.0
        return match_blend, mismatch_blend

    trio_match_raw = calib_ratio("trio_agree", "trio_hit", "new_top_trio_prob", True)
    trio_mismatch_raw = calib_ratio("trio_agree", "trio_hit", "new_top_trio_prob", False)
    pair_match_raw = calib_ratio("pair_agree", "pair_hit", "new_top_pair_prob", True)
    pair_mismatch_raw = calib_ratio("pair_agree", "pair_hit", "new_top_pair_prob", False)
    trio_match_ratio, trio_mismatch_ratio = blend_ratio_pair(
        trio_match_raw, trio_mismatch_raw, PRIOR_TRIO_MATCH, PRIOR_TRIO_MISMATCH, "trio(3連複)")
    pair_match_ratio, pair_mismatch_ratio = blend_ratio_pair(
        pair_match_raw, pair_mismatch_raw, PRIOR_PAIR_MATCH, PRIOR_PAIR_MISMATCH, "pair(2連複)")

    calibration = {
        "trio_match_ratio": round(trio_match_ratio, 4),
        "trio_mismatch_ratio": round(trio_mismatch_ratio, 4),
        "pair_match_ratio": round(pair_match_ratio, 4),
        "pair_mismatch_ratio": round(pair_mismatch_ratio, 4),
        "trio_match_ratio_raw": round(trio_match_raw, 4),
        "trio_mismatch_ratio_raw": round(trio_mismatch_raw, 4),
        "pair_match_ratio_raw": round(pair_match_raw, 4),
        "pair_mismatch_ratio_raw": round(pair_mismatch_raw, 4),
        "n_valid_races": int(len(merged)),
        "trio_match_n": int(merged["trio_agree"].sum()),
        "pair_match_n": int(merged["pair_agree"].sum()),
    }
    print("KNN較正比率:", calibration)

    # --- 本番用KNN: train+valid(手に入る最新まで)で学習し直す ---
    wide_prod = wide[(wide["date"] >= str(train_start.date())) & (wide["date"] < str(valid_end.date()))]
    print(f"KNN本番モデル学習プール: {len(wide_prod)}レース")
    knn_prod = ApproxRaceKNNModel(k=KNN_K, weighted=False).fit(wide_prod)

    return knn_prod, calibration


def train_strength_model(df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    df = df.copy()
    df["jcd_cat"] = df["jcd"].astype("category")
    df["級別num"] = df["級別"].map(CLASS_MAP).fillna(0)
    for f in BASE_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    finish_map = {"01": 5, "02": 4, "03": 3, "04": 2, "05": 1, "06": 0}
    df["relevance"] = df["着順"].map(finish_map).fillna(0)
    df["race_id"] = df["date"] + "_" + df["jcd"] + "_" + df["r"]

    max_date = df["date_dt"].max()
    valid_end = max_date + pd.Timedelta(days=1)
    valid_start = valid_end - pd.DateOffset(years=VALID_YEARS)
    train_end = valid_start
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)

    train_df = df[(df["date_dt"] >= train_start) & (df["date_dt"] < train_end)].copy()
    valid_df = df[(df["date_dt"] >= valid_start) & (df["date_dt"] < valid_end)].copy()
    print(f"強さモデル学習期間: {train_start.date()} 〜 {train_end.date()} ({train_df['race_id'].nunique()}レース)")
    print(f"valid期間: {valid_start.date()} 〜 {valid_end.date()} ({valid_df['race_id'].nunique()}レース)")

    # 欠損値の穴埋めは学習期間(train_df)のみの中央値を使う(valid期間や将来分を
    # 含めた全体の中央値を使うとリークになる)。この中央値は本番の日次予測
    # (run_scenario_predictions.py)でも同じ基準を使うようartifactsに保存する。
    medians = {f: float(train_df[f].median()) for f in BASE_FIELDS}
    for f in BASE_FIELDS:
        train_df[f] = train_df[f].fillna(medians[f])
        valid_df[f] = valid_df[f].fillna(medians[f])

    feature_cols = BASE_FIELDS + ["級別num", "boat_num", "jcd_cat"]

    def make_dataset(part):
        part = part.copy()
        part["boat_num"] = pd.to_numeric(part["boat_num"], errors="coerce")
        groups = part.groupby("race_id", sort=False).size().to_numpy()
        return lgb.Dataset(part[feature_cols], label=part["relevance"], group=groups,
                            categorical_feature=["jcd_cat"], free_raw_data=False)

    train_set = make_dataset(train_df)
    valid_set = make_dataset(valid_df)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3],
              "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "verbose": -1}
    model = lgb.train(params, train_set, num_boost_round=2000, valid_sets=[valid_set], valid_names=["valid"],
                       callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)])
    print("best_iteration:", model.best_iteration)
    return model, medians


def build_racer_style(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date_dt"].max()
    start = max_date - pd.DateOffset(years=STYLE_YEARS)
    recent = df[(df["date_dt"] >= start) & df["決まり手"].notna() & df["登番"].notna()]
    winner_k = recent[recent["着順"] == "01"]
    style = winner_k.groupby("登番")["決まり手"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    style["n_wins"] = winner_k.groupby("登番").size()
    style["まくり系率"] = style.get("まくり", 0) + style.get("まくり差し", 0)
    style["逃げ率"] = style.get("逃げ", 0)
    return style


SUBPLACE_MIN_NONWINS = 50


def build_subplace_profile(df: pd.DataFrame) -> pd.DataFrame:
    """各選手の「非1着時の2着率・3着率」(全期間、シナリオ1・2の2着/3着候補選定の
    根拠に使う)を計算する。"""
    non_win = df[(df["着順"] != "01") & df["登番"].notna()].copy()
    non_win["is_2nd"] = (non_win["着順"] == "02").astype(int)
    non_win["is_3rd"] = (non_win["着順"] == "03").astype(int)

    profile = non_win.groupby("登番").agg(
        n_non_wins=("着順", "size"), n_2nd=("is_2nd", "sum"), n_3rd=("is_3rd", "sum"))
    profile["非1着時2着率"] = (profile["n_2nd"] / profile["n_non_wins"] * 100).round(1)
    profile["非1着時3着率"] = (profile["n_3rd"] / profile["n_non_wins"] * 100).round(1)
    profile = profile[profile["n_non_wins"] >= SUBPLACE_MIN_NONWINS]
    return profile


def build_similar_race_table(df: pd.DataFrame) -> pd.DataFrame:
    """(jcd, 1号艇強さ帯, 荒れ度)ごとの決まり手%・3連単配当分布を集計する。"""
    max_date = df["date_dt"].max()
    start = max_date - pd.DateOffset(years=SIMILAR_RACE_YEARS)
    recent = df[(df["date_dt"] >= start) & df["決まり手"].notna()].copy()

    boat1 = recent[recent["boat_num"] == "1"][["date", "jcd", "r", "全国勝率"]].rename(
        columns={"全国勝率": "boat1_全国勝率"})
    race_level = recent.drop_duplicates(subset=["date", "jcd", "r"])[
        ["date", "jcd", "r", "決まり手", "波高", "風速", "3連単_払戻"]]
    race_level = race_level.merge(boat1, on=["date", "jcd", "r"], how="left")

    race_level["波高"] = pd.to_numeric(race_level["波高"], errors="coerce")
    race_level["風速"] = pd.to_numeric(race_level["風速"], errors="coerce")
    race_level["boat1_全国勝率"] = pd.to_numeric(race_level["boat1_全国勝率"], errors="coerce")
    race_level["3連単_払戻"] = pd.to_numeric(race_level["3連単_払戻"], errors="coerce")

    race_level["b1_bin"] = race_level["boat1_全国勝率"].apply(
        lambda x: strength_bin(x) if pd.notna(x) else None)
    race_level["rough"] = race_level.apply(
        lambda row: rough_bin(row["波高"], row["風速"]) if pd.notna(row["波高"]) and pd.notna(row["風速"]) else None,
        axis=1)

    # 配当帯は「全体(3年分)を件数ベースで20等分した固定の区間」を基準にする。
    # 各グループ自身の分位点で区切ると常にどの区間も5%になってしまい
    # (定義上そうなるだけで)偏りが見えなくなるため、母集団側の区間は固定し、
    # 各グループの配当がその固定区間にどう分布するか(5%からのズレ)を見る。
    all_payout = race_level["3連単_払戻"].dropna()
    overall_edges = all_payout.quantile(np.linspace(0, 1, 21)).to_numpy().copy()
    overall_edges[0], overall_edges[-1] = 0.0, np.inf

    rows = []
    for (jcd, b1_bin, rough), g in race_level.groupby(["jcd", "b1_bin", "rough"]):
        if b1_bin is None or rough is None:
            continue
        kimarite_dist = g["決まり手"].value_counts(normalize=True).mul(100).round(1).to_dict()
        payout = g["3連単_払戻"].dropna()
        bucket_idx = np.searchsorted(overall_edges, payout, side="right") - 1
        bucket_idx = np.clip(bucket_idx, 0, 19)
        bucket_counts = pd.Series(bucket_idx).value_counts().reindex(range(20), fill_value=0).tolist()
        rows.append({
            "jcd": jcd, "strength_bin": b1_bin, "rough_bin": rough,
            "n": len(g), "kimarite_dist": kimarite_dist,
            "payout_bucket_counts": bucket_counts, "payout_n": len(payout),
        })
    result = pd.DataFrame(rows)
    result.attrs["overall_payout_edges"] = overall_edges.tolist()
    return result


def main() -> None:
    print("ローリングアーカイブ読み込み・決まり手シードとのマージ中...")
    archive = load_merged_archive()
    print(f"アーカイブ: {len(archive)}行")

    print("1年複勝率等のローリング特徴量を計算中...")
    df = compute_trailing_year_stats(archive)

    print("\n=== 強さモデル(3年+jcd)を再学習 ===")
    model, base_field_medians = train_strength_model(df)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ARTIFACTS_DIR / "strength_model.txt"))
    pd.to_pickle(base_field_medians, ARTIFACTS_DIR / "base_field_medians.pkl")

    print("\n=== 近似100レース(KNN)モデルを再学習・一致較正比率を再計算 ===")
    max_date = df["date_dt"].max()
    valid_end = max_date + pd.Timedelta(days=1)
    valid_start = valid_end - pd.DateOffset(years=VALID_YEARS)
    train_end = valid_start
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    knn_model, calibration = train_knn_and_calibration(
        df, model, base_field_medians, train_start, train_end, valid_start, valid_end)
    import pickle
    with open(ARTIFACTS_DIR / "knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    import json
    with open(ARTIFACTS_DIR / "knn_calibration_ratios.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)

    print("\n=== 選手スタイル(5年)を再構築 ===")
    style = build_racer_style(df)
    style.to_pickle(ARTIFACTS_DIR / "racer_style_5y.pkl")
    print(f"選手数: {len(style)}")

    print("\n=== 非1着時2着率・3着率プロファイルを再構築 ===")
    subplace = build_subplace_profile(df)
    subplace.to_pickle(ARTIFACTS_DIR / "subplace_profile.pkl")
    print(f"選手数: {len(subplace)}")

    print("\n=== 類似レース表(3年)を再構築 ===")
    similar = build_similar_race_table(df)
    similar.to_pickle(ARTIFACTS_DIR / "similar_race_3y.pkl")
    overall_edges = similar.attrs.get("overall_payout_edges")
    pd.to_pickle(overall_edges, ARTIFACTS_DIR / "payout_overall_edges_3y.pkl")
    print(f"グループ数: {len(similar)}")

    print("\n完了。artifacts/scenario/ に strength_model.txt, base_field_medians.pkl, "
          "knn_model.pkl, knn_calibration_ratios.json, "
          "racer_style_5y.pkl, subplace_profile.pkl, similar_race_3y.pkl, "
          "payout_overall_edges_3y.pkl を保存しました。")


if __name__ == "__main__":
    main()
