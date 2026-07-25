"""展開シナリオ(本命/対抗/他有力)の買い目生成ロジック。

boatrace-niren-wide-experimentフォルダで検証済みのscenario_lib.pyを本番用に移植。
変更点:
  - データ読み込み元をこのリポジトリの成果物(artifacts/scenario/配下、
    retrain_weekly.pyが週次で生成する)に変更。
  - シナリオ2(対抗艇)を2連単からbuild_ticketsを使った3連単形式に変更
    (1%未満チケット除外込み)。
  - シナリオ3(保険/他有力)の点数上限を6→4に変更。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "scenario"
RACER_STYLE_PATH = ARTIFACTS_DIR / "racer_style_5y.pkl"
SUBPLACE_PATH = ARTIFACTS_DIR / "subplace_profile.pkl"
SIMILAR_RACE_PATH = ARTIFACTS_DIR / "similar_race_3y.pkl"
PAYOUT_EDGES_PATH = ARTIFACTS_DIR / "payout_overall_edges_3y.pkl"

MAIN_MAX_TICKETS = 9
SUB_MAX_TICKETS = 6  # シナリオ2(対抗艇3連単)の点数上限
HEDGE_MAX_TICKETS = 4  # シナリオ3(他有力/保険)の点数上限
STYLE_MIN_WINS = 30
SUBPLACE_MIN_NONWINS = 50
MIN_TICKET_PROB = 0.01  # 1%未満のチケットは表示しない


def load_racer_style() -> pd.DataFrame:
    """選手個人のまくり系決着率・逃げ率(直近5年、retrain_weekly.py生成)を読み込む。"""
    return pd.read_pickle(RACER_STYLE_PATH)


def load_subplace_profile() -> pd.DataFrame:
    return pd.read_pickle(SUBPLACE_PATH)


def load_similar_race_table() -> pd.DataFrame:
    """類似レース分析(会場×強さ帯×荒れ度、直近3年、retrain_weekly.py生成)を読み込む。"""
    return pd.read_pickle(SIMILAR_RACE_PATH)


def load_payout_overall_edges() -> list[float]:
    """配当帯の母集団側の固定区間(直近3年全体を件数ベースで20等分した21個の境界値)を読み込む。"""
    return pd.read_pickle(PAYOUT_EDGES_PATH)


def format_formation(tickets: list[tuple]) -> str:
    return "、".join("-".join(map(str, combo)) + f"({p*100:.1f}%)" for combo, p in tickets)


def rank_candidates(prefix: tuple[int, ...], trow: pd.Series) -> list[tuple[int, float]]:
    """prefixで固定した着順の次の位置について、残り艇ごとの確率を降順で返す。
    prefix=(winner,) -> 2着候補(3着は問わず合算)。prefix=(winner,second) -> 3着候補(そのまま)。
    """
    remaining = [n for n in (1, 2, 3, 4, 5, 6) if n not in prefix]
    probs = []
    if len(prefix) == 2:
        for c in remaining:
            key = f"prob_{prefix[0]}-{prefix[1]}-{c}"
            probs.append((c, trow.get(key, 0.0)))
    else:
        for b in remaining:
            total = sum(trow.get(f"prob_{prefix[0]}-{b}-{c}", 0.0) for c in remaining if c != b)
            probs.append((b, total))
    probs.sort(key=lambda x: -x[1])
    return probs


def select_candidates(ranked: list[tuple[int, float]], solo_threshold: float = 0.40,
                       pair_threshold: float = 0.65) -> list[int] | None:
    """確率の集中度に応じて1艇/2艇に絞るか、Noneで「絞らず全艇」を返す。
    - 最有力1艇だけでsolo_threshold(既定40%)以上を占めるなら1艇に絞る
    - 上位2艇の合計がpair_threshold(既定65%)以上なら2艇に絞る
    - それでも集中しなければ絞らない
    """
    total = sum(p for _, p in ranked)
    if total <= 0:
        return None
    if ranked[0][1] / total >= solo_threshold:
        return [ranked[0][0]]
    if len(ranked) >= 2 and (ranked[0][1] + ranked[1][1]) / total >= pair_threshold:
        return [ranked[0][0], ranked[1][0]]
    return None


def build_tickets(winner: int, trow: pd.Series, max_tickets: int) -> dict:
    """1着=winner固定で、2着→3着の順に集中度に応じた候補選定を行い、
    物語(1着=winner・2着候補)に沿ったチケットのみを作る(穴埋めはしない)。
    3着は絞り込むと的中率が落ちることが検証済みのため絞らない。
    確率がMIN_TICKET_PROB(既定1%)未満のチケットは表示しない。
    """
    ranked2 = rank_candidates((winner,), trow)
    cand2 = select_candidates(ranked2)
    pos2_list = cand2 if cand2 is not None else [b for b, _ in ranked2]

    tickets = []
    seen = set()
    third_info = {}
    for b in pos2_list:
        ranked3 = rank_candidates((winner, b), trow)
        cand3 = select_candidates(ranked3)
        third_info[b] = {"mode": "narrow" if cand3 is not None else "spread", "ranked": ranked3, "candidates": cand3}
        for c, _ in ranked3:
            prob = trow.get(f"prob_{winner}-{b}-{c}", 0.0)
            if prob < MIN_TICKET_PROB:
                continue
            if (winner, b, c) not in seen:
                seen.add((winner, b, c))
                tickets.append(((winner, b, c), prob))

    tickets.sort(key=lambda x: -x[1])
    tickets = tickets[:max_tickets]

    return {
        "tickets": tickets,
        "second": {"mode": "narrow" if cand2 is not None else "spread", "ranked": ranked2, "candidates": cand2},
        "third_by_second": third_info,
    }


def build_hedge_tickets(trow: pd.Series, exclude_triples: set[tuple], exclude_exacta: set[tuple],
                         max_tickets: int) -> list[tuple]:
    """シナリオ1(物語)にもシナリオ2(対抗艇の1・2着ペア)にも該当しない保険チケットを、
    レース全体の真の確率上位から作る。確率MIN_TICKET_PROB未満は含めない。
    """
    prob_cols = [c for c in trow.index if c.startswith("prob_")]
    all_ranked = trow[prob_cols].sort_values(ascending=False)
    tickets = []
    for label, prob in all_ranked.items():
        if len(tickets) >= max_tickets or prob < MIN_TICKET_PROB:
            break
        triple = tuple(int(x) for x in label.replace("prob_", "").split("-"))
        if triple in exclude_triples:
            continue
        if (triple[0], triple[1]) in exclude_exacta:
            continue
        tickets.append((triple, prob))
    return tickets


def favorite_boat_score(win_prob: np.ndarray, style_df: pd.DataFrame, toubans: dict, weighting: str = "multiply") -> dict:
    """全6艇についてスコアを返す(1着確率×まくり系率)。1号艇に限定しない。
    実績不足艇は全国平均のまくり系率(0.30)で仮置き。
    """
    scores = {}
    for n in range(1, 7):
        touban = toubans[n]
        has_style = touban in style_df.index and style_df.loc[touban, "n_wins"] >= STYLE_MIN_WINS
        style_val = (style_df.loc[touban, "まくり系率"] / 100) if has_style else 0.30
        wp = win_prob[n - 1]
        if weighting == "win_prob":
            scores[n] = wp
        elif weighting == "style":
            scores[n] = style_val
        else:
            scores[n] = wp * style_val
    return scores


def place_reasoning(ranked: list[tuple[int, float]], candidates: list[int] | None, win_prob: np.ndarray,
                     toubans: dict, subplace_df: pd.DataFrame, subplace_col: str) -> dict:
    def boat_info(boat: int, prob: float) -> dict:
        touban = toubans[boat]
        if touban in subplace_df.index and subplace_df.loc[touban, "n_non_wins"] >= SUBPLACE_MIN_NONWINS:
            rate = subplace_df.loc[touban, subplace_col]
            n_nonwins = int(subplace_df.loc[touban, "n_non_wins"])
        else:
            rate, n_nonwins = None, 0
        return {"boat": boat, "prob": prob, "own_win_prob": win_prob[boat - 1], "past_rate": rate, "n_non_wins": n_nonwins}

    prob_by_boat = dict(ranked)
    if candidates is None:
        return {"mode": "spread", "candidates": [boat_info(b, prob_by_boat[b]) for b, _ in ranked]}
    return {"mode": "narrow", "candidates": [boat_info(b, prob_by_boat[b]) for b in candidates]}


def racer_nige_rate(touban: str, style_df: pd.DataFrame) -> float | None:
    """選手の逃げ率(参考表示専用、買い目には影響しない)。実績不足ならNone。"""
    if touban in style_df.index and style_df.loc[touban, "n_wins"] >= STYLE_MIN_WINS:
        return float(style_df.loc[touban, "逃げ率"]) if "逃げ率" in style_df.columns else None
    return None


def generate_scenarios(win_prob: np.ndarray, trow: pd.Series, toubans: dict, style_df: pd.DataFrame,
                        subplace_df: pd.DataFrame, weighting: str = "multiply") -> dict:
    """シナリオ1(本命艇が勝つ、3連単・物語のみ)・シナリオ2(対抗艇が勝つ、3連単)・
    シナリオ3(1にも2にも該当しない保険、3連単、上位4点)の買い目を生成する。

    シナリオ1: 物語(1着=本命・2着候補)に沿ったチケットのみ。1%未満は含めない。
               本命艇の逃げ率を参考表示として付与(買い目には影響しない)。
    シナリオ2: 対抗艇(1着確率×まくり系率が最大の艇、本命艇を除く)が勝つ3連単。
               対抗艇のまくり系率を根拠として使用・表示する。
    シナリオ3: シナリオ1・2のどちらにも当てはまらない、レース全体の確率上位の保険(上位4点)。
    """
    favorite = int(np.argmax(win_prob)) + 1
    result1 = build_tickets(favorite, trow, MAIN_MAX_TICKETS)
    reasoning1_2nd = place_reasoning(result1["second"]["ranked"], result1["second"]["candidates"],
                                      win_prob, toubans, subplace_df, "非1着時2着率")
    reasoning1_3rd = {
        boat2nd: place_reasoning(info["ranked"], info["candidates"], win_prob, toubans, subplace_df, "非1着時3着率")
        for boat2nd, info in result1["third_by_second"].items()
    }
    scenario1 = {
        "winner_boat": favorite, "tickets": result1["tickets"],
        "second_place": reasoning1_2nd, "third_place_by_second": reasoning1_3rd,
        "nige_rate": racer_nige_rate(toubans[favorite], style_df),
    }

    scores = favorite_boat_score(win_prob, style_df, toubans, weighting)
    challenger = max((n for n in scores if n != favorite), key=scores.get)
    result2 = build_tickets(challenger, trow, SUB_MAX_TICKETS)
    reasoning2_2nd = place_reasoning(result2["second"]["ranked"], result2["second"]["candidates"],
                                      win_prob, toubans, subplace_df, "非1着時2着率")
    touban_c = toubans[challenger]
    makuri_rate = (style_df.loc[touban_c, "まくり系率"]
                   if touban_c in style_df.index and style_df.loc[touban_c, "n_wins"] >= STYLE_MIN_WINS else None)
    scenario2 = {
        "winner_boat": challenger, "tickets": result2["tickets"],
        "second_place": reasoning2_2nd, "scores": scores, "bet_type": "3連単",
        "makuri_rate": makuri_rate,
    }

    exclude_triples = {t for t, _ in scenario1["tickets"]} | {t for t, _ in scenario2["tickets"]}
    exclude_exacta = {(t[0], t[1]) for t, _ in scenario2["tickets"]}
    hedge_tickets = build_hedge_tickets(trow, exclude_triples, exclude_exacta, HEDGE_MAX_TICKETS)
    scenario3 = {"tickets": hedge_tickets, "bet_type": "3連単"}

    return {"scenario1": scenario1, "scenario2": scenario2, "scenario3": scenario3}
