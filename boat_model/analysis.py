from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .evaluate import EvaluationResult, safe_model_column_name
from .features import actual_trifecta_strings


TARGET_ANALYSIS_MODELS = [
    "C_lightgbm_direct120",
    "C_lightgbm_position6",
    "E_blend_B_knn_k500_count_pairwise_selected_wknn_0.5",
    "B_knn_k300_count",
    "D_pairwise_recent_no_master",
]

AGREEMENT_MODELS = [
    "C_lightgbm_direct120",
    "C_lightgbm_position6",
    "E_blend_B_knn_k500_count_pairwise_selected_wknn_0.5",
]

AGREEMENT_CONDITIONS = {
    "2plus_top5": "candidate_tickets_2plus",
    "all3_top5": "candidate_tickets_all3",
    "union_top5": "candidate_tickets_union",
}


def _parse_top5(top5_text: object) -> list[str]:
    if pd.isna(top5_text):
        return []
    values = []
    for part in str(top5_text).split("|"):
        if not part:
            continue
        values.append(part.split(":", 1)[0])
    return values


def _parse_top5_with_scores(top5_text: object) -> list[tuple[str, float, int]]:
    if pd.isna(top5_text):
        return []
    values = []
    for rank, part in enumerate(str(top5_text).split("|"), start=1):
        if not part:
            continue
        ticket, _, prob_text = part.partition(":")
        try:
            probability = float(prob_text) if prob_text else 0.0
        except ValueError:
            probability = 0.0
        values.append((ticket, probability, rank))
    return values


def _safe_qcut(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q=q, labels=labels)
    except ValueError:
        return pd.cut(ranked, bins=q, labels=labels, include_lowest=True)


def _pattern_inversion_rate(pattern: object) -> float:
    try:
        values = [int(x) for x in str(pattern).split("-")]
    except ValueError:
        return 0.0
    inversions = 0
    total = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            if values[i] > values[j]:
                inversions += 1
    return inversions / total if total else 0.0


def add_analysis_segments(test_df: pd.DataFrame) -> pd.DataFrame:
    """分析用に荒れ指数と払戻金帯を付ける。ターゲット結果は荒れ指数に使わない。"""
    out = test_df.copy()

    if "勝率1_minus_勝率2" not in out.columns:
        out["勝率1_minus_勝率2"] = out["勝率1"] - out["勝率2"]
    if "勝率1_minus_勝率3" not in out.columns:
        out["勝率1_minus_勝率3"] = out["勝率1"] - out["勝率3"]

    if "勝率順位パターン" in out.columns:
        pattern_inversion = out["勝率順位パターン"].map(_pattern_inversion_rate)
    else:
        pattern_inversion = pd.Series(0.0, index=out.index)

    components = pd.DataFrame(
        {
            "boat1_vs_2_weak": (-pd.to_numeric(out["勝率1_minus_勝率2"], errors="coerce")).rank(pct=True),
            "boat1_vs_3_weak": (-pd.to_numeric(out["勝率1_minus_勝率3"], errors="coerce")).rank(pct=True),
            "winrate_close": (-pd.to_numeric(out["勝率標準偏差"], errors="coerce")).rank(pct=True),
            "boat1_not_top": (1 - pd.to_numeric(out["1号艇が勝率1位か"], errors="coerce")).rank(pct=True),
            "pattern_disorder": pd.to_numeric(pattern_inversion, errors="coerce").rank(pct=True),
        }
    ).fillna(0.5)
    out["roughness_score"] = components.mean(axis=1)
    out["roughness_bin"] = _safe_qcut(
        out["roughness_score"],
        5,
        [
            "Q1_low_roughness",
            "Q2",
            "Q3",
            "Q4",
            "Q5_high_roughness",
        ],
    )

    payout = pd.to_numeric(out["3rt"], errors="coerce")
    out["payout_band"] = pd.cut(
        payout,
        bins=[-np.inf, 999, 2999, 9999, np.inf],
        labels=["under_1000", "1000_2999", "3000_9999", "10000_plus"],
        right=True,
    )
    return out


def _summary_by_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_columns, keys)}
        row.update(
            {
                "model": group["model"].iloc[0],
                "n_races": int(len(group)),
                "first_place_accuracy": float(group["first_place_hit"].mean()),
                "trifecta_top1_accuracy": float(group["trifecta_top1_hit"].mean()),
                "top3_contains_actual": float((group["actual_rank"] <= 3).mean()),
                "top5_contains_actual": float((group["actual_rank"] <= 5).mean()),
                "top10_contains_actual": float((group["actual_rank"] <= 10).mean()),
                "mean_actual_rank": float(group["actual_rank"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _details_from_results(
    test_df: pd.DataFrame,
    results: list[EvaluationResult],
    target_models: Iterable[str],
) -> pd.DataFrame:
    segmented = add_analysis_segments(test_df).reset_index(drop=True)
    target_set = set(target_models)
    frames = []
    for result in results:
        model = str(result.metrics["model"])
        if model not in target_set:
            continue
        detail = result.details.reset_index(drop=True).copy()
        detail["model"] = model
        detail["roughness_score"] = segmented["roughness_score"]
        detail["roughness_bin"] = segmented["roughness_bin"]
        detail["payout_band"] = segmented["payout_band"]
        frames.append(detail)
    if not frames:
        raise ValueError("No target models were found in evaluation results.")
    return pd.concat(frames, ignore_index=True)


def _details_from_predictions(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    target_models: Iterable[str],
) -> pd.DataFrame:
    segmented = add_analysis_segments(test_df).reset_index(drop=True)
    frames = []
    for model in target_models:
        prefix = safe_model_column_name(model)
        required = [
            f"{prefix}_actual_rank",
            f"{prefix}_first_place_hit",
            f"{prefix}_trifecta_top1_hit",
            f"{prefix}_top1_prediction",
            f"{prefix}_top5_predictions",
        ]
        missing = [col for col in required if col not in predictions.columns]
        if missing:
            raise ValueError(f"{model}: missing prediction columns: {missing}")
        detail = pd.DataFrame(
            {
                "jcd": predictions["jcd"].astype(str).str.zfill(2),
                "r": pd.to_numeric(predictions["r"], errors="coerce").astype(int),
                "actual_result": predictions["actual_result"],
                "top1_prediction": predictions[f"{prefix}_top1_prediction"],
                "top5_predictions": predictions[f"{prefix}_top5_predictions"],
                "actual_rank": pd.to_numeric(predictions[f"{prefix}_actual_rank"], errors="coerce"),
                "first_place_hit": predictions[f"{prefix}_first_place_hit"].astype(bool),
                "trifecta_top1_hit": predictions[f"{prefix}_trifecta_top1_hit"].astype(bool),
                "model": model,
                "roughness_score": segmented["roughness_score"],
                "roughness_bin": segmented["roughness_bin"],
                "payout_band": segmented["payout_band"],
            }
        )
        frames.append(detail)
    return pd.concat(frames, ignore_index=True)


def _write_group_analyses(details: pd.DataFrame, output_dir: Path) -> None:
    by_jcd = _summary_by_group(details, ["model", "jcd"]).sort_values(["model", "jcd"])
    by_jcd.to_csv(output_dir / "analysis_by_jcd.csv", index=False, encoding="utf-8-sig")

    by_race_no = _summary_by_group(details, ["model", "r"]).sort_values(["model", "r"])
    by_race_no.to_csv(output_dir / "analysis_by_race_no.csv", index=False, encoding="utf-8-sig")

    by_roughness = _summary_by_group(details, ["model", "roughness_bin"]).sort_values(
        ["model", "roughness_bin"]
    )
    by_roughness.to_csv(output_dir / "analysis_by_roughness.csv", index=False, encoding="utf-8-sig")

    by_payout = _summary_by_group(details, ["model", "payout_band"]).sort_values(
        ["model", "payout_band"]
    )
    payout_order = {
        "under_1000": 1,
        "1000_2999": 2,
        "3000_9999": 3,
        "10000_plus": 4,
    }
    by_payout["_payout_order"] = by_payout["payout_band"].astype(str).map(payout_order).fillna(99)
    by_payout = by_payout.sort_values(["model", "_payout_order"]).drop(columns=["_payout_order"])
    by_payout.to_csv(output_dir / "analysis_by_payout_band.csv", index=False, encoding="utf-8-sig")


def _agreement_rows_from_predictions(
    predictions: pd.DataFrame,
    agreement_models: Iterable[str],
) -> list[dict[str, object]]:
    models = list(agreement_models)
    prefixes = {model: safe_model_column_name(model) for model in models}
    top5_lists = {
        model: predictions[f"{prefixes[model]}_top5_predictions"].map(_parse_top5)
        for model in models
    }
    actual = predictions["actual_result"].astype(str)

    def summarize(rule_name: str, candidates_by_row: list[set[str]]) -> dict[str, object]:
        counts = np.asarray([len(candidates) for candidates in candidates_by_row], dtype=float)
        hits = np.asarray(
            [actual.iloc[i] in candidates for i, candidates in enumerate(candidates_by_row)],
            dtype=bool,
        )
        nonempty = counts > 0
        return {
            "agreement_rule": rule_name,
            "models": " | ".join(models),
            "n_races": int(len(predictions)),
            "races_with_candidates": int(nonempty.sum()),
            "avg_candidate_count": float(counts.mean()),
            "median_candidate_count": float(np.median(counts)),
            "max_candidate_count": int(counts.max()) if len(counts) else 0,
            "actual_in_candidates_rate_all_races": float(hits.mean()),
            "actual_in_candidates_rate_when_nonempty": float(hits[nonempty].mean()) if nonempty.any() else 0.0,
            "hits": int(hits.sum()),
            "total_candidates": int(counts.sum()),
            "hits_per_candidate": float(hits.sum() / counts.sum()) if counts.sum() else 0.0,
        }

    sets_by_model = {
        model: [set(values) for values in top5_lists[model]]
        for model in models
    }
    rows = []

    for left_idx in range(len(models)):
        for right_idx in range(left_idx + 1, len(models)):
            left = models[left_idx]
            right = models[right_idx]
            candidates = [
                sets_by_model[left][i] & sets_by_model[right][i]
                for i in range(len(predictions))
            ]
            rows.append(summarize(f"intersection_{left}_AND_{right}", candidates))

    at_least_2 = []
    all_3 = []
    union = []
    for i in range(len(predictions)):
        counts: dict[str, int] = {}
        for model in models:
            for ticket in sets_by_model[model][i]:
                counts[ticket] = counts.get(ticket, 0) + 1
        at_least_2.append({ticket for ticket, count in counts.items() if count >= 2})
        all_3.append({ticket for ticket, count in counts.items() if count == len(models)})
        union.append(set(counts))

    rows.append(summarize("tickets_appearing_in_at_least_2_models", at_least_2))
    rows.append(summarize("tickets_appearing_in_all_3_models", all_3))
    rows.append(summarize("union_of_all_top5_tickets", union))
    return rows


def _race_no_bin(race_no: object) -> str:
    race_no_int = int(race_no)
    if 1 <= race_no_int <= 6:
        return "early"
    if 7 <= race_no_int <= 9:
        return "middle"
    if 10 <= race_no_int <= 12:
        return "late"
    return "unknown"


def _ordered_candidate_sets(row: pd.Series, models: list[str]) -> dict[str, list[str]]:
    counts: dict[str, int] = {}
    score_sum: dict[str, float] = {}
    rank_sum: dict[str, int] = {}

    for model in models:
        prefix = safe_model_column_name(model)
        for ticket, probability, rank in _parse_top5_with_scores(row[f"{prefix}_top5_predictions"]):
            counts[ticket] = counts.get(ticket, 0) + 1
            score_sum[ticket] = score_sum.get(ticket, 0.0) + probability
            rank_sum[ticket] = rank_sum.get(ticket, 0) + rank

    def sort_key(ticket: str) -> tuple[float, float, float, str]:
        count = counts[ticket]
        avg_rank = rank_sum[ticket] / count
        return (-count, -score_sum[ticket], avg_rank, ticket)

    union = sorted(counts.keys(), key=sort_key)
    two_plus = [ticket for ticket in union if counts[ticket] >= 2]
    all_three = [ticket for ticket in union if counts[ticket] == len(models)]
    return {
        "candidate_tickets_2plus": two_plus,
        "candidate_tickets_all3": all_three,
        "candidate_tickets_union": union,
    }


def build_buy_grade_predictions(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    agreement_models: Iterable[str] = AGREEMENT_MODELS,
) -> pd.DataFrame:
    """事前に使える荒れ指数・レース番号・モデル一致から買いグレードを作る。"""
    models = list(agreement_models)
    missing = []
    for model in models:
        prefix = safe_model_column_name(model)
        col = f"{prefix}_top5_predictions"
        if col not in predictions.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing columns for buy grade analysis: {missing}")

    segmented = add_analysis_segments(test_df).reset_index(drop=True)
    rows = []
    for idx, pred_row in predictions.reset_index(drop=True).iterrows():
        candidates = _ordered_candidate_sets(pred_row, models)
        actual = str(pred_row["actual_result"])
        roughness_bin = str(segmented.loc[idx, "roughness_bin"])
        race_no = int(pred_row["r"])
        has_2plus = len(candidates["candidate_tickets_2plus"]) > 0
        has_all3 = len(candidates["candidate_tickets_all3"]) > 0
        low_roughness_2 = roughness_bin in {"Q1_low_roughness", "Q2"}
        low_roughness_3 = roughness_bin in {"Q1_low_roughness", "Q2", "Q3"}
        middle_or_late = 7 <= race_no <= 12

        if low_roughness_2 and middle_or_late and has_2plus:
            buy_grade = "A"
        elif low_roughness_3 and middle_or_late and has_2plus:
            buy_grade = "B"
        elif low_roughness_3 and has_all3:
            buy_grade = "C"
        else:
            buy_grade = "D"

        row = {
            "jcd": str(pred_row["jcd"]).zfill(2),
            "r": race_no,
            "race_no_bin": _race_no_bin(race_no),
            "roughness_bin": roughness_bin,
            "buy_grade": buy_grade,
            "actual_result": actual,
            "3rt": pd.to_numeric(segmented.loc[idx, "3rt"], errors="coerce"),
        }
        for suffix, candidate_col in [
            ("2plus", "candidate_tickets_2plus"),
            ("all3", "candidate_tickets_all3"),
            ("union", "candidate_tickets_union"),
        ]:
            values = candidates[candidate_col]
            row[candidate_col] = "|".join(values)
            row[f"actual_in_{suffix}"] = actual in values
            row[f"candidate_count_{suffix}"] = len(values)
        rows.append(row)

    full = pd.DataFrame(rows)
    return full[
        [
            "jcd",
            "r",
            "roughness_bin",
            "buy_grade",
            "actual_result",
            "3rt",
            "candidate_tickets_2plus",
            "candidate_tickets_all3",
            "candidate_tickets_union",
            "actual_in_2plus",
            "actual_in_all3",
            "actual_in_union",
            "candidate_count_2plus",
            "candidate_count_all3",
            "candidate_count_union",
            "race_no_bin",
        ]
    ]


def _candidate_list_from_text(value: object) -> list[str]:
    if pd.isna(value) or str(value) == "":
        return []
    return [ticket for ticket in str(value).split("|") if ticket]


def _candidate_metrics(frame: pd.DataFrame, candidate_col: str) -> dict[str, object]:
    candidate_lists = frame[candidate_col].map(_candidate_list_from_text)
    actual = frame["actual_result"].astype(str)
    candidate_counts = candidate_lists.map(len).to_numpy(dtype=float)
    hits = np.asarray(
        [actual.iloc[idx] in candidates for idx, candidates in enumerate(candidate_lists)],
        dtype=bool,
    )
    actual_first = actual.str.split("-", expand=True)[0]
    top_candidate_first = candidate_lists.map(lambda values: values[0].split("-")[0] if values else "")
    first_place_hit = (top_candidate_first.to_numpy() == actual_first.to_numpy()) & (candidate_counts > 0)

    top5_hits = np.asarray(
        [actual.iloc[idx] in candidates[:5] for idx, candidates in enumerate(candidate_lists)],
        dtype=bool,
    )
    top10_hits = np.asarray(
        [actual.iloc[idx] in candidates[:10] for idx, candidates in enumerate(candidate_lists)],
        dtype=bool,
    )
    ranks = []
    missing_rank = 16
    for idx, candidates in enumerate(candidate_lists):
        try:
            ranks.append(candidates.index(actual.iloc[idx]) + 1)
        except ValueError:
            ranks.append(missing_rank)

    total_candidates = float(candidate_counts.sum())
    return {
        "n_races": int(len(frame)),
        "avg_candidate_count": float(candidate_counts.mean()) if len(frame) else 0.0,
        "actual_in_candidates_rate": float(hits.mean()) if len(frame) else 0.0,
        "hits_per_candidate": float(hits.sum() / total_candidates) if total_candidates else 0.0,
        "first_place_accuracy": float(first_place_hit.mean()) if len(frame) else 0.0,
        "top5_contains_actual": float(top5_hits.mean()) if len(frame) else 0.0,
        "top10_contains_actual": float(top10_hits.mean()) if len(frame) else 0.0,
        "mean_actual_rank": float(np.mean(ranks)) if ranks else 0.0,
    }


def _flat_roi_metrics(
    frame: pd.DataFrame,
    *,
    candidate_col: str,
    rule_label: str,
    grade_label: str,
    stake_per_ticket: int = 100,
) -> dict[str, object]:
    candidate_lists = frame[candidate_col].map(_candidate_list_from_text)
    candidate_counts = candidate_lists.map(len).to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    payouts = pd.to_numeric(frame["3rt"], errors="coerce").fillna(0).to_numpy(dtype=float)

    hits = np.asarray(
        [actual[idx] in candidates for idx, candidates in enumerate(candidate_lists)],
        dtype=bool,
    )
    total_bet_amount = float((candidate_counts * stake_per_ticket).sum())
    returns = np.where(hits, payouts * (stake_per_ticket / 100.0), 0.0)
    hit_payouts = payouts[hits]

    return {
        "buy_grade": grade_label,
        "candidate_rule": rule_label,
        "n_races": int(len(frame)),
        "avg_candidate_count": float(candidate_counts.mean()) if len(frame) else 0.0,
        "total_bet_amount": int(total_bet_amount),
        "hit_count": int(hits.sum()),
        "hit_rate_per_race": float(hits.mean()) if len(frame) else 0.0,
        "hit_rate_per_ticket": float(hits.sum() / candidate_counts.sum()) if candidate_counts.sum() else 0.0,
        "total_return": int(returns.sum()),
        "roi": float(returns.sum() / total_bet_amount) if total_bet_amount else 0.0,
        "average_hit_payout": float(hit_payouts.mean()) if len(hit_payouts) else 0.0,
        "median_hit_payout": float(np.median(hit_payouts)) if len(hit_payouts) else 0.0,
        "max_hit_payout": int(hit_payouts.max()) if len(hit_payouts) else 0,
    }


def _weighted_strategy_metrics(
    frame: pd.DataFrame,
    *,
    strategy_name: str,
    ticket_stakes_by_row: list[dict[str, int]],
) -> dict[str, object]:
    actual = frame["actual_result"].astype(str).to_numpy()
    payouts = pd.to_numeric(frame["3rt"], errors="coerce").fillna(0).to_numpy(dtype=float)
    candidate_counts = np.asarray([len(stakes) for stakes in ticket_stakes_by_row], dtype=float)
    total_bets = np.asarray([sum(stakes.values()) for stakes in ticket_stakes_by_row], dtype=float)
    hit_stakes = np.asarray(
        [stakes.get(actual[idx], 0) for idx, stakes in enumerate(ticket_stakes_by_row)],
        dtype=float,
    )
    hits = hit_stakes > 0
    returns = payouts * (hit_stakes / 100.0)
    total_bet_amount = float(total_bets.sum())
    hit_payouts = payouts[hits]

    return {
        "strategy": strategy_name,
        "buy_grade": "A",
        "n_races": int(len(frame)),
        "avg_candidate_count": float(candidate_counts.mean()) if len(frame) else 0.0,
        "total_bet_amount": int(total_bet_amount),
        "hit_count": int(hits.sum()),
        "hit_rate_per_race": float(hits.mean()) if len(frame) else 0.0,
        "hit_rate_per_ticket": float(hits.sum() / candidate_counts.sum()) if candidate_counts.sum() else 0.0,
        "total_return": int(returns.sum()),
        "roi": float(returns.sum() / total_bet_amount) if total_bet_amount else 0.0,
        "average_hit_payout": float(hit_payouts.mean()) if len(hit_payouts) else 0.0,
        "median_hit_payout": float(np.median(hit_payouts)) if len(hit_payouts) else 0.0,
        "max_hit_payout": int(hit_payouts.max()) if len(hit_payouts) else 0,
    }


def write_buy_grade_roi_analysis(
    buy_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_specs = [
        ("A", "2plus", "candidate_tickets_2plus"),
        ("A", "all3", "candidate_tickets_all3"),
        ("A", "union", "candidate_tickets_union"),
        ("B", "2plus", "candidate_tickets_2plus"),
        ("C", "all3", "candidate_tickets_all3"),
        ("D", "union", "candidate_tickets_union"),
    ]
    roi_rows = []
    for grade, rule_label, candidate_col in roi_specs:
        frame = buy_predictions.loc[buy_predictions["buy_grade"] == grade].copy()
        roi_rows.append(
            _flat_roi_metrics(
                frame,
                candidate_col=candidate_col,
                rule_label=rule_label,
                grade_label=grade,
                stake_per_ticket=100,
            )
        )
    roi_analysis = pd.DataFrame(roi_rows)
    roi_analysis.to_csv(output_dir / "buy_grade_roi_analysis.csv", index=False, encoding="utf-8-sig")

    grade_a = buy_predictions.loc[buy_predictions["buy_grade"] == "A"].copy()
    two_plus_lists = grade_a["candidate_tickets_2plus"].map(_candidate_list_from_text).tolist()
    all3_lists = grade_a["candidate_tickets_all3"].map(_candidate_list_from_text).tolist()
    union_lists = grade_a["candidate_tickets_union"].map(_candidate_list_from_text).tolist()

    def fixed_stakes(candidate_lists: list[list[str]], stake: int) -> list[dict[str, int]]:
        return [{ticket: stake for ticket in candidates} for candidates in candidate_lists]

    mixed_stakes = []
    for two_plus, all3 in zip(two_plus_lists, all3_lists):
        stakes = {ticket: 100 for ticket in two_plus}
        for ticket in all3:
            stakes[ticket] = 200
        mixed_stakes.append(stakes)

    strategy_rows = [
        _weighted_strategy_metrics(
            grade_a,
            strategy_name="grade_A_2plus_all_100yen",
            ticket_stakes_by_row=fixed_stakes(two_plus_lists, 100),
        ),
        _weighted_strategy_metrics(
            grade_a,
            strategy_name="grade_A_all3_only_100yen",
            ticket_stakes_by_row=fixed_stakes(all3_lists, 100),
        ),
        _weighted_strategy_metrics(
            grade_a,
            strategy_name="grade_A_all3_200yen_2plus_only_100yen",
            ticket_stakes_by_row=mixed_stakes,
        ),
        _weighted_strategy_metrics(
            grade_a,
            strategy_name="grade_A_union_all_100yen",
            ticket_stakes_by_row=fixed_stakes(union_lists, 100),
        ),
    ]
    strategy_analysis = pd.DataFrame(strategy_rows)
    strategy_analysis.to_csv(output_dir / "bet_strategy_analysis.csv", index=False, encoding="utf-8-sig")

    return {
        "buy_grade_roi_analysis": roi_analysis,
        "bet_strategy_analysis": strategy_analysis,
    }


def _candidate_type_lists(row: pd.Series) -> dict[str, list[str]]:
    all3 = _candidate_list_from_text(row["candidate_tickets_all3"])
    two_plus = _candidate_list_from_text(row["candidate_tickets_2plus"])
    union = _candidate_list_from_text(row["candidate_tickets_union"])
    all3_set = set(all3)
    two_plus_set = set(two_plus)

    return {
        "all3": all3,
        "2plus_only": [ticket for ticket in two_plus if ticket not in all3_set],
        "union_only": [ticket for ticket in union if ticket not in two_plus_set],
    }


def build_candidate_type_predictions(buy_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in buy_predictions.iterrows():
        typed_lists = _candidate_type_lists(row)
        actual = str(row["actual_result"])
        for candidate_type, candidates in typed_lists.items():
            rows.append(
                {
                    "jcd": str(row["jcd"]).zfill(2),
                    "r": int(row["r"]),
                    "race_group": _race_no_bin(row["r"]),
                    "roughness_bin": row["roughness_bin"],
                    "buy_grade": row["buy_grade"],
                    "candidate_type": candidate_type,
                    "actual_result": actual,
                    "3rt": pd.to_numeric(row["3rt"], errors="coerce"),
                    "candidate_tickets": "|".join(candidates),
                    "candidate_count": len(candidates),
                    "actual_in_candidates": actual in candidates,
                    "exploration_note": "June 2026 exploratory analysis; do not treat as final adopted rule",
                }
            )
    return pd.DataFrame(rows)


def _candidate_type_roi_metrics(frame: pd.DataFrame) -> dict[str, object]:
    counts = pd.to_numeric(frame["candidate_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    hits = frame["actual_in_candidates"].astype(bool).to_numpy()
    payouts = pd.to_numeric(frame["3rt"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total_ticket_count = float(counts.sum())
    total_bet_amount = total_ticket_count * 100.0
    returns = np.where(hits, payouts, 0.0)
    hit_payouts = payouts[hits]
    return {
        "n_races": int(len(frame)),
        "total_ticket_count": int(total_ticket_count),
        "avg_candidate_count": float(counts.mean()) if len(frame) else 0.0,
        "hit_count": int(hits.sum()),
        "hit_rate_per_race": float(hits.mean()) if len(frame) else 0.0,
        "hit_rate_per_ticket": float(hits.sum() / total_ticket_count) if total_ticket_count else 0.0,
        "total_bet_amount": int(total_bet_amount),
        "total_return": int(returns.sum()),
        "roi": float(returns.sum() / total_bet_amount) if total_bet_amount else 0.0,
        "average_hit_payout": float(hit_payouts.mean()) if len(hit_payouts) else 0.0,
        "median_hit_payout": float(np.median(hit_payouts)) if len(hit_payouts) else 0.0,
        "max_hit_payout": int(hit_payouts.max()) if len(hit_payouts) else 0,
        "exploration_note": "June 2026 exploratory analysis; validate on future months before adoption",
    }


def _write_candidate_type_roi_tables(
    candidate_type_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_groups = candidate_type_predictions.groupby(
        ["buy_grade", "candidate_type"],
        observed=True,
        dropna=False,
    )
    for (buy_grade, candidate_type), group in summary_groups:
        row = {
            "buy_grade": buy_grade,
            "candidate_type": candidate_type,
        }
        row.update(_candidate_type_roi_metrics(group))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    grade_order = {"A": 1, "B": 2, "C": 3, "D": 4}
    type_order = {"all3": 1, "2plus_only": 2, "union_only": 3}
    summary["_grade_order"] = summary["buy_grade"].map(grade_order).fillna(99)
    summary["_type_order"] = summary["candidate_type"].map(type_order).fillna(99)
    summary = summary.sort_values(["_grade_order", "_type_order"]).drop(
        columns=["_grade_order", "_type_order"]
    )
    summary.to_csv(output_dir / "candidate_type_roi_analysis.csv", index=False, encoding="utf-8-sig")

    condition_rows = []
    condition_groups = candidate_type_predictions.groupby(
        ["buy_grade", "roughness_bin", "race_group", "jcd", "candidate_type"],
        observed=True,
        dropna=False,
    )
    for (buy_grade, roughness_bin, race_group, jcd, candidate_type), group in condition_groups:
        row = {
            "buy_grade": buy_grade,
            "roughness_bin": roughness_bin,
            "race_group": race_group,
            "jcd": jcd,
            "candidate_type": candidate_type,
        }
        row.update(_candidate_type_roi_metrics(group))
        condition_rows.append(row)
    by_condition = pd.DataFrame(condition_rows)
    rough_order = {
        "Q1_low_roughness": 1,
        "Q2": 2,
        "Q3": 3,
        "Q4": 4,
        "Q5_high_roughness": 5,
    }
    race_order = {"early": 1, "middle": 2, "late": 3}
    by_condition["_grade_order"] = by_condition["buy_grade"].map(grade_order).fillna(99)
    by_condition["_rough_order"] = by_condition["roughness_bin"].map(rough_order).fillna(99)
    by_condition["_race_order"] = by_condition["race_group"].map(race_order).fillna(99)
    by_condition["_type_order"] = by_condition["candidate_type"].map(type_order).fillna(99)
    by_condition = by_condition.sort_values(
        ["_grade_order", "_rough_order", "_race_order", "jcd", "_type_order"]
    ).drop(columns=["_grade_order", "_rough_order", "_race_order", "_type_order"])
    by_condition.to_csv(
        output_dir / "candidate_type_roi_by_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_cols = [
        "jcd",
        "r",
        "race_group",
        "roughness_bin",
        "buy_grade",
        "candidate_type",
        "actual_result",
        "3rt",
        "candidate_tickets",
        "candidate_count",
        "actual_in_candidates",
        "exploration_note",
    ]
    candidate_type_predictions[prediction_cols].to_csv(
        output_dir / "candidate_type_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "candidate_type_roi_analysis": summary,
        "candidate_type_roi_by_condition": by_condition,
        "candidate_type_predictions": candidate_type_predictions[prediction_cols],
    }


def write_combined_buy_condition_analysis(
    buy_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    grouped = buy_predictions.groupby(["roughness_bin", "race_no_bin"], observed=True, dropna=False)
    for (roughness_bin, race_no_bin), group in grouped:
        for condition_name, candidate_col in AGREEMENT_CONDITIONS.items():
            row = {
                "roughness_bin": roughness_bin,
                "race_no_bin": race_no_bin,
                "agreement_condition": condition_name,
            }
            row.update(_candidate_metrics(group, candidate_col))
            combined_rows.append(row)
    combined = pd.DataFrame(combined_rows)
    rough_order = {
        "Q1_low_roughness": 1,
        "Q2": 2,
        "Q3": 3,
        "Q4": 4,
        "Q5_high_roughness": 5,
    }
    race_order = {"early": 1, "middle": 2, "late": 3}
    condition_order = {"2plus_top5": 1, "all3_top5": 2, "union_top5": 3}
    combined["_rough_order"] = combined["roughness_bin"].map(rough_order).fillna(99)
    combined["_race_order"] = combined["race_no_bin"].map(race_order).fillna(99)
    combined["_condition_order"] = combined["agreement_condition"].map(condition_order).fillna(99)
    combined = combined.sort_values(["_rough_order", "_race_order", "_condition_order"]).drop(
        columns=["_rough_order", "_race_order", "_condition_order"]
    )
    combined.to_csv(output_dir / "analysis_by_combined_conditions.csv", index=False, encoding="utf-8-sig")

    grade_rows = []
    for grade, group in buy_predictions.groupby("buy_grade", observed=True, dropna=False):
        if grade in {"A", "B"}:
            candidate_col = "candidate_tickets_2plus"
        elif grade == "C":
            candidate_col = "candidate_tickets_all3"
        else:
            # Grade D is a diagnostic no-buy bucket; union shows the broadest candidate quality there.
            candidate_col = "candidate_tickets_union"
        row = {
            "buy_grade": grade,
            "candidate_rule_used": candidate_col.replace("candidate_tickets_", ""),
        }
        row.update(_candidate_metrics(group, candidate_col))
        grade_rows.append(row)
    grade_analysis = pd.DataFrame(grade_rows)
    grade_order = {"A": 1, "B": 2, "C": 3, "D": 4}
    grade_analysis["_grade_order"] = grade_analysis["buy_grade"].map(grade_order).fillna(99)
    grade_analysis = grade_analysis.sort_values("_grade_order").drop(columns=["_grade_order"])
    grade_analysis.to_csv(output_dir / "buy_grade_analysis.csv", index=False, encoding="utf-8-sig")

    prediction_cols = [
        "jcd",
        "r",
        "roughness_bin",
        "buy_grade",
        "actual_result",
        "3rt",
        "candidate_tickets_2plus",
        "candidate_tickets_all3",
        "candidate_tickets_union",
        "actual_in_2plus",
        "actual_in_all3",
        "actual_in_union",
        "candidate_count_2plus",
        "candidate_count_all3",
        "candidate_count_union",
    ]
    buy_predictions[prediction_cols].to_csv(
        output_dir / "buy_grade_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roi_outputs = write_buy_grade_roi_analysis(buy_predictions, output_dir)
    candidate_type_predictions = build_candidate_type_predictions(buy_predictions)
    candidate_type_outputs = _write_candidate_type_roi_tables(candidate_type_predictions, output_dir)
    return {
        "combined_conditions": combined,
        "buy_grade_analysis": grade_analysis,
        "buy_grade_predictions": buy_predictions[prediction_cols],
        **roi_outputs,
        **candidate_type_outputs,
    }


def write_model_agreement_analysis_from_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    agreement_models: Iterable[str] = AGREEMENT_MODELS,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _agreement_rows_from_predictions(predictions, agreement_models)
    agreement = pd.DataFrame(rows)
    agreement.to_csv(output_dir / "model_agreement_analysis.csv", index=False, encoding="utf-8-sig")
    return agreement


def write_specialty_analysis(
    test_df: pd.DataFrame,
    results: list[EvaluationResult],
    output_dir: str | Path,
    *,
    target_models: Iterable[str] = TARGET_ANALYSIS_MODELS,
    agreement_models: Iterable[str] = AGREEMENT_MODELS,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_models = list(target_models)
    agreement_models = list(agreement_models)
    details = _details_from_results(test_df, results, target_models)
    _write_group_analyses(details, output_dir)

    # Agreement analysis uses the same wide prediction shape that is also written for review.
    wide = pd.DataFrame(
        {
            "jcd": test_df["jcd"].astype(str).str.zfill(2).to_numpy(),
            "r": pd.to_numeric(test_df["r"], errors="coerce").astype(int).to_numpy(),
            "actual_result": actual_trifecta_strings(test_df).to_numpy(),
        }
    )
    for result in results:
        model = str(result.metrics["model"])
        if model not in agreement_models:
            continue
        prefix = safe_model_column_name(model)
        wide[f"{prefix}_top5_predictions"] = result.details["top5_predictions"].to_numpy()

    agreement = write_model_agreement_analysis_from_predictions(wide, output_dir, agreement_models=agreement_models)
    buy_predictions = build_buy_grade_predictions(test_df, wide, agreement_models=agreement_models)
    buy_outputs = write_combined_buy_condition_analysis(buy_predictions, output_dir)
    return {
        "details": details,
        "agreement": agreement,
        **buy_outputs,
    }


def write_specialty_analysis_from_predictions(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_models: Iterable[str] = TARGET_ANALYSIS_MODELS,
    agreement_models: Iterable[str] = AGREEMENT_MODELS,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = _details_from_predictions(test_df, predictions, target_models)
    _write_group_analyses(details, output_dir)
    agreement = write_model_agreement_analysis_from_predictions(
        predictions,
        output_dir,
        agreement_models=agreement_models,
    )
    buy_predictions = build_buy_grade_predictions(test_df, predictions, agreement_models=agreement_models)
    buy_outputs = write_combined_buy_condition_analysis(buy_predictions, output_dir)
    return {
        "details": details,
        "agreement": agreement,
        **buy_outputs,
    }
