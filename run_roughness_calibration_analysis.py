from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boat_model.artifacts import compute_roughness_score
from boat_model.data_loader import load_race_data
from boat_model.features import add_basic_features, target_indices
from run_full_probability_calibration import (
    EPS,
    MODEL_OUTER,
    PATTERN_LABELS,
    PROB_LABELS,
    PROB_DISPLAY,
    TICKET_PATTERN_IDX,
    actual_rank_fast,
    apply_outer_boost,
    brier_multiclass,
    ece_by_prob_bucket,
    fit_log_ticket_correction,
    markdown_table,
    metrics,
    probability_bucket_indices,
    rank_matrix,
    split_past_by_row_order,
)


DEFAULT_PAST = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_1_past_201307_202602.csv"
DEFAULT_RECENT = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_2_recent_202603_202605.csv"

ROUGHNESS_LABELS = ["rough_Q1", "rough_Q2", "rough_Q3", "rough_Q4", "rough_Q5"]
K_CANDIDATES = [500, 1000, 3000]
MIN_COUNT = 300
N_TICKETS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roughness-aware probability calibration analysis.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST)
    parser.add_argument("--recent-csv", default=DEFAULT_RECENT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split-a-size", type=int, default=200_000)
    parser.add_argument("--split-b-size", type=int, default=200_000)
    return parser.parse_args()


def load_featured(path: str, *, require_registrations: bool, name: str) -> pd.DataFrame:
    raw, _ = load_race_data(path, require_registrations=require_registrations, name=name)
    return add_basic_features(raw).reset_index(drop=True)


def roughness_thresholds(df: pd.DataFrame) -> list[float]:
    score = compute_roughness_score(df)
    return [float(score.quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]


def roughness_features(df: pd.DataFrame, thresholds: list[float]) -> tuple[np.ndarray, pd.DataFrame]:
    score = compute_roughness_score(df)
    bins = pd.cut(score, bins=[-np.inf, *thresholds, np.inf], labels=ROUGHNESS_LABELS, include_lowest=True)
    # Very rare edge NaNs can occur if thresholds are degenerate; keep them in the middle bucket.
    bins = bins.astype("object").where(pd.notna(bins), "rough_Q3")
    idx = pd.Categorical(bins, categories=ROUGHNESS_LABELS, ordered=True).codes
    detail = pd.DataFrame({"roughness_score": score, "roughness_bin": bins.astype(str), "roughness_idx": idx})
    return idx.astype(np.int8), detail


def target_outer_probability(df: pd.DataFrame, raw_direct: np.ndarray, direct_jcdr) -> np.ndarray:
    base = direct_jcdr.apply(df, raw_direct, strength=1.0, clip=0.60, top_n=20)
    return apply_outer_boost(
        df,
        base,
        pattern="outer_in_23_boost_by_outer_winrate",
        boost_score=0.40,
        clip=0.40,
        top_n=20,
    )


def count_by_cell(
    *,
    proba: np.ndarray,
    actual_idx: np.ndarray,
    cell_idx: np.ndarray,
    actual_cell_idx: np.ndarray,
    size: int,
) -> dict[str, np.ndarray]:
    flat_cell = cell_idx.reshape(-1).astype(np.int32)
    flat_prob = proba.reshape(-1).astype(np.float64)
    n_rows = np.bincount(flat_cell, minlength=size).astype(np.float64)
    sum_prob = np.bincount(flat_cell, weights=flat_prob, minlength=size).astype(np.float64)
    hits = np.bincount(actual_cell_idx.astype(np.int32), minlength=size).astype(np.float64)
    avg_prob = np.divide(sum_prob, n_rows, out=np.zeros_like(sum_prob), where=n_rows > 0)
    hit_rate = np.divide(hits, n_rows, out=np.zeros_like(hits), where=n_rows > 0)
    local_lift = np.divide(hits, sum_prob, out=np.ones_like(hits), where=sum_prob > EPS)
    return {
        "n_rows": n_rows,
        "sum_prob": sum_prob,
        "hits": hits,
        "avg_prob": avg_prob,
        "hit_rate": hit_rate,
        "local_lift": local_lift,
    }


def baseline_prob_pattern_lift(proba: np.ndarray, df: pd.DataFrame, *, k: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    actual_idx = target_indices(df)
    prob_idx = probability_bucket_indices(proba)
    pattern_idx = np.tile(TICKET_PATTERN_IDX.reshape(1, -1), (len(df), 1))
    cell_idx = prob_idx.astype(np.int32) * len(PATTERN_LABELS) + pattern_idx.astype(np.int32)
    actual_cell = prob_idx[np.arange(len(df)), actual_idx].astype(np.int32) * len(PATTERN_LABELS) + TICKET_PATTERN_IDX[actual_idx].astype(np.int32)
    counts = count_by_cell(
        proba=proba,
        actual_idx=actual_idx,
        cell_idx=cell_idx,
        actual_cell_idx=actual_cell,
        size=len(PROB_LABELS) * len(PATTERN_LABELS),
    )
    shrink = counts["n_rows"] / (counts["n_rows"] + k)
    lift = shrink * counts["local_lift"] + (1.0 - shrink) * 1.0
    return lift.astype(np.float32), counts


@dataclass
class RoughnessCalibration:
    calibration_method: str
    calibration_period: str
    k: int
    min_count: int
    thresholds: list[float]
    base_prob_pattern_lift: np.ndarray
    local_lift: np.ndarray | None
    local_n: np.ndarray | None

    @property
    def config_name(self) -> str:
        return f"{self.calibration_method}_k{self.k}_min{self.min_count}"


def build_roughness_calibrations(
    *,
    df: pd.DataFrame,
    proba: np.ndarray,
    calibration_period: str,
    thresholds: list[float],
) -> tuple[list[RoughnessCalibration], pd.DataFrame]:
    actual_idx = target_indices(df)
    prob_idx = probability_bucket_indices(proba)
    rough_idx, _ = roughness_features(df, thresholds)
    rough_matrix = np.tile(rough_idx.reshape(-1, 1), (1, N_TICKETS))
    pattern_matrix = np.tile(TICKET_PATTERN_IDX.reshape(1, -1), (len(df), 1))

    rows: list[dict[str, object]] = []
    models: list[RoughnessCalibration] = []
    for k in K_CANDIDATES:
        base_lift, base_counts = baseline_prob_pattern_lift(proba, df, k=k)
        models.append(
            RoughnessCalibration(
                calibration_method="prob_bucket_ticket_pattern",
                calibration_period=calibration_period,
                k=k,
                min_count=MIN_COUNT,
                thresholds=thresholds,
                base_prob_pattern_lift=base_lift,
                local_lift=None,
                local_n=None,
            )
        )
        rows.extend(
            calibration_rows(
                calibration_period=calibration_period,
                method="prob_bucket_ticket_pattern",
                k=k,
                counts=base_counts,
                lift=base_lift,
                fallback="none",
                dimensions=("prob_bucket", "ticket_pattern"),
            )
        )

        specs = [
            (
                "prob_bucket_roughness",
                prob_idx.astype(np.int32) * len(ROUGHNESS_LABELS) + rough_matrix.astype(np.int32),
                prob_idx[np.arange(len(df)), actual_idx].astype(np.int32) * len(ROUGHNESS_LABELS) + rough_idx.astype(np.int32),
                len(PROB_LABELS) * len(ROUGHNESS_LABELS),
                ("prob_bucket", "roughness_bin"),
            ),
            (
                "ticket_pattern_roughness",
                pattern_matrix.astype(np.int32) * len(ROUGHNESS_LABELS) + rough_matrix.astype(np.int32),
                TICKET_PATTERN_IDX[actual_idx].astype(np.int32) * len(ROUGHNESS_LABELS) + rough_idx.astype(np.int32),
                len(PATTERN_LABELS) * len(ROUGHNESS_LABELS),
                ("ticket_pattern", "roughness_bin"),
            ),
            (
                "prob_bucket_ticket_pattern_roughness",
                (prob_idx.astype(np.int32) * len(PATTERN_LABELS) + pattern_matrix.astype(np.int32)) * len(ROUGHNESS_LABELS)
                + rough_matrix.astype(np.int32),
                (prob_idx[np.arange(len(df)), actual_idx].astype(np.int32) * len(PATTERN_LABELS) + TICKET_PATTERN_IDX[actual_idx].astype(np.int32))
                * len(ROUGHNESS_LABELS)
                + rough_idx.astype(np.int32),
                len(PROB_LABELS) * len(PATTERN_LABELS) * len(ROUGHNESS_LABELS),
                ("prob_bucket", "ticket_pattern", "roughness_bin"),
            ),
        ]
        for method, cell_idx, actual_cell, size, dimensions in specs:
            counts = count_by_cell(
                proba=proba,
                actual_idx=actual_idx,
                cell_idx=cell_idx,
                actual_cell_idx=actual_cell,
                size=size,
            )
            shrink = counts["n_rows"] / (counts["n_rows"] + k)
            local = shrink * counts["local_lift"] + (1.0 - shrink) * 1.0
            models.append(
                RoughnessCalibration(
                    calibration_method=method,
                    calibration_period=calibration_period,
                    k=k,
                    min_count=MIN_COUNT,
                    thresholds=thresholds,
                    base_prob_pattern_lift=base_lift,
                    local_lift=local.astype(np.float32),
                    local_n=counts["n_rows"].astype(np.int64),
                )
            )
            rows.extend(
                calibration_rows(
                    calibration_period=calibration_period,
                    method=method,
                    k=k,
                    counts=counts,
                    lift=local,
                    fallback=f"prob_bucket_ticket_pattern_if_n_lt_{MIN_COUNT}",
                    dimensions=dimensions,
                )
            )
    return models, pd.DataFrame(rows)


def calibration_rows(
    *,
    calibration_period: str,
    method: str,
    k: int,
    counts: dict[str, np.ndarray],
    lift: np.ndarray,
    fallback: str,
    dimensions: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    for idx in range(len(lift)):
        decoded = decode_cell(idx, dimensions)
        row = {
            "calibration_period": calibration_period,
            "model_name": MODEL_OUTER,
            "calibration_method": method,
            "config_name": f"{method}_k{k}_min{MIN_COUNT}",
            "k": k,
            "min_count": MIN_COUNT,
            "cell_index": idx,
            "n_rows": int(counts["n_rows"][idx]),
            "sum_raw_prob": float(counts["sum_prob"][idx]),
            "actual_hits": int(counts["hits"][idx]),
            "avg_raw_prob": float(counts["avg_prob"][idx]),
            "actual_hit_rate": float(counts["hit_rate"][idx]),
            "local_lift": float(counts["local_lift"][idx]),
            "calibration_lift": float(lift[idx]),
            "fallback_rule": fallback,
        }
        row.update(decoded)
        rows.append(row)
    return rows


def decode_cell(idx: int, dimensions: tuple[str, ...]) -> dict[str, str]:
    out = {"prob_bucket": "", "prob_bucket_display": "", "ticket_pattern": "", "roughness_bin": ""}
    rem = idx
    if dimensions == ("prob_bucket", "ticket_pattern"):
        p = rem // len(PATTERN_LABELS)
        t = rem % len(PATTERN_LABELS)
        out.update(prob_bucket=PROB_LABELS[p], prob_bucket_display=PROB_DISPLAY[PROB_LABELS[p]], ticket_pattern=PATTERN_LABELS[t])
    elif dimensions == ("prob_bucket", "roughness_bin"):
        p = rem // len(ROUGHNESS_LABELS)
        r = rem % len(ROUGHNESS_LABELS)
        out.update(prob_bucket=PROB_LABELS[p], prob_bucket_display=PROB_DISPLAY[PROB_LABELS[p]], roughness_bin=ROUGHNESS_LABELS[r])
    elif dimensions == ("ticket_pattern", "roughness_bin"):
        t = rem // len(ROUGHNESS_LABELS)
        r = rem % len(ROUGHNESS_LABELS)
        out.update(ticket_pattern=PATTERN_LABELS[t], roughness_bin=ROUGHNESS_LABELS[r])
    elif dimensions == ("prob_bucket", "ticket_pattern", "roughness_bin"):
        r = rem % len(ROUGHNESS_LABELS)
        rem //= len(ROUGHNESS_LABELS)
        t = rem % len(PATTERN_LABELS)
        p = rem // len(PATTERN_LABELS)
        out.update(prob_bucket=PROB_LABELS[p], prob_bucket_display=PROB_DISPLAY[PROB_LABELS[p]], ticket_pattern=PATTERN_LABELS[t], roughness_bin=ROUGHNESS_LABELS[r])
    return out


def source_indices(model: RoughnessCalibration, proba: np.ndarray, rough_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob_idx = probability_bucket_indices(proba)
    rough_matrix = np.tile(rough_idx.reshape(-1, 1), (1, N_TICKETS))
    pattern_matrix = np.tile(TICKET_PATTERN_IDX.reshape(1, -1), (len(proba), 1))
    base_source = prob_idx.astype(np.int32) * len(PATTERN_LABELS) + pattern_matrix.astype(np.int32)
    if model.calibration_method == "prob_bucket_ticket_pattern":
        return base_source, base_source, np.ones_like(base_source, dtype=bool)
    if model.calibration_method == "prob_bucket_roughness":
        local_source = prob_idx.astype(np.int32) * len(ROUGHNESS_LABELS) + rough_matrix.astype(np.int32)
    elif model.calibration_method == "ticket_pattern_roughness":
        local_source = pattern_matrix.astype(np.int32) * len(ROUGHNESS_LABELS) + rough_matrix.astype(np.int32)
    elif model.calibration_method == "prob_bucket_ticket_pattern_roughness":
        local_source = (prob_idx.astype(np.int32) * len(PATTERN_LABELS) + pattern_matrix.astype(np.int32)) * len(ROUGHNESS_LABELS) + rough_matrix.astype(np.int32)
    else:
        raise ValueError(model.calibration_method)
    use_local = model.local_n[local_source] >= model.min_count
    return base_source, local_source, use_local


def apply_calibration(model: RoughnessCalibration, df: pd.DataFrame, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rough_idx, _ = roughness_features(df, model.thresholds)
    base_source, local_source, use_local = source_indices(model, proba, rough_idx)
    base = model.base_prob_pattern_lift[base_source]
    if model.local_lift is None:
        lift = base
    else:
        local = model.local_lift[local_source]
        lift = np.where(use_local, local, base)
    score = proba.astype(np.float64) * lift.astype(np.float64)
    denom = score.sum(axis=1, keepdims=True)
    calibrated = np.divide(score, denom, out=np.full_like(score, 1.0 / N_TICKETS), where=denom > EPS)
    return calibrated.astype(np.float32), lift.astype(np.float32)


def evaluate_config(dataset: str, df: pd.DataFrame, raw: np.ndarray, model: RoughnessCalibration) -> dict[str, object]:
    calibrated, _ = apply_calibration(model, df, raw)
    row = {
        "dataset": dataset,
        "model_name": MODEL_OUTER,
        "config_name": model.config_name,
        "calibration_method": model.calibration_method,
        "k": model.k,
        "min_count": model.min_count,
        "calibration_period": model.calibration_period,
    }
    row.update(metrics(df, raw, "raw"))
    raw_ece, _ = ece_by_prob_bucket(df, raw)
    row["raw_ECE"] = raw_ece
    row.update(metrics(df, calibrated, "calibrated"))
    cal_ece, _ = ece_by_prob_bucket(df, calibrated)
    row["calibrated_ECE"] = cal_ece
    for metric in ["logloss", "brier_score", "ECE", "top1_hit_rate", "top3_contains_rate", "top5_contains_rate", "mean_actual_rank"]:
        row[f"delta_{metric}"] = row[f"calibrated_{metric}"] - row[f"raw_{metric}"]
    return row


def metrics_by_roughness(dataset: str, df: pd.DataFrame, raw: np.ndarray, model: RoughnessCalibration) -> pd.DataFrame:
    calibrated, _ = apply_calibration(model, df, raw)
    rough_idx, detail = roughness_features(df, model.thresholds)
    rows = []
    for idx, label in enumerate(ROUGHNESS_LABELS):
        mask = rough_idx == idx
        if not mask.any():
            continue
        sub_df = df.loc[mask].reset_index(drop=True)
        raw_sub = raw[mask]
        cal_sub = calibrated[mask]
        row = {
            "dataset": dataset,
            "model_name": MODEL_OUTER,
            "config_name": model.config_name,
            "calibration_method": model.calibration_method,
            "k": model.k,
            "roughness_bin": label,
            "n_races": int(mask.sum()),
            "roughness_score_mean": float(detail.loc[mask, "roughness_score"].mean()),
        }
        row.update(metrics(sub_df, raw_sub, "raw"))
        raw_ece, _ = ece_by_prob_bucket(sub_df, raw_sub)
        row["raw_ECE"] = raw_ece
        row.update(metrics(sub_df, cal_sub, "calibrated"))
        cal_ece, _ = ece_by_prob_bucket(sub_df, cal_sub)
        row["calibrated_ECE"] = cal_ece
        actual_idx = target_indices(sub_df)
        actual_prob_raw = raw_sub[np.arange(len(sub_df)), actual_idx]
        actual_prob_cal = cal_sub[np.arange(len(sub_df)), actual_idx]
        row["actual_ticket_raw_prob_mean"] = float(actual_prob_raw.mean())
        row["actual_ticket_calibrated_prob_mean"] = float(actual_prob_cal.mean())
        for metric in ["logloss", "brier_score", "ECE", "top1_hit_rate", "top3_contains_rate", "top5_contains_rate", "mean_actual_rank"]:
            row[f"delta_{metric}"] = row[f"calibrated_{metric}"] - row[f"raw_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(path: Path, comparison: pd.DataFrame, by_bin: pd.DataFrame) -> None:
    c_rows = comparison[comparison["dataset"].eq("C_future_check")].sort_values(["calibrated_logloss", "calibrated_ECE"])
    recent_rows = comparison[comparison["dataset"].eq("validation_202603_202605")].copy()
    best = c_rows.head(1)
    recent_best = best[["config_name"]].merge(recent_rows, on="config_name", how="left")
    selected_bins = by_bin.merge(best[["config_name"]], on="config_name", how="inner")
    c_bins = selected_bins[selected_bins["dataset"].eq("C_future_check")]
    recent_bins = selected_bins[selected_bins["dataset"].eq("validation_202603_202605")]
    lines = [
        "# Roughness Calibration Summary",
        "",
        f"対象モデル: `{MODEL_OUTER}`",
        "",
        "C区間を主評価、2026/3-5を参考確認として、roughness_binを明示的に使う校正を比較しました。",
        f"少数セルは既存の `prob_bucket × ticket_pattern` へフォールバックし、min_count={MIN_COUNT}、kは {K_CANDIDATES} を比較しています。",
        "",
        "## C区間: 校正方式比較",
        markdown_table(
            c_rows[
                [
                    "config_name",
                    "calibration_method",
                    "k",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_brier_score",
                    "calibrated_brier_score",
                    "delta_brier_score",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top1_hit_rate",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ],
            16,
        ),
        "",
        "## C区間で選ばれた設定の2026/3-5固定確認",
        markdown_table(
            recent_best[
                [
                    "config_name",
                    "calibration_method",
                    "k",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_brier_score",
                    "calibrated_brier_score",
                    "delta_brier_score",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top1_hit_rate",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ],
            8,
        ),
        "",
        "## C区間: 荒さ別",
        markdown_table(
            c_bins[
                [
                    "roughness_bin",
                    "n_races",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                    "actual_ticket_raw_prob_mean",
                    "actual_ticket_calibrated_prob_mean",
                ]
            ],
            10,
        ),
        "",
        "## 2026/3-5: 荒さ別",
        markdown_table(
            recent_bins[
                [
                    "roughness_bin",
                    "n_races",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                    "actual_ticket_raw_prob_mean",
                    "actual_ticket_calibrated_prob_mean",
                ]
            ],
            10,
        ),
        "",
        "## Notes",
        "- roughness_scoreは既存の荒れ指数ロジックを使っています。",
        "- roughness_binの分位境界は、C評価用はB区間、2026/3-5評価用はB+C区間から作っています。",
        "- 2026/3-5の結果は校正値作成に使っていません。",
        "- EV用に採用するなら、C区間だけでなく2026/3-5でもlogloss/Brier/ECEが同方向に改善し、Top5が崩れない設定を優先してください。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    past = load_featured(args.past_csv, require_registrations=False, name="dataset_1_past")
    recent = load_featured(args.recent_csv, require_registrations=True, name="dataset_2_recent")
    _, b, c = split_past_by_row_order(past, args.split_a_size, args.split_b_size)

    final_cache = out / "final_fixed_correction_cache"
    sensitivity_cache = out / "correction_sensitivity_cache"
    b_raw = np.load(final_cache / "direct120_A_train_B_proba.npy")
    c_raw = np.load(sensitivity_cache / "direct120_AB_train_C_proba.npy")
    recent_raw = np.load(final_cache / "direct120_past55_recent_proba.npy")

    direct_jcdr = fit_log_ticket_correction(b, b_raw, "jcd_r_correction", k=500, min_count=300)
    b_target = target_outer_probability(b, b_raw, direct_jcdr)
    c_target = target_outer_probability(c, c_raw, direct_jcdr)
    recent_target = target_outer_probability(recent, recent_raw, direct_jcdr)

    b_thresholds = roughness_thresholds(b)
    bc_df = pd.concat([b, c], ignore_index=True)
    bc_target = np.vstack([b_target, c_target])
    bc_thresholds = roughness_thresholds(bc_df)

    print("Fitting B roughness calibration...")
    b_models, b_table = build_roughness_calibrations(
        df=b,
        proba=b_target,
        calibration_period="B_only",
        thresholds=b_thresholds,
    )
    print("Fitting B+C roughness calibration...")
    bc_models, bc_table = build_roughness_calibrations(
        df=bc_df,
        proba=bc_target,
        calibration_period="B_C_out_of_sample",
        thresholds=bc_thresholds,
    )

    rows = []
    bin_frames = []
    print("Evaluating C future check...")
    for model in b_models:
        rows.append(evaluate_config("C_future_check", c, c_target, model))
        bin_frames.append(metrics_by_roughness("C_future_check", c, c_target, model))
    print("Evaluating 2026/3-5 reference...")
    for model in bc_models:
        rows.append(evaluate_config("validation_202603_202605", recent, recent_target, model))
        bin_frames.append(metrics_by_roughness("validation_202603_202605", recent, recent_target, model))

    comparison = pd.DataFrame(rows)
    by_bin = pd.concat(bin_frames, ignore_index=True)
    calibration_table = pd.concat([b_table, bc_table], ignore_index=True)

    comparison.to_csv(out / "roughness_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    by_bin.to_csv(out / "roughness_calibration_by_bin.csv", index=False, encoding="utf-8-sig")
    # The full cell table is useful for debugging but not requested; keep it compactly in the comparison directory name.
    calibration_table.to_csv(out / "roughness_calibration_cell_table.csv", index=False, encoding="utf-8-sig")
    write_summary(out / "roughness_calibration_summary.md", comparison, by_bin)
    print(f"Wrote roughness calibration outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
