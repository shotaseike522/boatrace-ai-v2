from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from boat_model.data_loader import load_race_data
from boat_model.features import add_basic_features
from run_full_probability_calibration import brier_multiclass, ece_by_prob_bucket, fit_log_ticket_correction, markdown_table, metrics, split_past_by_row_order, target_indices
from run_roughness_calibration_analysis import (
    MODEL_OUTER,
    RoughnessCalibration,
    apply_calibration,
    build_roughness_calibrations,
    roughness_features,
    roughness_thresholds,
    target_outer_probability,
)


DEFAULT_PAST = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_1_past_201307_202602.csv"
DEFAULT_RECENT = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_2_recent_202603_202605.csv"
ROUGHNESS_LABELS = ["rough_Q1", "rough_Q2", "rough_Q3", "rough_Q4", "rough_Q5"]
SEGMENTS: dict[str, list[int] | None] = {
    "all_races": None,
    **{f"r{i:02d}": [i] for i in range(1, 13)},
    "early": [1, 2, 3, 4],
    "middle": [5, 6, 7, 8],
    "late": [9, 10],
    "final": [11, 12],
}
SEGMENT_ORDER = ["all_races", *[f"r{i:02d}" for i in range(1, 13)], "early", "middle", "late", "final"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid roughness calibration checks.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST)
    parser.add_argument("--recent-csv", default=DEFAULT_RECENT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split-a-size", type=int, default=200_000)
    parser.add_argument("--split-b-size", type=int, default=200_000)
    return parser.parse_args()


def load_featured(path: str, *, require_registrations: bool, name: str) -> pd.DataFrame:
    raw, _ = load_race_data(path, require_registrations=require_registrations, name=name)
    return add_basic_features(raw).reset_index(drop=True)


def find_model(models: list[RoughnessCalibration], method: str, k: int = 500) -> RoughnessCalibration:
    for model in models:
        if model.calibration_method == method and model.k == k:
            return model
    raise KeyError(f"Missing calibration model method={method}, k={k}")


def segment_mask(df: pd.DataFrame, segment: str) -> np.ndarray:
    races = SEGMENTS[segment]
    if races is None:
        return np.ones(len(df), dtype=bool)
    r = pd.to_numeric(df["r"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.isin(r, races)


def hybrid_probability(
    *,
    df: pd.DataFrame,
    raw: np.ndarray,
    traditional_model: RoughnessCalibration,
    roughness_model: RoughnessCalibration,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    traditional, _ = apply_calibration(traditional_model, df, raw)
    rough, _ = apply_calibration(roughness_model, df, raw)
    rough_idx, _ = roughness_features(df, roughness_model.thresholds)
    if strategy == "all_roughness":
        use_rough = np.ones(len(df), dtype=bool)
    elif strategy == "Q1_fallback":
        use_rough = rough_idx >= 1
    elif strategy == "Q1_Q2_fallback":
        use_rough = rough_idx >= 2
    elif strategy == "Q4_Q5_only":
        use_rough = rough_idx >= 3
    else:
        raise ValueError(strategy)
    out = np.where(use_rough[:, None], rough, traditional)
    return out.astype(np.float32), rough_idx


def eval_proba(df: pd.DataFrame, raw: np.ndarray, calibrated: np.ndarray, *, dataset: str, strategy: str, group_type: str, group_name: str) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "model_name": MODEL_OUTER,
        "hybrid_strategy": strategy,
        "group_type": group_type,
        "group_name": group_name,
        "n_races": len(df),
    }
    row.update(metrics(df, raw, "raw"))
    raw_ece, _ = ece_by_prob_bucket(df, raw)
    row["raw_ECE"] = raw_ece
    row.update(metrics(df, calibrated, "calibrated"))
    cal_ece, _ = ece_by_prob_bucket(df, calibrated)
    row["calibrated_ECE"] = cal_ece
    for metric in [
        "logloss",
        "brier_score",
        "ECE",
        "top1_hit_rate",
        "top3_contains_rate",
        "top5_contains_rate",
        "top10_contains_rate",
        "mean_actual_rank",
    ]:
        row[f"delta_{metric}"] = row[f"calibrated_{metric}"] - row[f"raw_{metric}"]
    actual_idx = target_indices(df)
    row["actual_ticket_raw_prob_mean"] = float(raw[np.arange(len(df)), actual_idx].mean())
    row["actual_ticket_calibrated_prob_mean"] = float(calibrated[np.arange(len(df)), actual_idx].mean())
    return row


def evaluate_dataset(
    *,
    dataset: str,
    df: pd.DataFrame,
    raw: np.ndarray,
    traditional_model: RoughnessCalibration,
    roughness_model: RoughnessCalibration,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bin_rows = []
    for strategy in ["all_roughness", "Q1_fallback", "Q1_Q2_fallback", "Q4_Q5_only"]:
        calibrated, rough_idx = hybrid_probability(
            df=df,
            raw=raw,
            traditional_model=traditional_model,
            roughness_model=roughness_model,
            strategy=strategy,
        )
        rows.append(eval_proba(df, raw, calibrated, dataset=dataset, strategy=strategy, group_type="all", group_name="all_races"))
        for segment in SEGMENT_ORDER:
            mask = segment_mask(df, segment)
            if not mask.any():
                continue
            rows.append(
                eval_proba(
                    df.loc[mask].reset_index(drop=True),
                    raw[mask],
                    calibrated[mask],
                    dataset=dataset,
                    strategy=strategy,
                    group_type="race_segment",
                    group_name=segment,
                )
            )
        for idx, label in enumerate(ROUGHNESS_LABELS):
            mask = rough_idx == idx
            if not mask.any():
                continue
            bin_rows.append(
                eval_proba(
                    df.loc[mask].reset_index(drop=True),
                    raw[mask],
                    calibrated[mask],
                    dataset=dataset,
                    strategy=strategy,
                    group_type="roughness_bin",
                    group_name=label,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def write_summary(path: Path, comp: pd.DataFrame, by_bin: pd.DataFrame) -> None:
    all_rows = comp[(comp["group_type"].eq("all")) & (comp["group_name"].eq("all_races"))].copy()
    c_all = all_rows[all_rows["dataset"].eq("C_future_check")].sort_values(["calibrated_logloss", "calibrated_ECE"])
    recent_all = all_rows[all_rows["dataset"].eq("validation_202603_202605")].copy()
    best = c_all.head(1)[["hybrid_strategy"]]
    recent_best = best.merge(recent_all, on="hybrid_strategy", how="left")
    c_bins = by_bin[by_bin["dataset"].eq("C_future_check")].copy()
    recent_bins = by_bin[by_bin["dataset"].eq("validation_202603_202605")].copy()
    q1_compare = by_bin[by_bin["group_name"].eq("rough_Q1")].copy()
    lines = [
        "# Roughness Hybrid Calibration Summary",
        "",
        f"対象モデル: `{MODEL_OUTER}`",
        "",
        "堅いレースだけ過補正になっていないかを見るため、roughness校正と従来校正のレース単位ハイブリッドを比較しました。",
        "",
        "## C区間: 全体比較",
        markdown_table(
            c_all[
                [
                    "hybrid_strategy",
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
                    "delta_top3_contains_rate",
                    "delta_top5_contains_rate",
                    "delta_top10_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ],
            8,
        ),
        "",
        "## 2026/3-5: C区間最良設定の固定確認",
        markdown_table(
            recent_best[
                [
                    "hybrid_strategy",
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
                    "delta_top3_contains_rate",
                    "delta_top5_contains_rate",
                    "delta_top10_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ],
            8,
        ),
        "",
        "## Q1過補正チェック",
        markdown_table(
            q1_compare[
                [
                    "dataset",
                    "hybrid_strategy",
                    "n_races",
                    "raw_logloss",
                    "calibrated_logloss",
                    "delta_logloss",
                    "raw_ECE",
                    "calibrated_ECE",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ].sort_values(["dataset", "calibrated_logloss"]),
            12,
        ),
        "",
        "## C区間: 荒さ別",
        markdown_table(
            c_bins[
                [
                    "hybrid_strategy",
                    "group_name",
                    "n_races",
                    "delta_logloss",
                    "delta_brier_score",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ].sort_values(["hybrid_strategy", "group_name"]),
            24,
        ),
        "",
        "## 2026/3-5: 荒さ別",
        markdown_table(
            recent_bins[
                [
                    "hybrid_strategy",
                    "group_name",
                    "n_races",
                    "delta_logloss",
                    "delta_brier_score",
                    "delta_ECE",
                    "delta_top5_contains_rate",
                    "delta_mean_actual_rank",
                ]
            ].sort_values(["hybrid_strategy", "group_name"]),
            24,
        ),
        "",
        "## Notes",
        "- `all_roughness`: 全レースでroughness入り校正。",
        "- `Q1_fallback`: rough_Q1だけ従来のprob_bucket×ticket_pattern校正へ戻します。",
        "- `Q1_Q2_fallback`: rough_Q1/Q2を従来校正へ戻します。",
        "- `Q4_Q5_only`: rough_Q4/Q5だけroughness入り校正にします。",
        "- C区間を主評価、2026/3-5は固定適用の参考確認です。",
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

    print("Fitting B calibration models...")
    b_models, _ = build_roughness_calibrations(df=b, proba=b_target, calibration_period="B_only", thresholds=b_thresholds)
    print("Fitting B+C calibration models...")
    bc_models, _ = build_roughness_calibrations(df=bc_df, proba=bc_target, calibration_period="B_C_out_of_sample", thresholds=bc_thresholds)

    b_traditional = find_model(b_models, "prob_bucket_ticket_pattern", 500)
    b_rough = find_model(b_models, "prob_bucket_ticket_pattern_roughness", 500)
    bc_traditional = find_model(bc_models, "prob_bucket_ticket_pattern", 500)
    bc_rough = find_model(bc_models, "prob_bucket_ticket_pattern_roughness", 500)

    print("Evaluating C future check...")
    c_comp, c_bin = evaluate_dataset(
        dataset="C_future_check",
        df=c,
        raw=c_target,
        traditional_model=b_traditional,
        roughness_model=b_rough,
    )
    print("Evaluating 2026/3-5 reference...")
    recent_comp, recent_bin = evaluate_dataset(
        dataset="validation_202603_202605",
        df=recent,
        raw=recent_target,
        traditional_model=bc_traditional,
        roughness_model=bc_rough,
    )

    comp = pd.concat([c_comp, recent_comp], ignore_index=True)
    by_bin = pd.concat([c_bin, recent_bin], ignore_index=True)
    comp.to_csv(out / "roughness_hybrid_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    by_bin.to_csv(out / "roughness_hybrid_by_bin.csv", index=False, encoding="utf-8-sig")
    write_summary(out / "roughness_hybrid_summary.md", comp, by_bin)
    print(f"Wrote roughness hybrid outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
