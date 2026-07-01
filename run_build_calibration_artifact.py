"""校正モデルをビルドして artifacts/calibration_artifact.joblib に保存する。

ローカルで1回だけ実行する。出力ファイルをリポジトリにコミットすれば、
GitHub Actions は過去CSV（55万行）や大容量.npyキャッシュなしで
校正済み予測を実行できるようになる。

実行例:
    python run_build_calibration_artifact.py \
        --past-csv "C:/Users/trium/OneDrive/Desktop/boat/dataset_1_past_201307_202606.csv" \
        --a-train-b-proba outputs/final_fixed_correction_cache/direct120_A_train_B_proba.npy \
        --ab-train-c-proba outputs/correction_sensitivity_cache/direct120_AB_train_C_proba.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BLOCKED_DEP_PATHS = {
    str((REPO_ROOT / "work" / "python_deps").resolve()).lower(),
}
sys.path = [
    entry
    for entry in sys.path
    if str(Path(entry or ".").resolve()).lower() not in BLOCKED_DEP_PATHS
]

import joblib
import numpy as np

from boat_model.data_loader import load_race_data
from boat_model.features import add_basic_features
from run_full_probability_calibration import (
    fit_log_ticket_correction,
    split_past_by_row_order,
)
from run_roughness_calibration_analysis import (
    build_roughness_calibrations,
    roughness_thresholds,
    target_outer_probability,
)
from run_roughness_hybrid_calibration import find_model

SPLIT_A_SIZE = 200_000
SPLIT_B_SIZE = 200_000
TRADITIONAL_METHOD = "prob_bucket_ticket_pattern"
ROUGHNESS_METHOD = "prob_bucket_ticket_pattern_roughness"
CALIBRATION_K = 500
DEFAULT_OUTPUT = "artifacts/calibration_artifact.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save calibration artifact.")
    parser.add_argument("--past-csv", required=True, help="Past race CSV (55万 rows).")
    parser.add_argument(
        "--a-train-b-proba",
        required=True,
        help="Pre-computed direct120 predictions for B split (A-trained).",
    )
    parser.add_argument(
        "--ab-train-c-proba",
        required=True,
        help="Pre-computed direct120 predictions for C split (AB-trained).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output joblib path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("過去データを読み込み中...")
    raw, _ = load_race_data(args.past_csv, require_registrations=False, name="past")
    past = add_basic_features(raw).reset_index(drop=True)
    print(f"  {len(past):,} レース読み込み完了")

    _, b, c = split_past_by_row_order(past, SPLIT_A_SIZE, SPLIT_B_SIZE)
    b_raw = np.load(args.a_train_b_proba)
    c_raw = np.load(args.ab_train_c_proba)
    print(f"  B={len(b):,}行 / C={len(c):,}行")

    print("jcd_r補正を構築中...")
    direct_jcdr = fit_log_ticket_correction(b, b_raw, "jcd_r_correction", k=500, min_count=300)

    print("roughness校正を構築中...")
    b_target = target_outer_probability(b, b_raw, direct_jcdr)
    c_target = target_outer_probability(c, c_raw, direct_jcdr)

    bc_df = __import__("pandas").concat([b, c], ignore_index=True)
    bc_target = np.vstack([b_target, c_target]).astype(np.float32)
    thresholds = roughness_thresholds(bc_df)
    models, calibration_table = build_roughness_calibrations(
        df=bc_df,
        proba=bc_target,
        calibration_period="B_C_out_of_sample",
        thresholds=thresholds,
    )
    traditional_model = find_model(models, TRADITIONAL_METHOD, CALIBRATION_K)
    roughness_model = find_model(models, ROUGHNESS_METHOD, CALIBRATION_K)

    artifact = {
        "direct_jcdr": direct_jcdr,
        "traditional_model": traditional_model,
        "roughness_model": roughness_model,
        "calibration_table": calibration_table,
        "thresholds": thresholds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"保存完了: {out_path.resolve()} ({size_mb:.1f} MB)")
    print("このファイルをリポジトリにコミットすればGitHub Actionsが使えます。")


if __name__ == "__main__":
    main()
