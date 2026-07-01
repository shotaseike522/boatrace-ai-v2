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
import pandas as pd

from boat_model.artifacts import load_joblib_artifact
from boat_model.data_loader import load_race_data
from boat_model.features import INDEX_TO_TRIFECTA, WINRATE_COLS, add_basic_features, normalize_probability_matrix
from run_full_probability_calibration import (
    MODEL_OUTER,
    PATTERN_LABELS,
    PROB_LABELS,
    TICKET_PATTERN_IDX,
    fit_log_ticket_correction,
    probability_bucket_indices,
    rank_matrix,
    split_past_by_row_order,
)
from run_roughness_calibration_analysis import (
    ROUGHNESS_LABELS,
    apply_calibration,
    build_roughness_calibrations,
    roughness_features,
    roughness_thresholds,
    source_indices,
    target_outer_probability,
)
from run_roughness_hybrid_calibration import find_model

try:
    from boat_model.artifacts import (  # type: ignore[attr-defined]
        load_knn_artifacts,
        knn_predict_proba_and_neighbors,
        similar_race_diagnostics,
    )
    _KNN_AVAILABLE = True
except (ImportError, AttributeError):
    _KNN_AVAILABLE = False


DEFAULT_PAST = r"C:\Users\trium\OneDrive\Desktop\boat\dataset_1_past_201307_202602.csv"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_A_B_CACHE = "outputs/final_fixed_correction_cache/direct120_A_train_B_proba.npy"
DEFAULT_AB_C_CACHE = "outputs/correction_sensitivity_cache/direct120_AB_train_C_proba.npy"
DEFAULT_CALIBRATION_ARTIFACT = "artifacts/calibration_artifact.joblib"

TRADITIONAL_METHOD = "prob_bucket_ticket_pattern"
ROUGHNESS_METHOD = "prob_bucket_ticket_pattern_roughness"
CALIBRATION_K = 500
MIN_COUNT = 300
SPLIT_A_SIZE = 200_000
SPLIT_B_SIZE = 200_000
DISPLAY_LABEL = "AI予想確率"
DISPLAY_NOTE = "過去データで実績補正した推定確率です。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create production site CSV from future race entries using "
            "outer_in_23_boost_by_outer_winrate_score0.40_clip0.40_top20 "
            "and Q1_fallback calibrated probabilities."
        )
    )
    parser.add_argument("--input", required=True, help="Future race entry CSV without r1/r2/r3/3rt.")
    parser.add_argument("--output", required=True, help="Site display CSV.")
    parser.add_argument("--internal-output", default="", help="Optional internal audit CSV. Defaults to *_internal.csv.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST, help="Historical CSV used only to rebuild correction/calibration.")
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--a-train-b-proba", default=DEFAULT_A_B_CACHE)
    parser.add_argument("--ab-train-c-proba", default=DEFAULT_AB_C_CACHE)
    parser.add_argument(
        "--calibration-artifact",
        default=DEFAULT_CALIBRATION_ARTIFACT,
        help="Pre-built calibration joblib. If present, skips rebuilding from past CSV.",
    )
    return parser.parse_args()


def read_future_races(path: str | Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(path)

    required = ["jcd", "r", *WINRATE_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input race CSV is missing required columns: {missing}")

    out = df.copy()
    out["jcd"] = out["jcd"].astype("string").str.strip().str.zfill(2)
    out["r"] = pd.to_numeric(out["r"], errors="coerce").astype("Int64")
    for col in WINRATE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["r"].isna().any():
        raise ValueError("Input race CSV has missing or invalid r values.")
    return out.reset_index(drop=True)


def date_values(df: pd.DataFrame) -> pd.Series:
    candidates = ["date", "race_date", "日付", "開催日", "年月日", "ymd"]
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([""] * len(df), index=df.index)


def load_featured_past(path: str | Path) -> pd.DataFrame:
    past, _ = load_race_data(path, require_registrations=False, name="past_201307_202602")
    return add_basic_features(past).reset_index(drop=True)


def direct120_predict_from_artifact(races: pd.DataFrame, artifacts_dir: str | Path) -> np.ndarray:
    model_path = Path(artifacts_dir) / "models" / "lightgbm_direct120.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing direct120 artifact: {model_path}")
    try:
        artifact = load_joblib_artifact(model_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not load lightgbm_direct120.joblib. This production script does not use fallback. "
            "Install/enable joblib and lightgbm in the Python environment, then rerun."
        ) from exc
    proba = artifact["model_object"].predict_proba(races)
    return normalize_probability_matrix(proba).astype(np.float32)


def calibration_n_lookup(table: pd.DataFrame, method: str, k: int) -> np.ndarray:
    sub = table[(table["calibration_method"].eq(method)) & (table["k"].eq(k))].copy()
    if sub.empty:
        raise ValueError(f"Calibration table does not contain method={method}, k={k}")
    lookup = np.zeros(int(sub["cell_index"].max()) + 1, dtype=np.int32)
    lookup[sub["cell_index"].astype(int).to_numpy()] = sub["n_rows"].astype(int).to_numpy()
    return lookup


def build_models_for_calibration(args: argparse.Namespace, future_raw_direct: np.ndarray, future: pd.DataFrame):
    past = load_featured_past(args.past_csv)
    _, b, c = split_past_by_row_order(past, SPLIT_A_SIZE, SPLIT_B_SIZE)
    b_raw = np.load(args.a_train_b_proba)
    c_raw = np.load(args.ab_train_c_proba)

    direct_jcdr = fit_log_ticket_correction(b, b_raw, "jcd_r_correction", k=500, min_count=300)
    b_target = target_outer_probability(b, b_raw, direct_jcdr)
    c_target = target_outer_probability(c, c_raw, direct_jcdr)
    future_ranking = target_outer_probability(future, future_raw_direct, direct_jcdr)

    bc_df = pd.concat([b, c], ignore_index=True)
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
    return future_ranking, calibration_table, traditional_model, roughness_model


def load_or_build_calibration(
    args: argparse.Namespace, future_raw_direct: np.ndarray, future: pd.DataFrame
):
    artifact_path = Path(args.calibration_artifact)
    if artifact_path.exists():
        print(f"校正モデルを読み込み中: {artifact_path}")
        artifact = joblib.load(artifact_path)
        direct_jcdr = artifact["direct_jcdr"]
        traditional_model = artifact["traditional_model"]
        roughness_model = artifact["roughness_model"]
        calibration_table = artifact["calibration_table"]
        future_ranking = target_outer_probability(future, future_raw_direct, direct_jcdr)
        return future_ranking, calibration_table, traditional_model, roughness_model
    print(f"校正アーティファクトが見つかりません: {artifact_path}")
    print("過去CSVから再構築します（初回のみ）...")
    return build_models_for_calibration(args, future_raw_direct, future)


def compute_knn_columns(future: pd.DataFrame, artifacts_dir: str | Path) -> pd.DataFrame | None:
    knn_dir = Path(artifacts_dir) / "knn"
    if not _KNN_AVAILABLE:
        print("⚠️ KNN: インポート不可。similar_*列はスキップします。")
        return None
    if not knn_dir.exists():
        print(f"⚠️ KNN: {knn_dir} が存在しません。similar_*列はスキップします。")
        return None
    try:
        knn_artifact = load_knn_artifacts(knn_dir)
        _, neighbors100 = knn_predict_proba_and_neighbors(future, knn_artifact, k=500, diagnostic_k=100)
        similar_df = similar_race_diagnostics(knn_artifact, neighbors100)
        return similar_df.reset_index(drop=True)
    except Exception as exc:
        print(f"⚠️ KNN処理でエラー: {exc}")
        return None


def apply_q1_fallback(
    *,
    df: pd.DataFrame,
    raw: np.ndarray,
    traditional_model,
    roughness_model,
    calibration_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    traditional_calibrated, traditional_lift = apply_calibration(traditional_model, df, raw)
    roughness_calibrated, roughness_lift = apply_calibration(roughness_model, df, raw)
    rough_idx, rough_detail = roughness_features(df, roughness_model.thresholds)
    use_roughness = rough_idx >= 1

    calibrated = np.where(use_roughness[:, None], roughness_calibrated, traditional_calibrated).astype(np.float32)
    lift = np.where(use_roughness[:, None], roughness_lift, traditional_lift).astype(np.float32)

    traditional_base, _, _ = source_indices(traditional_model, raw, rough_idx)
    rough_base, rough_local, rough_use_local = source_indices(roughness_model, raw, rough_idx)
    traditional_n_lookup = calibration_n_lookup(calibration_table, TRADITIONAL_METHOD, CALIBRATION_K)
    roughness_n_lookup = calibration_n_lookup(calibration_table, ROUGHNESS_METHOD, CALIBRATION_K)

    traditional_n = traditional_n_lookup[traditional_base]
    roughness_local_n = roughness_n_lookup[rough_local]
    roughness_fallback_n = traditional_n_lookup[rough_base]
    roughness_n = np.where(rough_use_local, roughness_local_n, roughness_fallback_n).astype(np.int32)
    calibration_n = np.where(use_roughness[:, None], roughness_n, traditional_n).astype(np.int32)

    details = {
        "lift": lift,
        "calibration_n": calibration_n,
        "use_roughness": use_roughness,
        "roughness_uses_local_cell": rough_use_local,
    }
    return calibrated, rough_idx, rough_detail, details


def ticket_pattern(ticket_idx: int) -> str:
    return PATTERN_LABELS[int(TICKET_PATTERN_IDX[int(ticket_idx)])]


def prob_bucket(raw_prob: float) -> str:
    idx = int(probability_bucket_indices(np.asarray([raw_prob], dtype=np.float32))[0])
    return PROB_LABELS[idx]


def top20_by_calibrated_probability(
    *,
    calibrated: np.ndarray,
    calibrated_ranks: np.ndarray,
    raw_ranks: np.ndarray,
) -> np.ndarray:
    """Site display order: calibrated_prob desc, then ranks/ticket asc."""
    ticket_order = np.arange(calibrated.shape[1], dtype=np.int16)
    rows = []
    for race_i in range(calibrated.shape[0]):
        order = np.lexsort(
            (
                ticket_order,
                raw_ranks[race_i],
                calibrated_ranks[race_i],
                -calibrated[race_i],
            )
        )
        rows.append(order[:20])
    return np.vstack(rows).astype(np.int16)


def validate_site_output(site: pd.DataFrame, *, expected_rows: int) -> None:
    if len(site) != expected_rows:
        raise RuntimeError(f"Output row count mismatch: input={expected_rows}, output={len(site)}")
    prob_cols = [f"ai_top{i}_prob" for i in range(1, 21)]
    ticket_cols = [f"ai_top{i}_ticket" for i in range(1, 21)]
    missing = [col for col in [*prob_cols, *ticket_cols] if col not in site.columns]
    if missing:
        raise RuntimeError(f"Missing site output columns: {missing}")

    probs = site[prob_cols].apply(pd.to_numeric, errors="coerce")
    tickets = site[ticket_cols]
    prob_order_violations = int((probs.diff(axis=1).iloc[:, 1:] > 1e-12).any(axis=1).sum())
    duplicate_ticket_rows = int(tickets.apply(lambda r: len(set(r.dropna())) < len(r.dropna()), axis=1).sum())
    if prob_order_violations:
        raise ValueError(f"ai_top probabilities are not sorted descending: {prob_order_violations} rows")
    if duplicate_ticket_rows:
        raise ValueError(f"duplicate tickets found in ai_top columns: {duplicate_ticket_rows} rows")


def build_site_outputs(
    *,
    source: pd.DataFrame,
    future: pd.DataFrame,
    ranking_proba: np.ndarray,
    calibrated: np.ndarray,
    rough_idx: np.ndarray,
    rough_detail: pd.DataFrame,
    details: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_ranks = rank_matrix(ranking_proba)
    calibrated_ranks = rank_matrix(calibrated)
    top20 = top20_by_calibrated_probability(
        calibrated=calibrated,
        calibrated_ranks=calibrated_ranks,
        raw_ranks=raw_ranks,
    )
    dates = date_values(source).reset_index(drop=True)

    site_rows: list[dict[str, object]] = []
    internal_rows: list[dict[str, object]] = []
    for race_i, ticket_indices in enumerate(top20):
        race = future.iloc[race_i]
        race_id = f"site_{race_i:06d}"
        roughness_bin = ROUGHNESS_LABELS[int(rough_idx[race_i])]
        uses_roughness = bool(details["use_roughness"][race_i])
        method_used = (
            "roughness_prob_bucket_ticket_pattern_roughness"
            if uses_roughness
            else "traditional_prob_bucket_ticket_pattern"
        )
        row: dict[str, object] = {
            "date": dates.iloc[race_i],
            "jcd": str(race["jcd"]).zfill(2),
            "r": int(race["r"]),
            "roughness_score": float(rough_detail.loc[race_i, "roughness_score"]),
            "roughness_bin": roughness_bin,
            "probability_label": DISPLAY_LABEL,
            "probability_note": DISPLAY_NOTE,
        }
        seen: set[str] = set()
        for display_rank, ticket_idx in enumerate(ticket_indices, start=1):
            ticket_idx = int(ticket_idx)
            ticket = INDEX_TO_TRIFECTA[ticket_idx]
            if ticket in seen:
                raise RuntimeError(f"Duplicate ticket in Top20 for race row {race_i}: {ticket}")
            seen.add(ticket)
            raw_prob = float(ranking_proba[race_i, ticket_idx])
            calibrated_prob = float(calibrated[race_i, ticket_idx])
            row[f"ai_top{display_rank}_ticket"] = ticket
            row[f"ai_top{display_rank}_prob"] = calibrated_prob

            if uses_roughness:
                rough_local = bool(details["roughness_uses_local_cell"][race_i, ticket_idx])
                calibration_source = (
                    "roughness_prob_bucket_ticket_pattern_roughness"
                    if rough_local
                    else "roughness_fallback_prob_bucket_ticket_pattern"
                )
            else:
                calibration_source = "traditional_prob_bucket_ticket_pattern"
            internal_rows.append(
                {
                    "race_id": race_id,
                    "date": dates.iloc[race_i],
                    "jcd": str(race["jcd"]).zfill(2),
                    "r": int(race["r"]),
                    "model_name": MODEL_OUTER,
                    "calibration_strategy": "Q1_fallback",
                    "display_rank": display_rank,
                    "ticket": ticket,
                    "ai_prob": calibrated_prob,
                    "raw_prob": raw_prob,
                    "calibrated_prob": calibrated_prob,
                    "raw_rank": int(raw_ranks[race_i, ticket_idx]),
                    "calibrated_rank": int(calibrated_ranks[race_i, ticket_idx]),
                    "ticket_pattern": ticket_pattern(ticket_idx),
                    "prob_bucket": prob_bucket(raw_prob),
                    "roughness_score": float(rough_detail.loc[race_i, "roughness_score"]),
                    "roughness_bin": roughness_bin,
                    "calibration_method_used": method_used,
                    "calibration_source": calibration_source,
                    "calibration_lift": float(details["lift"][race_i, ticket_idx]),
                    "calibration_n": int(details["calibration_n"][race_i, ticket_idx]),
                }
            )
        if len(seen) != 20:
            raise RuntimeError(f"Top20 count is not 20 for race row {race_i}")
        site_rows.append(row)

    site = pd.DataFrame(site_rows)
    internal = pd.DataFrame(internal_rows)
    for rank in range(1, 21):
        for suffix in ["ticket", "prob"]:
            col = f"ai_top{rank}_{suffix}"
            if col not in site.columns:
                raise RuntimeError(f"Missing site output column: {col}")
    return site, internal


def internal_output_path(output: str | Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    output_path = Path(output)
    return output_path.with_name(f"{output_path.stem}_internal{output_path.suffix}")


def main() -> None:
    args = parse_args()
    source = read_future_races(args.input)
    future = add_basic_features(source).reset_index(drop=True)

    raw_direct = direct120_predict_from_artifact(future, args.artifacts_dir)
    ranking_proba, calibration_table, traditional_model, roughness_model = load_or_build_calibration(
        args,
        raw_direct,
        future,
    )
    calibrated, rough_idx, rough_detail, details = apply_q1_fallback(
        df=future,
        raw=ranking_proba,
        traditional_model=traditional_model,
        roughness_model=roughness_model,
        calibration_table=calibration_table,
    )
    site, internal = build_site_outputs(
        source=source,
        future=future,
        ranking_proba=ranking_proba,
        calibrated=calibrated,
        rough_idx=rough_idx,
        rough_detail=rough_detail,
        details=details,
    )

    knn_df = compute_knn_columns(future, args.artifacts_dir)
    if knn_df is not None:
        site = pd.concat([site, knn_df], axis=1)

    forbidden = ["raw", "calibration_lift", "calibration_source", "calibration_n", "actual_ticket", "3rt", "logloss", "brier", "ece", "ev"]
    bad_columns = [col for col in site.columns if any(term in col.lower() for term in forbidden)]
    if bad_columns:
        raise RuntimeError(f"Forbidden columns found in site output: {bad_columns}")
    validate_site_output(site, expected_rows=len(source))

    output_path = Path(args.output)
    audit_path = internal_output_path(args.output, args.internal_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    site.to_csv(output_path, index=False, encoding="utf-8-sig")
    internal.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path.resolve()}")
    print(f"Wrote {audit_path.resolve()}")


if __name__ == "__main__":
    main()
