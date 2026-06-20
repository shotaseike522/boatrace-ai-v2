from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / "work" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from boat_model.artifacts import (  # noqa: E402
    ensure_artifact_dirs,
    save_direct_model,
    save_knn_artifacts,
    save_pairwise_model,
    save_position_models,
    save_roughness_artifact,
)
from boat_model.data_loader import load_race_data  # noqa: E402
from boat_model.features import add_basic_features  # noqa: E402
from boat_model.models import ApproxRaceKNNModel, ModelCBasicDirect, ModelCBasicPosition, PairwiseRankModel  # noqa: E402


# 学習用CSVは data/ フォルダに置く想定。
# 自分のPC固有のパスではなく、リポジトリ内の相対パスをデフォルトにしている。
DEFAULT_PAST = "data/dataset_1_past_201307_202602.csv"
DEFAULT_RECENT = "data/dataset_2_recent_202603_202605.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train production artifacts for site prediction.")
    parser.add_argument("--past-csv", default=DEFAULT_PAST)
    parser.add_argument("--recent-csv", default=DEFAULT_RECENT)
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    start = time.time()
    artifacts_root = Path(args.artifacts_dir)
    models_dir, knn_dir = ensure_artifact_dirs(artifacts_root)

    for label, path_str in [("--past-csv", args.past_csv), ("--recent-csv", args.recent_csv)]:
        if not Path(path_str).exists():
            raise FileNotFoundError(
                f"学習用CSVが見つかりません: {path_str}\n"
                f"data/ フォルダに学習用CSVを置くか、{label} で正しいパスを指定してください。\n"
                f"例: python run_train_models.py {label} data/your_file.csv"
            )

    print("Loading training data...")
    past_raw, _ = load_race_data(args.past_csv, require_registrations=False, name="dataset_1_past")
    recent_raw, _ = load_race_data(args.recent_csv, require_registrations=True, name="dataset_2_recent")
    past = add_basic_features(past_raw)
    recent = add_basic_features(recent_raw)

    saved_paths: list[str] = []

    print("Training C_lightgbm_direct120...")
    direct = ModelCBasicDirect(prefer_lightgbm=True).fit(past)
    saved_paths.append(str(save_direct_model(direct, models_dir)))

    print("Training C_lightgbm_position6...")
    position = ModelCBasicPosition(prefer_lightgbm=True).fit(past)
    saved_paths.extend(str(path) for path in save_position_models(position, models_dir))

    print("Training P_knn500 artifact...")
    knn = ApproxRaceKNNModel(k=500, weighted=False).fit(past)
    saved_paths.extend(str(path) for path in save_knn_artifacts(knn, past, knn_dir))

    print("Training P_pairwise...")
    pairwise = PairwiseRankModel(include_racer_master=False).fit(recent)
    saved_paths.append(str(save_pairwise_model(pairwise, models_dir)))

    saved_paths.append(str(save_roughness_artifact(past, artifacts_root)))

    metadata = {
        "created_by": "run_train_models.py",
        "past_csv": args.past_csv,
        "recent_csv": args.recent_csv,
        "models": {
            "C_lightgbm_direct120": "artifacts/models/lightgbm_direct120.joblib",
            "C_lightgbm_position6": [
                "artifacts/models/lightgbm_position1.joblib",
                "artifacts/models/lightgbm_position2.joblib",
                "artifacts/models/lightgbm_position3.joblib",
            ],
            "P_knn500": "artifacts/knn/",
            "P_pairwise": "artifacts/models/pairwise_model.joblib",
        },
        "saved_paths": saved_paths,
        "elapsed_seconds": round(time.time() - start, 1),
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Artifacts saved to {artifacts_root}")
    print(f"Finished in {metadata['elapsed_seconds']}s")


if __name__ == "__main__":
    run()

