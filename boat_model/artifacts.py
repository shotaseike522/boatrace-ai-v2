from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    BOATS,
    INDEX_TO_TRIFECTA,
    TRIFECTA_PERMUTATIONS,
    normalize_probability_matrix,
    top_prediction_strings,
)


ROUGHNESS_LABELS = [
    "Q1_low_roughness",
    "Q2",
    "Q3",
    "Q4",
    "Q5_high_roughness",
]


def _joblib():
    try:
        import joblib  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("joblib is required. Install it with: pip install -r requirements.txt") from exc
    return joblib


def ensure_artifact_dirs(root: str | Path = "artifacts") -> tuple[Path, Path]:
    root_path = Path(root)
    models_dir = root_path / "models"
    knn_dir = root_path / "knn"
    models_dir.mkdir(parents=True, exist_ok=True)
    knn_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, knn_dir


def save_direct_model(model: Any, models_dir: str | Path) -> Path:
    path = Path(models_dir) / "lightgbm_direct120.joblib"
    _joblib().dump(
        {
            "model_name": model.model_name,
            "model_object": model,
            "feature_columns": model.feature_columns_,
            "output": "trifecta_120_probability",
        },
        path,
    )
    return path


def save_position_models(model: Any, models_dir: str | Path) -> list[Path]:
    paths = []
    for position, classifier in enumerate(model.classifiers_, start=1):
        path = Path(models_dir) / f"lightgbm_position{position}.joblib"
        _joblib().dump(
            {
                "model_name": model.model_name,
                "position": position,
                "encoder": model.encoder_,
                "classifier": classifier,
                "feature_columns": model.feature_columns_,
                "output": f"position_{position}_boat_probability",
            },
            path,
        )
        paths.append(path)
    return paths


def save_pairwise_model(model: Any, models_dir: str | Path) -> Path:
    path = Path(models_dir) / "pairwise_model.joblib"
    _joblib().dump(
        {
            "model_name": model.model_name,
            "model_object": model,
            "feature_columns": model.feature_columns_,
            "output": "trifecta_120_probability",
            "include_racer_master": model.include_racer_master,
        },
        path,
    )
    return path


def compute_roughness_score(df: pd.DataFrame) -> pd.Series:
    def pattern_inversion_rate(pattern: object) -> float:
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

    work = df.copy()
    if "勝率1_minus_勝率2" not in work.columns:
        work["勝率1_minus_勝率2"] = work["勝率1"] - work["勝率2"]
    if "勝率1_minus_勝率3" not in work.columns:
        work["勝率1_minus_勝率3"] = work["勝率1"] - work["勝率3"]
    pattern_inversion = (
        work["勝率順位パターン"].map(pattern_inversion_rate)
        if "勝率順位パターン" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    components = pd.DataFrame(
        {
            "boat1_vs_2_weak": (-pd.to_numeric(work["勝率1_minus_勝率2"], errors="coerce")).rank(pct=True),
            "boat1_vs_3_weak": (-pd.to_numeric(work["勝率1_minus_勝率3"], errors="coerce")).rank(pct=True),
            "winrate_close": (-pd.to_numeric(work["勝率標準偏差"], errors="coerce")).rank(pct=True),
            "boat1_not_top": (1 - pd.to_numeric(work["1号艇が勝率1位か"], errors="coerce")).rank(pct=True),
            "pattern_disorder": pd.to_numeric(pattern_inversion, errors="coerce").rank(pct=True),
        }
    ).fillna(0.5)
    return components.mean(axis=1)


def roughness_thresholds(df: pd.DataFrame) -> list[float]:
    score = compute_roughness_score(df)
    return [float(score.quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]


def assign_roughness_bins(score: pd.Series, thresholds: list[float]) -> pd.Series:
    bins = [-np.inf] + list(thresholds) + [np.inf]
    return pd.cut(score, bins=bins, labels=ROUGHNESS_LABELS, include_lowest=True)


def save_roughness_artifact(df: pd.DataFrame, artifacts_root: str | Path) -> Path:
    root = Path(artifacts_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "roughness_thresholds.json"
    path.write_text(
        json.dumps(
            {
                "labels": ROUGHNESS_LABELS,
                "thresholds": roughness_thresholds(df),
                "note": "Quantile thresholds learned from training data. Higher score means more roughness.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def save_knn_artifacts(knn_model: Any, train_df: pd.DataFrame, knn_dir: str | Path) -> list[Path]:
    knn_path = Path(knn_dir)
    knn_path.mkdir(parents=True, exist_ok=True)
    paths = []
    np.save(knn_path / "past_features.npy", knn_model.X_)
    np.save(knn_path / "past_labels.npy", knn_model.labels_)
    paths.extend([knn_path / "past_features.npy", knn_path / "past_labels.npy"])

    feature_stats = {
        "feature_columns": knn_model.feature_columns_,
        "mean": knn_model.encoder_.means_.tolist(),
        "std": knn_model.encoder_.stds_.tolist(),
        "median": knn_model.encoder_.medians_.tolist(),
        "k": knn_model.k,
        "weighted": knn_model.weighted,
        "output": "trifecta_120_probability",
    }
    stats_path = knn_path / "feature_stats.json"
    stats_path.write_text(json.dumps(feature_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(stats_path)

    race_info = train_df[["jcd", "r", "r1", "r2", "r3", "3rt"]].copy()
    race_info["trifecta"] = race_info[["r1", "r2", "r3"]].astype(str).agg("-".join, axis=1)
    race_info.to_csv(knn_path / "past_race_info.csv", index=False, encoding="utf-8-sig")
    paths.append(knn_path / "past_race_info.csv")
    return paths


def load_joblib_artifact(path: str | Path) -> Any:
    return _joblib().load(path)


def load_knn_artifacts(knn_dir: str | Path) -> dict[str, Any]:
    knn_path = Path(knn_dir)
    stats = json.loads((knn_path / "feature_stats.json").read_text(encoding="utf-8"))
    race_info = pd.read_csv(knn_path / "past_race_info.csv", encoding="utf-8-sig")
    jcd_values = race_info["jcd"].astype(str).str.zfill(2).to_numpy()
    r_values = pd.to_numeric(race_info["r"], errors="coerce").astype(int).to_numpy()
    all_indices = np.arange(len(race_info), dtype=np.int64)
    group_jcd_r: dict[tuple[str, int], np.ndarray] = {}
    group_jcd: dict[str, np.ndarray] = {}
    group_r: dict[int, np.ndarray] = {}
    for idx, (jcd, race_no) in enumerate(zip(jcd_values, r_values)):
        group_jcd_r.setdefault((jcd, int(race_no)), []).append(idx)
        group_jcd.setdefault(jcd, []).append(idx)
        group_r.setdefault(int(race_no), []).append(idx)
    group_jcd_r = {key: np.asarray(value, dtype=np.int64) for key, value in group_jcd_r.items()}
    group_jcd = {key: np.asarray(value, dtype=np.int64) for key, value in group_jcd.items()}
    group_r = {key: np.asarray(value, dtype=np.int64) for key, value in group_r.items()}
    return {
        "feature_columns": stats["feature_columns"],
        "mean": np.asarray(stats["mean"], dtype=float),
        "std": np.asarray(stats["std"], dtype=float),
        "median": np.asarray(stats["median"], dtype=float),
        "k": int(stats.get("k", 500)),
        "weighted": bool(stats.get("weighted", False)),
        "past_features": np.load(knn_path / "past_features.npy"),
        "past_labels": np.load(knn_path / "past_labels.npy"),
        "race_info": race_info,
        "group_jcd_r": group_jcd_r,
        "group_jcd": group_jcd,
        "group_r": group_r,
        "all_indices": all_indices,
    }


def transform_knn_features(df: pd.DataFrame, knn_artifact: dict[str, Any]) -> np.ndarray:
    columns = knn_artifact["feature_columns"]
    matrix = []
    for col in columns:
        matrix.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    X = np.column_stack(matrix)
    X = np.where(np.isfinite(X), X, knn_artifact["median"])
    return ((X - knn_artifact["mean"]) / knn_artifact["std"]).astype(np.float32)


def _candidate_indices_for_knn(row: pd.Series, race_info: pd.DataFrame, k: int) -> np.ndarray:
    jcd = str(row["jcd"]).zfill(2)
    race_no = int(row["r"])
    if isinstance(race_info, dict):
        both = race_info["group_jcd_r"].get((jcd, race_no), np.asarray([], dtype=np.int64))
        if len(both) >= k:
            return both
        jcd_only = race_info["group_jcd"].get(jcd, np.asarray([], dtype=np.int64))
        r_only = race_info["group_r"].get(race_no, np.asarray([], dtype=np.int64))
        merged = np.unique(np.concatenate([both, jcd_only, r_only]))
        if len(merged) >= k:
            return merged
        return race_info["all_indices"]

    jcd_values = race_info["jcd"].astype(str).str.zfill(2)
    r_values = pd.to_numeric(race_info["r"], errors="coerce").astype(int)
    both = np.flatnonzero((jcd_values == jcd).to_numpy() & (r_values == race_no).to_numpy())
    if len(both) >= k:
        return both
    jcd_only = np.flatnonzero((jcd_values == jcd).to_numpy())
    r_only = np.flatnonzero((r_values == race_no).to_numpy())
    merged = np.unique(np.concatenate([both, jcd_only, r_only]))
    if len(merged) >= k:
        return merged
    return np.arange(len(race_info))


def knn_predict_proba_and_neighbors(
    df: pd.DataFrame,
    knn_artifact: dict[str, Any],
    *,
    k: int | None = None,
    diagnostic_k: int = 100,
) -> tuple[np.ndarray, list[np.ndarray]]:
    k = int(k or knn_artifact["k"])
    X_eval = transform_knn_features(df, knn_artifact)
    X_train = knn_artifact["past_features"].astype(np.float32)
    labels = knn_artifact["past_labels"]
    race_info = {
        "group_jcd_r": knn_artifact["group_jcd_r"],
        "group_jcd": knn_artifact["group_jcd"],
        "group_r": knn_artifact["group_r"],
        "all_indices": knn_artifact["all_indices"],
    }
    out = np.zeros((len(df), len(TRIFECTA_PERMUTATIONS)), dtype=float)
    diagnostic_neighbors = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        candidates = _candidate_indices_for_knn(row, race_info, k)
        diff = X_train[candidates] - X_eval[row_idx]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        needed = max(k, diagnostic_k)
        if len(candidates) > needed:
            local = np.argpartition(dist, needed - 1)[:needed]
        else:
            local = np.arange(len(candidates))
        nearest = candidates[local]
        nearest_dist = dist[local]
        order = np.argsort(nearest_dist)
        nearest = nearest[order]
        nearest_dist = nearest_dist[order]

        model_nearest = nearest[: min(k, len(nearest))]
        model_dist = nearest_dist[: min(k, len(nearest))]
        if knn_artifact.get("weighted"):
            weights = 1.0 / np.maximum(model_dist, 1e-6)
            weights = np.minimum(weights, 1e6)
        else:
            weights = np.ones_like(model_dist, dtype=float)
        counts = np.bincount(labels[model_nearest], weights=weights, minlength=len(TRIFECTA_PERMUTATIONS)).astype(float)
        counts += 0.2
        out[row_idx] = counts / counts.sum()
        diagnostic_neighbors.append(nearest[: min(diagnostic_k, len(nearest))])
    return normalize_probability_matrix(out), diagnostic_neighbors


def position_artifacts_predict_proba(df: pd.DataFrame, position_artifacts: list[dict[str, Any]]) -> np.ndarray:
    p_list = []
    for artifact in position_artifacts:
        X = artifact["encoder"].transform(df)
        p_list.append(artifact["classifier"].predict_proba(X))
    p1, p2, p3 = p_list
    out = np.zeros((len(df), len(TRIFECTA_PERMUTATIONS)), dtype=float)
    for idx, (first, second, third) in enumerate(TRIFECTA_PERMUTATIONS):
        out[:, idx] = p1[:, first - 1] * p2[:, second - 1] * p3[:, third - 1]
    return normalize_probability_matrix(out)


def top1_labels(proba: np.ndarray) -> list[str]:
    top = np.argmax(proba, axis=1)
    return [INDEX_TO_TRIFECTA[int(idx)] for idx in top]


def probability_summary_columns(prefix: str, proba: np.ndarray, top_n: int = 5) -> dict[str, list[Any]]:
    top = np.argmax(proba, axis=1)
    return {
        f"{prefix}_top1": top1_labels(proba),
        f"{prefix}_top1_probability": [float(proba[i, top[i]]) for i in range(len(top))],
        f"{prefix}_top{top_n}": top_prediction_strings(proba, top_n=top_n, with_probability=True),
    }


def similar_race_diagnostics(knn_artifact: dict[str, Any], neighbors: list[np.ndarray]) -> pd.DataFrame:
    """近似100レースの配当分布・着順発生率を仕様書(model_inventory.md)準拠の列名で出力する。

    AI予想(モデル予測)とは別枠の「過去事実」として扱うため、列名はすべて
    ``similar_`` プレフィックスで統一し、AI予想側の ``ai_`` プレフィックスとは
    混ざらないようにしている。
    """
    race_info = knn_artifact["race_info"].reset_index(drop=True)
    top_n = 10
    rows = []
    for indices in neighbors:
        similar = race_info.iloc[indices].copy()
        payouts = pd.to_numeric(similar["3rt"], errors="coerce").fillna(0)
        n = len(similar)

        result_counts = similar["trifecta"].value_counts()
        top_tickets = result_counts.head(top_n)

        row: dict[str, Any] = {
            "similar_count": int(n),
            "similar_avg_payout": float(payouts.mean()) if n else 0.0,
            "similar_median_payout": float(payouts.median()) if n else 0.0,
            "similar_max_payout": int(payouts.max()) if n else 0,
            "similar_under_1000_count": int((payouts < 1000).sum()),
            "similar_1000_2999_count": int(((payouts >= 1000) & (payouts < 3000)).sum()),
            "similar_3000_9999_count": int(((payouts >= 3000) & (payouts < 10000)).sum()),
            "similar_over_10000_count": int((payouts >= 10000).sum()),
            "similar_under_1000_rate": float((payouts < 1000).mean()) if n else 0.0,
            "similar_1000_2999_rate": float(((payouts >= 1000) & (payouts < 3000)).mean()) if n else 0.0,
            "similar_3000_9999_rate": float(((payouts >= 3000) & (payouts < 10000)).mean()) if n else 0.0,
            "similar_over_10000_rate": float((payouts >= 10000).mean()) if n else 0.0,
        }

        tickets = list(top_tickets.items())
        for rank in range(1, top_n + 1):
            prefix = f"similar_rank{rank}"
            if rank - 1 < len(tickets):
                ticket, count = tickets[rank - 1]
                row[f"{prefix}_ticket"] = ticket
                row[f"{prefix}_count"] = int(count)
                row[f"{prefix}_rate"] = float(count) / n if n else 0.0
            else:
                row[f"{prefix}_ticket"] = ""
                row[f"{prefix}_count"] = 0
                row[f"{prefix}_rate"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def assign_marks(
    agreement_count: np.ndarray,
    ai_rank: np.ndarray,
) -> list[str]:
    """仕様書の予想印ルールを適用する。

    ◎: 3モデル(direct120 / position6 / blend)すべてのTop5が一致
    ○: 2モデル以上のTop5が一致
    ▲: AI総合Top5(印なしの場合)
    △: AI総合Top10(印なしの場合)
    """
    marks = []
    for count, rank in zip(agreement_count, ai_rank):
        if count >= 3:
            marks.append("\u25ce")
        elif count >= 2:
            marks.append("\u25cb")
        elif rank <= 5:
            marks.append("\u25b2")
        elif rank <= 10:
            marks.append("\u25b3")
        else:
            marks.append("")
    return marks


def build_ai_top10_table(
    ai_proba: np.ndarray,
    direct_proba: np.ndarray,
    position_proba: np.ndarray,
    blend_proba: np.ndarray,
    *,
    top_n: int = 10,
    agreement_top_n: int = 5,
) -> pd.DataFrame:
    """AI予想Top10を、各サブモデルの順位・一致数・予想印つきで横持ち展開する。

    AI予想 = モデルによる予測、という思想に沿い、ここで作る列はすべて
    ``ai_`` プレフィックスにそろえ、近似100レース由来の ``similar_`` 列とは
    明確に分離する。
    """
    n_races, n_tickets = ai_proba.shape

    def rank_matrix(proba: np.ndarray) -> np.ndarray:
        order = np.argsort(-proba, axis=1)
        ranks = np.empty_like(order)
        row_idx = np.arange(proba.shape[0])[:, None]
        ranks[row_idx, order] = np.arange(1, proba.shape[1] + 1)
        return ranks

    direct_rank = rank_matrix(direct_proba)
    position_rank = rank_matrix(position_proba)
    blend_rank = rank_matrix(blend_proba)

    ai_order = np.argsort(-ai_proba, axis=1)

    columns: dict[str, list[Any]] = {}
    for rank in range(1, top_n + 1):
        prefix = f"ai_top{rank}"
        tickets, probs = [], []
        direct_ranks, position_ranks, blend_ranks = [], [], []
        top5_agree, top10_agree = [], []

        for race_idx in range(n_races):
            ticket_idx = int(ai_order[race_idx, rank - 1])
            tickets.append(INDEX_TO_TRIFECTA[ticket_idx])
            probs.append(float(ai_proba[race_idx, ticket_idx]))

            d_rank = int(direct_rank[race_idx, ticket_idx])
            p_rank = int(position_rank[race_idx, ticket_idx])
            b_rank = int(blend_rank[race_idx, ticket_idx])
            direct_ranks.append(d_rank)
            position_ranks.append(p_rank)
            blend_ranks.append(b_rank)

            agree5 = sum(r <= agreement_top_n for r in (d_rank, p_rank, b_rank))
            agree10 = sum(r <= 10 for r in (d_rank, p_rank, b_rank))
            top5_agree.append(agree5)
            top10_agree.append(agree10)

        marks_for_rank = assign_marks(np.asarray(top5_agree), np.full(n_races, rank))

        columns[f"{prefix}_ticket"] = tickets
        columns[f"{prefix}_prob"] = probs
        columns[f"{prefix}_mark"] = marks_for_rank
        columns[f"{prefix}_direct120_rank"] = direct_ranks
        columns[f"{prefix}_position6_rank"] = position_ranks
        columns[f"{prefix}_blend_rank"] = blend_ranks
        columns[f"{prefix}_top5_agreement_count"] = top5_agree
        columns[f"{prefix}_top10_agreement_count"] = top10_agree

    return pd.DataFrame(columns)


def compute_quinella_top3(ai_proba: np.ndarray, top_n: int = 3) -> pd.DataFrame:
    """3連単120通りの確率から2連複（1・2着を順不同）の上位3点を計算する。

    2連複「A-B」の確率 = 3連単のうち1着A・2着B または 1着B・2着A となる全通りの合計。
    例: 2連複「1-2」= 1-2-3 + 1-2-4 + 1-2-5 + 1-2-6 + 2-1-3 + 2-1-4 + 2-1-5 + 2-1-6 の合計。

    Returns
    -------
    pd.DataFrame
        quinella_top1_pair / quinella_top1_prob
        quinella_top2_pair / quinella_top2_prob
        quinella_top3_pair / quinella_top3_prob
    """
    from itertools import combinations

    # 2連複の全ペア（15通り）を定義し、対応する3連単インデックスを事前計算
    quinella_pairs = list(combinations(BOATS, 2))  # (1,2), (1,3), ..., (5,6)
    pair_to_trifecta_indices: dict[tuple[int, int], list[int]] = {pair: [] for pair in quinella_pairs}
    for idx, (first, second, third) in enumerate(TRIFECTA_PERMUTATIONS):
        key = tuple(sorted([first, second]))
        if key in pair_to_trifecta_indices:
            pair_to_trifecta_indices[key].append(idx)

    n_races = ai_proba.shape[0]
    columns: dict[str, list] = {}

    for rank in range(1, top_n + 1):
        columns[f"quinella_top{rank}_pair"] = []
        columns[f"quinella_top{rank}_prob"] = []

    for race_idx in range(n_races):
        race_proba = ai_proba[race_idx]
        # 各2連複ペアの確率を合算
        quinella_probs = {
            pair: float(race_proba[indices].sum())
            for pair, indices in pair_to_trifecta_indices.items()
        }
        sorted_pairs = sorted(quinella_probs.items(), key=lambda x: x[1], reverse=True)

        for rank in range(1, top_n + 1):
            if rank - 1 < len(sorted_pairs):
                pair, prob = sorted_pairs[rank - 1]
                columns[f"quinella_top{rank}_pair"].append(f"{pair[0]}-{pair[1]}")
                columns[f"quinella_top{rank}_prob"].append(round(prob, 6))
            else:
                columns[f"quinella_top{rank}_pair"].append("")
                columns[f"quinella_top{rank}_prob"].append(float("nan"))

    return pd.DataFrame(columns)

