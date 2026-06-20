from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BOATS, REGISTRATION_COLS, WINRATE_COLS, normalize_registration_series


BASE_RACE_COLUMNS = ["jcd", "r"] + WINRATE_COLS + ["r1", "r2", "r3", "3rt"]
RECENT_RACE_COLUMNS = BASE_RACE_COLUMNS + REGISTRATION_COLS


@dataclass
class DataQualityReport:
    name: str
    rows_before: int
    rows_after: int
    missing_cells: int
    invalid_target_rows: int
    duplicate_rows: int
    notes: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "missing_cells": self.missing_cells,
            "invalid_target_rows": self.invalid_target_rows,
            "duplicate_rows": self.duplicate_rows,
            "notes": " | ".join(self.notes),
        }


def _read_csv_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _valid_target_mask(df: pd.DataFrame) -> pd.Series:
    target = df[["r1", "r2", "r3"]].apply(pd.to_numeric, errors="coerce")
    in_range = target.apply(lambda col: col.between(1, 6)).all(axis=1)
    unique = target.nunique(axis=1) == 3
    return in_range & unique


def load_race_data(
    path: str | Path,
    *,
    require_registrations: bool,
    name: str,
) -> tuple[pd.DataFrame, DataQualityReport]:
    df = _read_csv_auto(path)
    rows_before = len(df)
    required = RECENT_RACE_COLUMNS if require_registrations else BASE_RACE_COLUMNS
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{name}: required columns are missing: {missing_columns}")

    df = df.copy()
    df["jcd"] = df["jcd"].astype("string").str.strip().str.zfill(2)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("Int64")
    df = _coerce_numeric(df, WINRATE_COLS + ["r1", "r2", "r3", "3rt"])
    for col in REGISTRATION_COLS:
        if col in df.columns:
            df[col] = normalize_registration_series(df[col])

    invalid_mask = ~_valid_target_mask(df)
    invalid_target_rows = int(invalid_mask.sum())
    if invalid_target_rows:
        df = df.loc[~invalid_mask].copy()

    for col in ["r1", "r2", "r3"]:
        df[col] = df[col].astype(int)

    missing_cells = int(df[required].isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    notes: list[str] = []

    for col in WINRATE_COLS:
        abnormal = int((df[col].notna() & ~df[col].between(0, 15)).sum())
        if abnormal:
            notes.append(f"{col}: {abnormal} values outside 0-15")

    if df["r"].isna().any():
        notes.append("race number r has missing values")
    if df["jcd"].isna().any():
        notes.append("jcd has missing values")

    report = DataQualityReport(
        name=name,
        rows_before=rows_before,
        rows_after=len(df),
        missing_cells=missing_cells,
        invalid_target_rows=invalid_target_rows,
        duplicate_rows=duplicate_rows,
        notes=notes,
    )
    return df.reset_index(drop=True), report


def load_racer_master(path: str | Path) -> pd.DataFrame:
    master = _read_csv_auto(path)
    if "登録番号" not in master.columns:
        raise ValueError("racer_master: 登録番号 column is missing")

    master = master.copy()
    master["登録番号"] = normalize_registration_series(master["登録番号"])
    if "更新日" in master.columns:
        master["更新日"] = pd.to_datetime(master["更新日"], errors="coerce")

    metric_cols = [col for col in master.columns if "コース_" in col]
    for col in metric_cols:
        raw = master[col].astype("string").str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        master[col] = pd.to_numeric(raw, errors="coerce")

    # racer_master の「率」は 33.3 のような百分率なので、0.333 にそろえる。
    rate_cols = [col for col in metric_cols if col.endswith("率")]
    for col in rate_cols:
        median = pd.to_numeric(master[col], errors="coerce").median()
        if np.isfinite(median) and median > 1.5:
            master[col] = master[col] / 100.0

    if "更新日" in master.columns:
        master = master.sort_values(["登録番号", "更新日"]).drop_duplicates("登録番号", keep="last")
    else:
        master = master.drop_duplicates("登録番号", keep="last")
    return master.reset_index(drop=True)


def save_quality_reports(reports: list[DataQualityReport], output_path: str | Path) -> None:
    rows = [report.as_dict() for report in reports]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def split_recent_train_validation(
    df: pd.DataFrame,
    *,
    date_column: str | None = None,
    validation_fraction: float = 31.0 / 92.0,
    strict_calendar_split: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """2026年3-4月相当の学習データと2026年5月相当の検証データに分ける。

    入力CSVに日付列がある場合は暦月で厳密に分割する。今回指定された列一覧には
    日付列がないため、通常は各場(jcd)の行順が時系列であるという前提で、末尾
    31/92を5月相当の検証データとして使う。この近似は6月テストを使わないため
    テストリークではないが、厳密な月分割ではない。
    """
    candidates = [
        date_column,
        "date",
        "race_date",
        "日付",
        "開催日",
        "年月日",
        "ymd",
    ]
    candidates = [col for col in candidates if col]
    chosen = next((col for col in candidates if col in df.columns), None)

    if chosen is not None:
        dates = pd.to_datetime(df[chosen], errors="coerce")
        train_mask = dates.dt.to_period("M").isin([pd.Period("2026-03"), pd.Period("2026-04")])
        valid_mask = dates.dt.to_period("M").eq(pd.Period("2026-05"))
        train = df.loc[train_mask].copy()
        valid = df.loc[valid_mask].copy()
        if train.empty or valid.empty:
            raise ValueError(
                f"{chosen} で2026-03/04学習・2026-05検証に分割できませんでした。"
            )
        report = pd.DataFrame(
            [
                {
                    "split_method": "calendar_month",
                    "date_column": chosen,
                    "train_rows": len(train),
                    "validation_rows": len(valid),
                    "validation_fraction": len(valid) / max(len(df), 1),
                    "notes": "exact calendar split",
                }
            ]
        )
        return train.reset_index(drop=True), valid.reset_index(drop=True), report

    if strict_calendar_split:
        raise ValueError(
            "dataset_2_recent_202603_202605.csv に日付列がないため、"
            "2026年3-4月/5月の厳密な暦月分割ができません。"
        )

    train_parts = []
    valid_parts = []
    rows = []
    for jcd, group in df.groupby("jcd", sort=False, dropna=False):
        group = group.copy()
        cutoff = int(round(len(group) * (1.0 - validation_fraction)))
        cutoff = min(max(cutoff, 1), len(group) - 1)
        train_part = group.iloc[:cutoff]
        valid_part = group.iloc[cutoff:]
        train_parts.append(train_part)
        valid_parts.append(valid_part)
        rows.append(
            {
                "split_method": "row_order_within_jcd_assumption",
                "date_column": "",
                "jcd": jcd,
                "train_rows": len(train_part),
                "validation_rows": len(valid_part),
                "validation_fraction": len(valid_part) / max(len(group), 1),
                "notes": "no date column; assumes rows are chronological within jcd",
            }
        )

    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    report = pd.DataFrame(rows)
    return train, valid, report
