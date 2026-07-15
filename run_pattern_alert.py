"""AI予想Top5 と 近似100レースTop5 の一致パターンを検出し、LINE配信用の
小さなJSONファイル(outputs/pattern_alert_YYYYMMDD.json)を出力する。

パターン: AI予想Top5(ai_top1-5)と近似100レースTop5(similar_rank1-5)の
計10チケット全てに共通する2艇ペアがあり、かつ最も多く出てきた3艇の
組み合わせ(最頻組み合わせ)の一致数が6以上のレースを抽出する。

このスクリプトは通常AI予想（run_site_predictions_calibrated.py）や
サイト表示には一切影響しない。読み取り専用で、失敗しても他の日次処理を
止めないよう常に空配列を書き出すフォールバックを持つ。

実行例:
    python run_pattern_alert.py --date 20260716
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytz
from datetime import datetime

VENUES_NAME_ONLY = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}
MIN_MATCH_COUNT = 6
OUTPUTS_DIR = "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect AI/KNN Top5 consensus pattern for LINE alert.")
    jst = pytz.timezone("Asia/Tokyo")
    default_date = datetime.now(jst).strftime("%Y%m%d")
    parser.add_argument("--date", default=default_date, help="対象日付(YYYYMMDD)。デフォルトは本日(JST)。")
    parser.add_argument("--output", default="", help="出力先。デフォルトは outputs/pattern_alert_{date}.json")
    parser.add_argument("--min-count", type=int, default=MIN_MATCH_COUNT)
    return parser.parse_args()


def ticket_set(ticket) -> frozenset | None:
    if pd.isna(ticket) or ticket == "":
        return None
    return frozenset(int(x) for x in str(ticket).split("-"))


def detect_pattern(pred: pd.DataFrame, min_count: int) -> list[dict]:
    all_pairs = [frozenset(c) for c in combinations(range(1, 7), 2)]
    alerts: list[dict] = []

    for _, row in pred.iterrows():
        tickets = [ticket_set(row.get(f"ai_top{r}_ticket", "")) for r in range(1, 6)]
        tickets += [ticket_set(row.get(f"similar_rank{r}_ticket", "")) for r in range(1, 6)]
        if any(t is None for t in tickets):
            continue

        common_pairs = [p for p in all_pairs if all(p.issubset(t) for t in tickets)]
        if not common_pairs:
            continue

        combo, count = Counter(tickets).most_common(1)[0]
        if count < min_count:
            continue

        jcd = str(row["jcd"]).zfill(2)
        alerts.append(
            {
                "jcd": jcd,
                "venue": VENUES_NAME_ONLY.get(jcd, jcd),
                "r": int(row["r"]),
                "ticket": "-".join(str(b) for b in sorted(combo)),
                "count": int(count),
            }
        )
    alerts.sort(key=lambda a: (a["jcd"], a["r"]))
    return alerts


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path(OUTPUTS_DIR) / f"pattern_alert_{args.date}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_path = Path(OUTPUTS_DIR) / f"site_predictions_{args.date}.csv"
    if not pred_path.exists():
        print(f"⚠️ {pred_path} が見つからないため、空の結果を出力します。")
        output_path.write_text("[]", encoding="utf-8")
        return

    try:
        pred = pd.read_csv(pred_path, dtype={"jcd": str})
        required = ["ai_top1_ticket", "similar_rank1_ticket"]
        missing = [c for c in required if c not in pred.columns]
        if missing:
            print(f"⚠️ 必要な列がありません({missing})。空の結果を出力します。")
            output_path.write_text("[]", encoding="utf-8")
            return
        alerts = detect_pattern(pred, args.min_count)
    except Exception as exc:  # このスクリプトの失敗で他の日次処理を止めないためのフォールバック
        print(f"⚠️ パターン検出でエラーが発生しました: {exc}")
        output_path.write_text("[]", encoding="utf-8")
        return

    output_path.write_text(json.dumps(alerts, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path.resolve()} ({len(alerts)} 件)")


if __name__ == "__main__":
    main()
