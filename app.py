"""競艇AI予想サイト (Streamlit)

outputs/site_predictions_YYYYMMDD.csv (run_site_predictions.py の出力)を読み込み、
スマートフォン中心のレイアウトで以下を表示する。

  1. 競艇場を選ぶ（本日開催中のみ）
  2. レースを選ぶ（1〜12、2段グリッド）
  3. 表示中の場・レース
  4. 荒れやすさ（0-100点 + ラベル + 波バー）
  5. AI予想Top5（◎○▲△の予想印つき）
  6. 近似100レースの配当分布
  7. 近似100レースの着順発生率Top5

列名は run_site_predictions.py が出力する仕様(ai_top1〜10_*, similar_*,
roughness_score, roughness_label)に合わせている。
"""

from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd
import pytz
import streamlit as st

VENUES_MAP = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

OUTPUTS_DIR = "outputs"

ROUGHNESS_LABEL_COLOR = {
    "超堅め": "#1C8C5C",
    "堅め": "#1C8C5C",
    "普通": "#5A6B7D",
    "荒れ注意": "#C98A00",
    "波乱含み": "#E2342B",
}

MARK_STYLE = {
    "◎": ("#FFF4DD", "#C98A00"),
    "○": ("#E6F1FB", "#0046AD"),
    "▲": ("#F0F4F8", "#5A6B7D"),
    "△": ("#F0F4F8", "#5A6B7D"),
    "": ("#F0F4F8", "#5A6B7D"),
}


# ====================================================
# スタイル（白ベース + 青アクセント、スマホ最優先）
# ====================================================
def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F5F9FC;
            --surface: #FFFFFF;
            --primary: #0046AD;
            --primary-deep: #00308A;
            --accent: #00A0E9;
            --ink: #1A2433;
            --ink-soft: #5A6B7D;
            --line: #E2E9F0;
        }
        .stApp { background: var(--bg); }

        /* Streamlit標準ヘッダーは高さを潰して透明化し、自前の固定ヘッダーに差し替える */
        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
            min-height: 0;
        }
        header[data-testid="stHeader"] * {
            visibility: hidden;
        }

        /* 自前の固定ヘッダー（スクロールしても画面上部に貼り付く） */
        .fixed-brand-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            background: linear-gradient(180deg, var(--primary-deep) 0%, var(--primary) 100%);
            color: white;
            padding: 14px 20px;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.04em;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        }

        /* 固定ヘッダー分の余白を本文側に確保 */
        .block-container { max-width: 480px; padding-top: 3.2rem; padding-bottom: 2rem; }

        .ai-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .ai-card-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--ink-soft);
            margin-bottom: 10px;
        }
        .race-select-display {
            display: flex;
            align-items: baseline;
            gap: 10px;
            margin: 4px 2px 14px;
        }
        .race-select-display .now-label {
            font-size: 11px;
            color: var(--ink-soft);
            font-weight: 500;
        }
        .race-select-display .venue {
            font-size: 26px;
            font-weight: 900;
            color: var(--primary-deep);
        }
        .race-select-display .rno {
            font-size: 26px;
            font-weight: 700;
            color: var(--primary-deep);
        }

        /* 競艇場グリッド: 3列固定。st.columnsを使わず、コンテナにgridを当てて
           中の st.button 群を強制的に横並びグリッドにする。
           Streamlitはモバイル幅でst.columnsを縦積みに変えてしまうため、この方式を取る。 */
        div[data-testid="stVerticalBlock"]:has(> div.venue-grid-marker) div[data-testid="stVerticalBlock"] {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        div[data-testid="stVerticalBlock"]:has(> div.venue-grid-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
            width: 100%;
        }

        /* レースグリッド: 6列固定（1〜6, 7〜12の2段になる） */
        div[data-testid="stVerticalBlock"]:has(> div.race-grid-marker) div[data-testid="stVerticalBlock"] {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 6px;
        }
        div[data-testid="stVerticalBlock"]:has(> div.race-grid-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
            width: 100%;
        }

        div[data-testid="stButton"] button {
            border-radius: 9px;
            border: 1.5px solid var(--line);
            background: var(--bg);
            color: var(--ink);
            font-weight: 700;
            width: 100%;
            white-space: pre-line;
            line-height: 1.3;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--primary);
            color: var(--primary);
        }
        .venue-btn-active button {
            background: var(--primary) !important;
            border-color: var(--primary) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ====================================================
# データ読み込み
# ====================================================
@st.cache_data(ttl=300)
def load_today_predictions() -> tuple[pd.DataFrame | None, str | None]:
    """outputs/site_predictions_YYYYMMDD.csv のうち最新日付のものを読み込む。"""
    jst = pytz.timezone("Asia/Tokyo")
    today_str = datetime.now(jst).strftime("%Y%m%d")

    candidate = os.path.join(OUTPUTS_DIR, f"site_predictions_{today_str}.csv")
    if os.path.exists(candidate):
        df = pd.read_csv(candidate, dtype={"jcd": str})
        return df, today_str

    pattern = os.path.join(OUTPUTS_DIR, "site_predictions_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    latest = files[-1]
    df = pd.read_csv(latest, dtype={"jcd": str})
    date_label = os.path.basename(latest).replace("site_predictions_", "").replace(".csv", "")
    return df, date_label


# ====================================================
# 画面パーツ
# ====================================================
def render_venue_picker(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-card-title">競艇場を選ぶ（本日開催中）</div>', unsafe_allow_html=True)
    available_jcds = sorted(df["jcd"].astype(str).unique().tolist())

    st.markdown('<div class="venue-grid-marker"></div>', unsafe_allow_html=True)
    for jcd in available_jcds:
        venue_name = VENUES_MAP.get(jcd, jcd)
        is_active = st.session_state["target_jcd"] == jcd
        if is_active:
            st.markdown('<div class="venue-btn-active">', unsafe_allow_html=True)
        if st.button(f"{jcd}\n{venue_name}", key=f"venue_{jcd}", use_container_width=True):
            st.session_state["target_jcd"] = jcd
            st.session_state["target_rno"] = None
            st.rerun()
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)


def render_race_picker(df: pd.DataFrame) -> None:
    if not st.session_state["target_jcd"]:
        return
    st.markdown('<div class="ai-card-title">レースを選ぶ</div>', unsafe_allow_html=True)
    venue_races = sorted(
        df.loc[df["jcd"].astype(str) == st.session_state["target_jcd"], "r"].astype(int).unique().tolist()
    )

    st.markdown('<div class="race-grid-marker"></div>', unsafe_allow_html=True)
    for rno in range(1, 13):
        if rno not in venue_races:
            st.markdown(
                f'<div style="text-align:center;color:var(--line);padding:8px 0;">{rno}</div>',
                unsafe_allow_html=True,
            )
            continue
        is_active = st.session_state["target_rno"] == rno
        if is_active:
            st.markdown('<div class="venue-btn-active">', unsafe_allow_html=True)
        if st.button(str(rno), key=f"race_{rno}", use_container_width=True):
            st.session_state["target_rno"] = rno
            st.rerun()
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)


def render_current_selection() -> None:
    if not (st.session_state["target_jcd"] and st.session_state["target_rno"]):
        return
    venue_name = VENUES_MAP.get(st.session_state["target_jcd"], st.session_state["target_jcd"])
    html = (
        '<div class="race-select-display">'
        '<span class="now-label">表示中</span>'
        f'<span class="venue">{venue_name}</span>'
        f'<span class="rno">{st.session_state["target_rno"]}R</span>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_roughness(row: pd.Series) -> None:
    score = row.get("roughness_score", 0)
    label = row.get("roughness_label", "-")
    color = ROUGHNESS_LABEL_COLOR.get(str(label), "#5A6B7D")

    html = (
        '<div class="ai-card">'
        '<div class="ai-card-title">荒れやすさ</div>'
        f'<div style="font-size:20px;font-weight:700;color:{color};">{label}</div>'
        f'<div style="font-size:13px;color:var(--ink-soft);margin-bottom:10px;">{score:.0f} / 100</div>'
        '<div style="position:relative;height:24px;border-radius:12px;'
        'background:linear-gradient(90deg,#E1F3EC 0%,#FFF4DD 50%,#FBE5E2 100%);overflow:hidden;">'
        f'<div style="position:absolute;top:0;bottom:0;left:{score}%;width:3px;background:#1A2433;"></div>'
        "</div>"
        '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--ink-soft);margin-top:4px;">'
        "<span>超堅め</span><span>堅め</span><span>普通</span><span>荒れ注意</span><span>波乱含み</span>"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_ai_predictions(row: pd.Series, top_n: int = 5) -> None:
    rows_html = []
    for rank in range(1, top_n + 1):
        ticket = row.get(f"ai_top{rank}_ticket", "")
        prob = row.get(f"ai_top{rank}_prob", 0.0)
        mark = row.get(f"ai_top{rank}_mark", "")
        bg, fg = MARK_STYLE.get(str(mark), MARK_STYLE[""])
        prob_pct = float(prob) * 100 if pd.notna(prob) else 0.0
        rows_html.append(
            f'<div style="display:flex;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);">'
            f'<div style="width:30px;height:30px;border-radius:8px;background:{bg};color:{fg};'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;margin-right:12px;">{mark}</div>'
            f'<div style="flex:1;font-size:19px;font-weight:700;">{ticket}</div>'
            f'<div style="font-size:15px;color:var(--primary);font-weight:700;">{prob_pct:.1f}%</div>'
            f"</div>"
        )

    legend_html = (
        '<div style="display:flex;gap:14px;font-size:11px;color:var(--ink-soft);margin-top:10px;flex-wrap:wrap;">'
        "<span>◎ 3モデル一致</span><span>○ 2モデル一致</span><span>▲△ 総合順位</span>"
        "</div>"
    )
    card_html = (
        '<div class="ai-card"><div class="ai-card-title">'
        f"AI予想 Top{top_n}</div>"
        f"{''.join(rows_html)}{legend_html}</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_payout_distribution(row: pd.Series) -> None:
    avg_payout = row.get("similar_avg_payout", 0)
    median_payout = row.get("similar_median_payout", 0)

    bins = [
        ("〜999円", row.get("similar_under_1000_rate", 0)),
        ("1k〜3k", row.get("similar_1000_2999_rate", 0)),
        ("3k〜1万", row.get("similar_3000_9999_rate", 0)),
        ("1万円〜", row.get("similar_over_10000_rate", 0)),
    ]
    max_rate = max([r for _, r in bins] + [0.01])

    bars_html = []
    danger_idx = 3
    for i, (label, rate) in enumerate(bins):
        pct = float(rate) * 100 if pd.notna(rate) else 0.0
        height = max((rate / max_rate) * 60, 4) if max_rate > 0 else 4
        color = "#E2342B" if i == danger_idx else "#00A0E9"
        bars_html.append(
            '<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;">'
            f'<div style="font-size:11px;font-weight:700;color:var(--ink-soft);">{pct:.0f}%</div>'
            f'<div style="width:100%;background:{color};border-radius:4px 4px 0 0;height:{height}px;"></div>'
            f'<div style="font-size:9px;color:var(--ink-soft);">{label}</div>'
            "</div>"
        )

    card_html = (
        '<div class="ai-card">'
        '<div class="ai-card-title">近似100レース・配当分布</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">'
        '<div style="background:var(--bg);border-radius:10px;padding:10px 12px;">'
        '<div style="font-size:11px;color:var(--ink-soft);">平均払戻</div>'
        f'<div style="font-size:19px;font-weight:700;">{avg_payout:,.0f}円</div>'
        "</div>"
        '<div style="background:var(--bg);border-radius:10px;padding:10px 12px;">'
        '<div style="font-size:11px;color:var(--ink-soft);">中央値</div>'
        f'<div style="font-size:19px;font-weight:700;">{median_payout:,.0f}円</div>'
        "</div>"
        "</div>"
        '<div style="display:flex;align-items:flex-end;gap:8px;height:90px;">'
        f"{''.join(bars_html)}"
        "</div>"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_rank_frequency(row: pd.Series, top_n: int = 5) -> None:
    items = []
    max_rate = 0.0001
    for rank in range(1, top_n + 1):
        rate = row.get(f"similar_rank{rank}_rate", 0)
        if pd.notna(rate):
            max_rate = max(max_rate, float(rate))

    for rank in range(1, top_n + 1):
        ticket = row.get(f"similar_rank{rank}_ticket", "")
        rate = row.get(f"similar_rank{rank}_rate", 0)
        rate_pct = float(rate) * 100 if pd.notna(rate) else 0.0
        bar_width = (rate / max_rate) * 100 if max_rate > 0 else 0
        if not ticket or pd.isna(ticket):
            continue
        items.append(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<div style="font-size:12px;color:var(--ink-soft);width:16px;">{rank}</div>'
            f'<div style="font-weight:700;font-size:14px;width:56px;">{ticket}</div>'
            '<div style="flex:1;height:8px;background:var(--bg);border-radius:4px;overflow:hidden;">'
            f'<div style="height:100%;background:var(--primary);border-radius:4px;width:{bar_width:.0f}%;"></div>'
            "</div>"
            f'<div style="font-size:12px;color:var(--ink-soft);width:32px;text-align:right;">{rate_pct:.0f}%</div>'
            "</div>"
        )

    card_html = (
        '<div class="ai-card">'
        '<div class="ai-card-title">近似100レース・着順発生率</div>'
        f"{''.join(items)}"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ====================================================
# メイン
# ====================================================
def main() -> None:
    st.set_page_config(page_title="競艇AI予想", layout="centered")
    inject_style()

    st.markdown('<div class="fixed-brand-bar">競艇AI予想</div>', unsafe_allow_html=True)

    if "target_jcd" not in st.session_state:
        st.session_state["target_jcd"] = None
    if "target_rno" not in st.session_state:
        st.session_state["target_rno"] = None

    df, date_label = load_today_predictions()

    if df is None:
        st.warning(
            "予測データが見つかりません。\n\n"
            "先に `python run_train_models.py` でモデルを学習し、"
            "`python run_site_predictions.py --input ... --output ...` で予測CSVを作成してください。"
        )
        return

    st.markdown(
        f'<div style="font-size:11px;color:var(--ink-soft);margin-bottom:10px;">データ日付: {date_label}</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        render_venue_picker(df)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["target_jcd"]:
        with st.container():
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            render_race_picker(df)
            st.markdown("</div>", unsafe_allow_html=True)

    render_current_selection()

    if st.session_state["target_jcd"] and st.session_state["target_rno"]:
        target = df[
            (df["jcd"].astype(str) == st.session_state["target_jcd"])
            & (df["r"].astype(int) == st.session_state["target_rno"])
        ]
        if target.empty:
            st.info("選択したレースのデータが見つかりません。")
        else:
            row = target.iloc[0]
            render_roughness(row)
            render_ai_predictions(row, top_n=5)
            render_payout_distribution(row)
            render_rank_frequency(row, top_n=5)
    else:
        st.info("競艇場とレースを選ぶと、AI予想が表示されます。")


if __name__ == "__main__":
    main()
