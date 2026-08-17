"""競艇AI予想サイト (Streamlit) — 展開シナリオ方式

outputs/scenario_predictions_YYYYMMDD.csv (run_scenario_predictions.py の出力)を
読み込み、スマートフォン中心のレイアウトで以下を表示する。

  1. 競艇場を選ぶ（本日開催中のみ）
  2. レースを選ぶ（1〜12、2段グリッド）
  3. 表示中の場・レース
  4. 本命買い目（シナリオ1、3連単、逃げ率つき）
  5. 対抗買い目（シナリオ2、3連単、まくり率つき）
  6. 他有力買い目（シナリオ3、3連単）
  7. 類似レース分析（決まり手%・配当分布）

上部には「本日のおすすめレース」として、アラインドペア2連複
(本命・対抗の2着候補が同じ艇の組み合わせになる場合)が閾値51%以上のレースを
確率順に表示する(固定件数ではなく、該当するレースだけを表示)。

列名は run_scenario_predictions.py が出力する仕様
(favorite_boat, scenario1_tickets等)に合わせている。
"""

from __future__ import annotations

import glob
import json
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

# AI予想モデルの切替トグル。"codex_all_head" = Codex製all_head_hierarchical
# モデル(2023-07-01〜2026-06-30の比較でTop1/Top5/logloss/Brier/ECEすべて
# 旧モデルを上回ったため採用)、"scenario_legacy" = 旧シナリオ方式(本命/対抗/
# 他有力の3枚カード)へロールバックする場合はこちらに戻す。両モデルとも
# daily_prep.pyで毎日計算されているため、この定数を書き換えて再デプロイ
# するだけで切り替えられる。
ACTIVE_AI_MODEL = "codex_all_head"


# ====================================================
# スタイル（白ベース + 青アクセント、スマホ最優先）
# ====================================================
def inject_style() -> None:
    # Google Analytics（GA4）計測タグ
    # st.markdownで直接DOMに挿入することでiframe内に閉じ込められる問題を回避する
    st.markdown(
        """
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-3P0PV90GHQ"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-3P0PV90GHQ');
        </script>
        """,
        unsafe_allow_html=True,
    )
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
            color: var(--ink);
        }
        /* 保険: ai-card内のテキストは、個別にcolor指定が無くても
           OS/ブラウザのダークモード設定の影響を受けず常に読める色にする */
        .ai-card div {
            color: inherit;
        }
        /* st.container(border=True, key=...)版のカード(内部にst.button等の
           ネイティブウィジェットを置く場所で使用。st.markdownのdiv開閉タグを
           分割する方式だと、st.markdown呼び出し単位でHTMLが個別にパースされる
           ためウィジェットを実際には囲えず、カードが空枠になってしまう不具合が
           あったため、Streamlitネイティブの枠付きコンテナに置き換えた。
           key=を指定すると Streamlit が st-key-<key> という安定したCSSクラスを
           付与してくれるため、それを使って個別にスタイルを当てる) */
        .st-key-ai-card-pickup, .st-key-ai-card-venue, .st-key-ai-card-race {
            background: var(--surface) !important;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            padding: 16px !important;
            margin-bottom: 14px;
        }
        .st-key-ai-card-pickup *, .st-key-ai-card-venue *, .st-key-ai-card-race * {
            color: inherit;
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

        /* st.columns は画面が狭いと自動的に縦積みに変わってしまうため、
           横並び(flex-direction: row)を画面幅に関わらず強制する。
           Streamlitの標準DOM: st.columns() は [data-testid="stHorizontalBlock"] を生成し、
           その直下に各列が [data-testid="stColumn"] として並ぶ。 */
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 8px;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 0 !important;
            width: auto !important;
            min-width: 0 !important;
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
        /* 選択中の会場/レース/おすすめレースボタンのハイライト。旧実装は
           <div class="venue-btn-active">をst.markdownで開き、別のst.button呼び出しを
           挟んで別のst.markdownで閉じる方式だったが、Streamlitはst.markdown呼び出し
           ごとにHTMLを個別にパースするため実際にはボタンを囲えず、機能していなかった。
           st.button(type="primary")というネイティブAPIに置き換えた。 */
        div[data-testid="stButton"] button[kind="primary"] {
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
    """outputs/scenario_predictions_YYYYMMDD.csv のうち最新日付のものを読み込む。"""
    jst = pytz.timezone("Asia/Tokyo")
    today_str = datetime.now(jst).strftime("%Y%m%d")

    candidate = os.path.join(OUTPUTS_DIR, f"scenario_predictions_{today_str}.csv")
    if os.path.exists(candidate):
        df = pd.read_csv(candidate, dtype={"jcd": str})
        return df, today_str

    pattern = os.path.join(OUTPUTS_DIR, "scenario_predictions_*.csv")
    files = sorted(f for f in glob.glob(pattern) if not f.endswith("_internal.csv"))
    if not files:
        return None, None
    latest = files[-1]
    df = pd.read_csv(latest, dtype={"jcd": str})
    date_label = os.path.basename(latest).replace("scenario_predictions_", "").replace(".csv", "")
    return df, date_label


@st.cache_data(ttl=300)
def load_today_all_head_predictions() -> pd.DataFrame | None:
    """outputs/all_head_predictions_YYYYMMDD.csv (run_all_head_predictions.pyの出力、
    Codex製all_head_hierarchicalモデル)のうち最新日付のものを読み込む。"""
    jst = pytz.timezone("Asia/Tokyo")
    today_str = datetime.now(jst).strftime("%Y%m%d")

    candidate = os.path.join(OUTPUTS_DIR, f"all_head_predictions_{today_str}.csv")
    if os.path.exists(candidate):
        return pd.read_csv(candidate, dtype={"jcd": str})

    pattern = os.path.join(OUTPUTS_DIR, "all_head_predictions_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pd.read_csv(files[-1], dtype={"jcd": str})


# ====================================================
# 画面パーツ
# ====================================================
def render_venue_picker(df: pd.DataFrame) -> None:
    st.markdown('<div class="ai-card-title">競艇場を選ぶ（本日開催中）</div>', unsafe_allow_html=True)
    available_jcds = sorted(df["jcd"].astype(str).unique().tolist())

    n_cols = 3
    rows = [available_jcds[i : i + n_cols] for i in range(0, len(available_jcds), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for i, jcd in enumerate(row):
            venue_name = VENUES_MAP.get(jcd, jcd)
            is_active = st.session_state["target_jcd"] == jcd
            with cols[i]:
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{jcd}\n{venue_name}", key=f"venue_{jcd}", use_container_width=True, type=btn_type):
                    st.session_state["target_jcd"] = jcd
                    st.session_state["target_rno"] = None
                    st.rerun()


def render_race_picker(df: pd.DataFrame) -> None:
    if not st.session_state["target_jcd"]:
        return
    st.markdown('<div class="ai-card-title">レースを選ぶ</div>', unsafe_allow_html=True)
    venue_races = sorted(
        df.loc[df["jcd"].astype(str) == st.session_state["target_jcd"], "r"].astype(int).unique().tolist()
    )

    for row_start in (1, 7):
        cols = st.columns(6)
        for offset in range(6):
            rno = row_start + offset
            with cols[offset]:
                if rno not in venue_races:
                    st.markdown(
                        f'<div style="text-align:center;color:var(--line);padding:8px 0;">{rno}</div>',
                        unsafe_allow_html=True,
                    )
                    continue
                is_active = st.session_state["target_rno"] == rno
                btn_type = "primary" if is_active else "secondary"
                if st.button(str(rno), key=f"race_{rno}", use_container_width=True, type=btn_type):
                    st.session_state["target_rno"] = rno
                    st.rerun()


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


def _parse_tickets(raw: object) -> list[tuple[str, float]]:
    """scenario{1,2,3}_tickets列(JSON文字列 '[["1-4-3",0.034],...]')をパースする。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        return [(t, p) for t, p in json.loads(raw)]
    except (ValueError, TypeError):
        return []


def _render_ticket_card(title: str, subtitle: str, tickets: list[tuple[str, float]]) -> None:
    if not tickets:
        rows_html = '<div style="font-size:13px;color:var(--ink-soft);padding:8px 0;">該当なし</div>'
    else:
        rows_html = "".join(
            f'<div style="display:flex;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);">'
            f'<div style="flex:1;font-size:19px;font-weight:700;color:var(--ink);">{ticket}</div>'
            f'<div style="font-size:15px;color:var(--primary);font-weight:700;">{prob*100:.1f}%</div>'
            f"</div>"
            for ticket, prob in tickets
        )
    card_html = (
        '<div class="ai-card">'
        f'<div class="ai-card-title">{title}</div>'
        f'<div style="font-size:12px;color:var(--ink-soft);margin-bottom:8px;">{subtitle}</div>'
        f"{rows_html}"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


AI_BOLD_THRESHOLD = 0.05  # 予測確率5%以上は太字、未満は通常字にする


def _render_ai_ticket_rows(tickets: list[tuple[str, float]]) -> str:
    if not tickets:
        return '<div style="font-size:13px;color:var(--ink-soft);padding:8px 0;">該当なし</div>'
    rows_html = []
    for ticket, prob in tickets:
        weight = 700 if prob >= AI_BOLD_THRESHOLD else 400
        rows_html.append(
            '<div style="display:flex;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);">'
            f'<div style="flex:1;font-size:19px;font-weight:{weight};color:var(--ink);">{ticket}</div>'
            f'<div style="font-size:15px;color:var(--primary);font-weight:{weight};">{prob * 100:.1f}%</div>'
            "</div>"
        )
    return "".join(rows_html)


def render_all_head_predictions(row: pd.Series) -> None:
    """Codex製all_head_hierarchicalモデルによる3連単予測を、
    「1着可能性が最も高い号艇(本命)のTop5」「2番目に高い号艇(対抗)のTop3」の
    2グループで表示する。予測確率5%以上は太字、5%未満は通常字にする。

    内部の較正方式(temperature/isotonic/outer残差補正)やlogloss等の指標は
    サイトには一切表示しない。表示名・補足文は指示書の仕様通り固定する。
    """
    favorite_boat = row.get("favorite_boat", "-")
    challenger_boat = row.get("challenger_boat", "-")

    favorite_tickets: list[tuple[str, float]] = []
    for k in range(1, 6):
        ticket = row.get(f"favorite_top{k}_ticket")
        prob = row.get(f"favorite_top{k}_prob")
        if ticket is None or pd.isna(ticket) or prob is None or pd.isna(prob):
            continue
        favorite_tickets.append((str(ticket), float(prob)))

    challenger_tickets: list[tuple[str, float]] = []
    for k in range(1, 4):
        ticket = row.get(f"challenger_top{k}_ticket")
        prob = row.get(f"challenger_top{k}_prob")
        if ticket is None or pd.isna(ticket) or prob is None or pd.isna(prob):
            continue
        challenger_tickets.append((str(ticket), float(prob)))

    card_html = (
        '<div class="ai-card">'
        '<div class="ai-card-title">AI予想確率</div>'
        '<div style="font-size:12px;color:var(--ink-soft);margin-bottom:8px;">過去データで実績補正した推定確率です。</div>'
        f'<div style="font-size:13px;font-weight:700;color:var(--ink);margin-top:6px;">本命（{favorite_boat}号艇）</div>'
        f"{_render_ai_ticket_rows(favorite_tickets)}"
        f'<div style="font-size:13px;font-weight:700;color:var(--ink);margin-top:12px;">対抗（{challenger_boat}号艇）</div>'
        f"{_render_ai_ticket_rows(challenger_tickets)}"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_scenario1(row: pd.Series) -> None:
    """シナリオ1(本命買い目)。本命艇の逃げ率を参考値として見出しに添える。"""
    boat = row.get("favorite_boat", "-")
    nige = row.get("favorite_nige_rate", None)
    subtitle = f"{boat}号艇が優勝" + (f"(逃げ率{float(nige):.0f}%)" if pd.notna(nige) else "")
    tickets = _parse_tickets(row.get("scenario1_tickets"))
    _render_ticket_card("本命買い目", subtitle, tickets)


def render_scenario2(row: pd.Series) -> None:
    """シナリオ2(対抗買い目)。対抗艇のまくり系決着率を参考値として見出しに添える。"""
    boat = row.get("challenger_boat", "-")
    makuri = row.get("challenger_makuri_rate", None)
    subtitle = f"{boat}号艇が優勝" + (f"(まくり率{float(makuri):.0f}%)" if pd.notna(makuri) else "")
    tickets = _parse_tickets(row.get("scenario2_tickets"))
    _render_ticket_card("対抗買い目", subtitle, tickets)


def render_scenario3(row: pd.Series) -> None:
    """シナリオ3(他有力/保険買い目)。シナリオ1・2に含まれない上位確率の目。"""
    tickets = _parse_tickets(row.get("scenario3_tickets"))
    _render_ticket_card("他有力買い目", "本命・対抗に入らない高確率目", tickets)


def render_one_head_high_confidence(row: pd.Series) -> None:
    """1号艇高信頼候補（検証中）を表示する。該当レース以外は何も表示しない。

    NOTE: まだ本番の描画フローには組み込んでいない（呼び出し元コメント参照）。
    将来組み込む際は outputs/one_head_high_confidence_site_predictions.csv を
    jcd, r で本線dfにマージし、列名は one_head_ticket / one_head_score /
    one_head_roughness_bin / one_head_signal_name / one_head_signal_note の
    ようにプレフィックスを付けること（既存の roughness_bin 等と衝突するため）。
    「25%確定」「勝てる」「回収率100%以上確定」のような断定表現は使わない。
    """
    ticket = row.get("one_head_ticket", "")
    if not ticket or pd.isna(ticket) or ticket == "":
        return
    score = row.get("one_head_score", None)
    roughness_bin = row.get("one_head_roughness_bin", "")
    signal_name = row.get("one_head_signal_name", "1号艇高信頼候補（検証中）")
    signal_note = row.get(
        "one_head_signal_note",
        "過去検証で高的中率だった固定条件に該当した買い目です。現在は前向き検証中です。",
    )
    score_text = f"{float(score):.3f}" if pd.notna(score) else "-"
    card_html = (
        '<div class="ai-card">'
        f'<div class="ai-card-title">{signal_name}</div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;">'
        f'<div style="font-family:\'Roboto Condensed\',sans-serif;font-size:22px;font-weight:700;color:var(--ink);">{ticket}</div>'
        '<div style="text-align:right;font-size:13px;color:var(--ink-soft);">'
        f'<div>スコア：{score_text}</div>'
        f'<div>荒れ度：{roughness_bin}</div>'
        '</div>'
        '</div>'
        f'<div style="font-size:11px;color:var(--ink-soft);">{signal_note}</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_similar_race_analysis(row: pd.Series) -> None:
    """類似レース分析: 決まり手%(数字列挙型)＋3連単配当分布(実際の割合＋平均比)を表示する。"""
    try:
        kimarite = json.loads(row["kimarite_dist"]) if pd.notna(row.get("kimarite_dist")) else None
    except (ValueError, TypeError):
        kimarite = None
    try:
        labels = json.loads(row["payout_bucket_labels"]) if pd.notna(row.get("payout_bucket_labels")) else []
    except (ValueError, TypeError):
        labels = []
    try:
        pct_actual = json.loads(row["payout_bucket_pct"]) if pd.notna(row.get("payout_bucket_pct")) else None
    except (ValueError, TypeError):
        pct_actual = None
    try:
        deviation = json.loads(row["payout_bucket_deviation"]) if pd.notna(row.get("payout_bucket_deviation")) else None
    except (ValueError, TypeError):
        deviation = None

    if kimarite:
        kimarite_line = "・".join(f"{k}{v:.0f}%" for k, v in sorted(kimarite.items(), key=lambda x: -x[1]))
        kimarite_html = f'<div style="font-size:14px;color:var(--ink);margin-bottom:12px;">決まり手: {kimarite_line}</div>'
    else:
        kimarite_html = '<div style="font-size:13px;color:var(--ink-soft);margin-bottom:12px;">決まり手データなし</div>'

    bars_html = ""
    if pct_actual and deviation and labels:
        n = len(pct_actual)
        max_pct = max(pct_actual + [5.0]) * 1.1  # 平均線が常に収まるよう5%も含めて余白を確保
        chart_w, chart_h = 320, 90
        gap = 2
        bar_w = (chart_w - gap * (n - 1)) / n
        baseline_y = chart_h - (5.0 / max_pct) * chart_h

        bars_svg = []
        for i, pct in enumerate(pct_actual):
            x = i * (bar_w + gap)
            h = (pct / max_pct) * chart_h
            y = chart_h - h
            color = "#E2342B" if pct > 5.0 else "#0046AD"
            bars_svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h,1):.1f}" '
                f'rx="3" fill="{color}"><title>{labels[i]}: {pct:.1f}%({deviation[i]:+.1f}pt)</title></rect>'
            )
        svg = (
            f'<svg viewBox="0 0 {chart_w} {chart_h + 14}" style="width:100%;height:auto;" '
            'role="img" aria-label="配当分布の棒グラフ。横軸は配当額が低い順、縦軸は出現割合。破線は全体平均5%">'
            f'<line x1="0" y1="{baseline_y:.1f}" x2="{chart_w}" y2="{baseline_y:.1f}" '
            'stroke="var(--ink-soft)" stroke-width="1" stroke-dasharray="3,3"/>'
            f'<text x="{chart_w}" y="{baseline_y - 4:.1f}" text-anchor="end" font-size="8" fill="var(--ink-soft)">平均5%</text>'
            f"{''.join(bars_svg)}"
            f'<text x="0" y="{chart_h + 12}" font-size="9" fill="var(--ink-soft)">{labels[0]}</text>'
            f'<text x="{chart_w}" y="{chart_h + 12}" text-anchor="end" font-size="9" fill="var(--ink-soft)">{labels[-1]}</text>'
            "</svg>"
        )
        bars_html = (
            '<div style="font-size:11px;color:var(--ink-soft);margin-bottom:8px;">'
            "配当分布(直近3年の類似レースでの実際の出現割合、配当額が低い順に20区間。"
            "破線=全体平均5%。赤=平均より出やすい、青=平均より出にくい)</div>"
            f"{svg}"
        )

    card_html = (
        '<div class="ai-card">'
        '<div class="ai-card-title">類似レース分析</div>'
        f"{kimarite_html}"
        f"{bars_html}"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ====================================================
# メイン
# ====================================================
def render_pickup_races(df: pd.DataFrame) -> None:
    """アラインドペア2連複(本命・対抗の2着候補が同じ艇の組になり、かつ
    combined_probが閾値51%以上)のレースを確率順にピックアップする。
    該当件数は日によって変動する(固定件数ではない)。
    タップすると下の競艇場・レースボタンが連動して選択され、予想内容が表示される。
    """
    if "aligned_pair_flag" not in df.columns:
        return

    pickup = (
        df[df["aligned_pair_flag"].astype(str) == "True"]  # CSV読み込み時にbool/文字列どちらの型でも拾えるようにstr比較
        [["jcd", "r", "aligned_pair", "aligned_pair_prob"]]
        .dropna(subset=["aligned_pair", "aligned_pair_prob"])
        .sort_values("aligned_pair_prob", ascending=False)
        .reset_index(drop=True)
    )
    if pickup.empty:
        st.markdown(
            '<div class="ai-card-title">本日のおすすめレース</div>'
            '<div style="font-size:13px;color:var(--ink-soft);">本日は該当するレースがありませんでした。</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="ai-card-title">本日のおすすめレース</div>', unsafe_allow_html=True)

    n_cols = 3
    rows = [pickup.iloc[i : i + n_cols] for i in range(0, len(pickup), n_cols)]
    for row_df in rows:
        cols = st.columns(n_cols)
        for i, (_, item) in enumerate(row_df.iterrows()):
            jcd = str(item["jcd"]).zfill(2)
            rno = int(item["r"])
            pair = str(item["aligned_pair"]).replace("-", "＝")
            prob_pct = float(item["aligned_pair_prob"]) * 100
            venue_name = VENUES_MAP.get(jcd, jcd)
            is_active = (
                st.session_state["target_jcd"] == jcd
                and st.session_state["target_rno"] == rno
            )
            with cols[i]:
                label = f"{venue_name} {rno}R\n{pair} {prob_pct:.0f}%"
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"pickup_{jcd}_{rno}", use_container_width=True, type=btn_type):
                    st.session_state["target_jcd"] = jcd
                    st.session_state["target_rno"] = rno
                    st.rerun()


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
            "先に `python retrain_monthly.py` でモデルを学習し、"
            "`python run_scenario_predictions.py --input ... --output ...` で予測CSVを作成してください。"
        )
        return

    st.markdown(
        f'<div style="font-size:11px;color:var(--ink-soft);margin-bottom:10px;">データ日付: {date_label}</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="ai-card-pickup"):
        render_pickup_races(df)

    with st.container(border=True, key="ai-card-venue"):
        render_venue_picker(df)

    if st.session_state["target_jcd"]:
        with st.container(border=True, key="ai-card-race"):
            render_race_picker(df)

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
            if ACTIVE_AI_MODEL == "codex_all_head":
                all_head_df = load_today_all_head_predictions()
                if all_head_df is None:
                    st.warning(
                        "Codexモデルの予測データが見つかりません。\n\n"
                        "先に `python run_all_head_predictions.py --input ... --output ...` "
                        "で予測CSVを作成してください。"
                    )
                else:
                    ah_target = all_head_df[
                        (all_head_df["jcd"].astype(str) == st.session_state["target_jcd"])
                        & (all_head_df["r"].astype(int) == st.session_state["target_rno"])
                    ]
                    if ah_target.empty:
                        st.info("選択したレースのAI予想データが見つかりません。")
                    else:
                        render_all_head_predictions(ah_target.iloc[0])
            else:
                render_scenario1(row)
                render_scenario2(row)
                render_scenario3(row)
            render_similar_race_analysis(row)
    else:
        st.info("競艇場とレースを選ぶと、展開シナリオが表示されます。")


if __name__ == "__main__":
    main()
