"""番組表(B形式)・競走成績(K形式)の固定長テキストを解析するモジュール。

バイトオフセットは実データで検証済み(2005-2026年、レイアウト変化なし):
- Bファイルの選手行: 79バイト固定長
- Kファイルの選手成績行: 66バイト固定長

構造は絶対行番号ではなく、目印(BBGN/KBGN, "-----"区切り線, ヘッダー行)を
探すマーカーベースの解析にしている。21年分でイベント名の長さ等による
空行数の揺れがあっても崩れないようにするため。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
FULLWIDTH_ALNUM = str.maketrans("０１２３４５６７８９Ｈｍ", "0123456789Hm")

# 7z実行ファイルの候補(Windowsローカル環境 / GitHub Actions Ubuntuランナーの両対応)。
# UbuntuランナーではワークフローでAPTから `p7zip-full` をインストールし、
# `7z` または `7zr` がPATH上に来ることを想定している。
_SEVENZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    "7z", "7za", "7zr",
]


def _find_sevenzip() -> str:
    for candidate in _SEVENZIP_CANDIDATES:
        if os.path.sep in candidate or candidate.endswith(".exe"):
            if Path(candidate).exists():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    raise FileNotFoundError(
        "7z実行ファイルが見つかりません。Windowsでは7-Zipをインストール、"
        "GitHub Actions(Ubuntu)では `sudo apt-get install -y p7zip-full` を"
        "ワークフローに追加してください。"
    )


def extract_lzh_text(lzh_path: Path, tmp_dir: Path) -> bytes:
    """LZHファイルを展開し、中のTXTファイルの生バイトを返す。"""
    sevenzip = _find_sevenzip()
    subprocess.run(
        [sevenzip, "x", f"-o{tmp_dir}", "-y", str(lzh_path)],
        check=True, capture_output=True,
    )
    txt_files = list(tmp_dir.glob("*.TXT")) + list(tmp_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"LZH内にTXTが見つかりません: {lzh_path}")
    data = txt_files[0].read_bytes()
    txt_files[0].unlink()
    return data


def _dec(b: bytes) -> str:
    return b.decode("cp932", errors="replace").strip()


def _venue_blocks(raw: bytes, marker: bytes) -> list[tuple[str, bytes]]:
    """XXBBGN / XXKBGN で競艇場ごとのブロックに分割する。戻り値: [(jcd, block_bytes), ...]"""
    pattern = re.compile(marker)
    matches = list(pattern.finditer(raw))
    blocks = []
    for i, m in enumerate(matches):
        jcd = m.group(1).decode("ascii")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        blocks.append((jcd, raw[start:end]))
    return blocks


def parse_b_venue_block(jcd: str, block: bytes) -> list[dict]:
    """1競艇場分のBブロックから、レースごとの6艇データを抽出する。"""
    lines = block.split(b"\r\n")
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # ヘッダー行(「艇 選手 選手」を含む)の次の区切り線の後に6艇分のデータがある
        if b"\x92\xf8" in line and b"\x91I\x8e\xe8" in line:  # 艇, 選手 (cp932)
            # レース番号と距離を直近数行以内の "○Ｒ … Ｈ1800ｍ" 表記から探す。
            # レース距離は天候と違い、当日朝の番組表(B形式)heading行に
            # 既に確定値として記載されている(全角"Ｈ1800ｍ"表記)ため、
            # 天候・風速等とは異なり購入前に取得可能。
            race_no = None
            distance = None
            for back in range(1, 6):
                if i - back < 0:
                    break
                text = _dec(lines[i - back])
                m = re.search(r"([０-９0-9]{1,2})\s*[ＲR]", text)
                if m:
                    race_no = int(m.group(1).translate(FULLWIDTH_DIGITS))
                    dm = re.search(r"H\s*(\d{3,4})\s*m", text.translate(FULLWIDTH_ALNUM))
                    if dm:
                        distance = int(dm.group(1))
                    break
            # ヘッダー2行目 + 区切り線をスキップして選手行へ
            j = i + 3
            boats = []
            while j < len(lines) and len(lines[j]) >= 79 and lines[j][0:1].isdigit():
                boats.append(_parse_b_racer_line(lines[j][:79]))
                j += 1
            if race_no is not None and boats:
                rows.append({"jcd": jcd, "r": race_no, "boats": boats, "距離": distance})
            i = j
        else:
            i += 1
    return rows


def _parse_b_racer_line(line_bytes: bytes) -> dict:
    return {
        "艇番": _dec(line_bytes[0:1]),
        "登番": _dec(line_bytes[2:6]),
        "選手名": _dec(line_bytes[6:14]),
        "年齢": _dec(line_bytes[14:16]),
        "支部": _dec(line_bytes[16:20]),
        "体重": _dec(line_bytes[20:22]),
        "級別": _dec(line_bytes[22:24]),
        "全国勝率": _dec(line_bytes[25:29]),
        "全国2率": _dec(line_bytes[30:35]),
        "当地勝率": _dec(line_bytes[36:40]),
        "当地2率": _dec(line_bytes[41:46]),
        "モーターNo": _dec(line_bytes[47:49]),
        "モーター2率": _dec(line_bytes[50:55]),
        "ボートNo": _dec(line_bytes[56:58]),
        "ボート2率": _dec(line_bytes[59:64]),
    }


def _parse_wide_and_ninki(lines: list[bytes], start_idx: int, search_limit: int = 15) -> dict:
    """艇成績行の直後にある単勝/複勝/２連単/２連複/拡連複/３連単/３連複の内訳欄から、
    2連複の人気順位と、拡連複3通り(組・払戻・人気)を取り出す。
    例:
        ２連複   1-2        180  人気     1
        拡連複   1-2        110  人気     1
                 1-3        160  人気     3
                 2-3        330  人気     6
    """
    result: dict = {}
    end = min(start_idx + search_limit, len(lines))
    idx = start_idx
    while idx < end:
        text = _dec(lines[idx])
        if re.match(r"\s*\d{1,2}R\s", text) and idx > start_idx:
            break  # 次のレース見出しに到達
        m_quinella = re.match(r"２連複\s+\S+\s+\d+\s+人気\s+(\d+)", text)
        if m_quinella:
            result["2連複_人気"] = int(m_quinella.group(1))
            idx += 1
            continue
        m_wide = re.match(r"拡連複\s+(\S+)\s+(\d+)\s+人気\s+(\d+)", text)
        if m_wide:
            result["拡連複1_組"] = m_wide.group(1)
            result["拡連複1_払戻"] = int(m_wide.group(2))
            result["拡連複1_人気"] = int(m_wide.group(3))
            for offset, n in ((1, 2), (2, 3)):
                if idx + offset < len(lines):
                    # _dec()が前後空白をstripする(継続行は元々先頭が空白のみで始まる)ため、
                    # 行頭の空白は要求しない
                    m_cont = re.match(r"(\S+)\s+(\d+)\s+人気\s+(\d+)", _dec(lines[idx + offset]))
                    if m_cont:
                        result[f"拡連複{n}_組"] = m_cont.group(1)
                        result[f"拡連複{n}_払戻"] = int(m_cont.group(2))
                        result[f"拡連複{n}_人気"] = int(m_cont.group(3))
            idx += 3
            continue
        if text.strip().startswith("３連複"):
            break
        idx += 1
    return result


def parse_k_venue_block(jcd: str, block: bytes) -> tuple[list[dict], dict]:
    """1競艇場分のKブロックから、レースごとの6艇成績と払戻金を抽出する。

    戻り値: (races, payout_by_race)
    """
    lines = block.split(b"\r\n")
    payout_by_race: dict[int, dict] = {}
    races = []
    i = 0
    while i < len(lines):
        text = _dec(lines[i])
        # 払戻金一覧の行 (例: "1R  1-2-4     710    1-2-4     350 ...")
        m_pay = re.match(r"\s*(\d{1,2})R\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)", text)
        if m_pay:
            rno = int(m_pay.group(1))
            payout_by_race[rno] = {
                "3連単_組": m_pay.group(2), "3連単_払戻": m_pay.group(3),
                "3連複_組": m_pay.group(4), "3連複_払戻": m_pay.group(5),
                "2連単_組": m_pay.group(6), "2連単_払戻": m_pay.group(7),
                "2連複_組": m_pay.group(8), "2連複_払戻": m_pay.group(9),
            }
            i += 1
            continue

        # レース詳細見出し (例: "   1R       一般          H1200m  雨  風 南西  8m  波  7cm")
        # クラス名部分(「一　般　　　」等、全角スペースが文字間に入る/「進入固定」等の付記あり)は
        # 内容を問わず非貪欲マッチで読み飛ばす
        m_race = re.match(r"\s*(\d{1,2})R\s+.*?H(\d+)m\s+(\S+)\s+風\s+(\S*)\s*(\d+)m\s+波\s+(\d+)cm", text)
        if m_race:
            rno = int(m_race.group(1))
            weather_info = {
                "距離": m_race.group(2), "天候": m_race.group(3),
                "風向": m_race.group(4), "風速": m_race.group(5), "波高": m_race.group(6),
            }
            # ヘッダー行→区切り線→6艇分の成績行
            j = i + 1
            found_header = False
            kimarite = None
            for _ in range(6):
                if j < len(lines) and b"\x92\x85" in lines[j][:6]:  # "着"を含む行を探す
                    found_header = True
                    # ヘッダー行の末尾に決まり手(逃げ/差し/まくり/まくり差し/抜き/恵まれ)が付く
                    # ※見出し中の項目名は半角カタカナ(ﾚｰｽﾀｲﾑ)である点に注意
                    header_text = _dec(lines[j])
                    if "ﾚｰｽﾀｲﾑ" in header_text:
                        kimarite = header_text.split("ﾚｰｽﾀｲﾑ")[-1].strip() or None
                    break
                j += 1
            if not found_header:
                i += 1
                continue
            j += 2  # ヘッダー2行目・区切り線をスキップして選手行へ
            boats = []
            while j < len(lines) and len(lines[j]) >= 66:
                # 着順欄は数字以外に事故失格コード(S1/S2/K0/K1/L0/L1/F)が入ることがあるため、
                # 艇番(常に1桁の数字のはず)で選手行かどうかを判定する
                if not lines[j][6:7].isdigit():
                    break
                boats.append(_parse_k_result_line(lines[j][:66]))
                j += 1
            wide_ninki = _parse_wide_and_ninki(lines, j)
            races.append({"jcd": jcd, "r": rno, "weather": weather_info, "boats": boats,
                          "決まり手": kimarite, **wide_ninki})
            i = j
        else:
            i += 1
    return races, payout_by_race


def _parse_k_result_line(line_bytes: bytes) -> dict:
    return {
        "着順": _dec(line_bytes[2:4]),
        "艇番": _dec(line_bytes[6:7]),
        "登番": _dec(line_bytes[8:12]),
        "選手名": _dec(line_bytes[13:29]),
        "モーター": _dec(line_bytes[29:32]),
        "ボート": _dec(line_bytes[32:38]),
        "展示": _dec(line_bytes[38:43]),
        "進入": _dec(line_bytes[43:47]),
        "ST": _dec(line_bytes[47:55]),
        "レースタイム": _dec(line_bytes[55:66]),
    }


def parse_b_file(raw: bytes) -> list[dict]:
    rows = []
    for jcd, block in _venue_blocks(raw, rb"(\d{2})BBGN"):
        rows.extend(parse_b_venue_block(jcd, block))
    return rows


def parse_k_file(raw: bytes) -> tuple[list[dict], dict]:
    all_races = []
    all_payouts = {}
    for jcd, block in _venue_blocks(raw, rb"(\d{2})KBGN"):
        races, payouts = parse_k_venue_block(jcd, block)
        all_races.extend(races)
        for rno, p in payouts.items():
            all_payouts[(jcd, rno)] = p
    return all_races, all_payouts
