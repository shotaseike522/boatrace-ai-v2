"""mbrace.or.jp公式の番組表(B)・競走成績(K)LZHファイルを、指定した日付分だけ
ダウンロードするモジュール。

URL規則: https://www1.mbrace.or.jp/od2/{B|K}/{YYYYMM}/{b|k}{YYMMDD}.lzh
(download_lzh_archive.pyで確認済みの規則と同じ)

boatrace.jpのHTML画面をスクレイピングするより、この公式の固定長テキスト
アーカイブを直接取得する方が、形式が変わらず壊れにくい。番組表(B)は当日分も
レース開始前に公開されるため、朝の予想生成にそのまま使える。
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

BASE_URL = "https://www1.mbrace.or.jp/od2"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.boatrace.jp/owpc/pc/extra/data/download.html",
}
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 20.0


def download_lzh(kind: str, yyyymmdd: str, out_dir: Path) -> Path | None:
    """kind: 'B' または 'K'。yyyymmdd: 'YYYYMMDD'。
    戻り値: 保存先パス(取得できた場合) または None(その日は未開催/未公開)。
    """
    yyyymm = yyyymmdd[:6]
    yymmdd = yyyymmdd[2:]
    prefix = "b" if kind == "B" else "k"
    filename = f"{prefix}{yymmdd}.lzh"
    out_path = out_dir / filename
    url = f"{BASE_URL}/{kind}/{yyyymm}/{filename}"

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(RETRY_BACKOFF_SEC)
            continue
        if res.status_code == 200 and len(res.content) > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(res.content)
            return out_path
        if res.status_code == 404:
            return None
        time.sleep(RETRY_BACKOFF_SEC)
    return None
