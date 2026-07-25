"""Plackett-Luceモデルによる、6艇の「強さスコア」から
2連複(上位2着に入る2艇の組)・拡連複(上位3着以内に入る2艇の組)の
厳密な的中確率を計算するモジュール。

考え方:
  各艇iに強さθ_i(>0)を割り当てる。順位付けは逐次的に「まだ決まっていない
  艇の中からθに比例した確率で1着→2着→...と抜き取っていく」というモデル
  (Plackett-Luce)に従うと仮定する。6艇なので全順列は6!=720通りしかなく、
  シミュレーションではなく厳密に全720通りの確率を計算して積み上げる。
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

N_BOATS = 6
ALL_PERMS = np.array(list(permutations(range(N_BOATS))))  # shape (720, 6)
N_PERMS = len(ALL_PERMS)

PAIRS = [(i, j) for i in range(N_BOATS) for j in range(i + 1, N_BOATS)]  # 15通り
N_PAIRS = len(PAIRS)

TRIPLES = list(permutations(range(N_BOATS), 3))  # 120通り(順序あり、3連単用)
N_TRIPLES = len(TRIPLES)


def _build_pair_masks() -> tuple[np.ndarray, np.ndarray]:
    """(quinella_mask, wide_mask) を作る。shapeは共に (N_PAIRS, N_PERMS) のbool。
    quinella_mask[p, k] = 順列kの上位2着が pair p と一致するか
    wide_mask[p, k]     = 順列kの上位3着に pair p の両方が含まれるか
    """
    top2 = ALL_PERMS[:, :2]
    top3 = ALL_PERMS[:, :3]
    quinella_mask = np.zeros((N_PAIRS, N_PERMS), dtype=bool)
    wide_mask = np.zeros((N_PAIRS, N_PERMS), dtype=bool)
    for p_idx, (a, b) in enumerate(PAIRS):
        quinella_mask[p_idx] = ((top2[:, 0] == a) & (top2[:, 1] == b)) | \
                                ((top2[:, 0] == b) & (top2[:, 1] == a))
        a_in_top3 = (top3 == a).any(axis=1)
        b_in_top3 = (top3 == b).any(axis=1)
        wide_mask[p_idx] = a_in_top3 & b_in_top3
    return quinella_mask, wide_mask


QUINELLA_MASK, WIDE_MASK = _build_pair_masks()  # 各 (15, 720)


def _build_triple_mask() -> np.ndarray:
    """trifecta_mask[t, k] = 順列kの上位3着が TRIPLES[t] と完全一致(順序込み)するか。
    shape (120, 720)。
    """
    top3 = ALL_PERMS[:, :3]
    mask = np.zeros((N_TRIPLES, N_PERMS), dtype=bool)
    for t_idx, (a, b, c) in enumerate(TRIPLES):
        mask[t_idx] = (top3[:, 0] == a) & (top3[:, 1] == b) & (top3[:, 2] == c)
    return mask


TRIFECTA_MASK = _build_triple_mask()  # (120, 720)


def permutation_probs(theta: np.ndarray) -> np.ndarray:
    """theta: shape (N, 6) の正の強さスコア。戻り値: shape (N, 720) の各順列の確率。
    メモリ節約のためNは呼び出し側でバッチ分割すること(目安: 1バッチ2万件程度)。
    """
    n = theta.shape[0]
    gathered = theta[:, ALL_PERMS]  # (N, 720, 6): 各順列順に並べたθ
    remaining = np.cumsum(gathered[:, :, ::-1], axis=2)[:, :, ::-1]  # 各位置以降の残り合計
    ratios = gathered / remaining  # (N, 720, 6)
    probs = np.prod(ratios, axis=2)  # (N, 720)
    return probs


def pair_probabilities(theta: np.ndarray, batch_size: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """theta: shape (N, 6)。戻り値: (quinella_probs, wide_probs) 各 shape (N, 15)。
    PAIRS[k] = (i, j) (0-indexed艇番) に対応。
    """
    n = theta.shape[0]
    quinella_out = np.zeros((n, N_PAIRS))
    wide_out = np.zeros((n, N_PAIRS))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        perm_probs = permutation_probs(theta[start:end])  # (b, 720)
        quinella_out[start:end] = perm_probs @ QUINELLA_MASK.T
        wide_out[start:end] = perm_probs @ WIDE_MASK.T
    return quinella_out, wide_out


def pair_label(pair_idx: int) -> str:
    a, b = PAIRS[pair_idx]
    return f"{a + 1}-{b + 1}"  # 1-indexed艇番表記


def trifecta_probabilities(theta: np.ndarray, batch_size: int = 20000) -> np.ndarray:
    """theta: shape (N, 6)。戻り値: shape (N, 120) の3連単的中確率。
    TRIPLES[k] = (1着, 2着, 3着) (0-indexed艇番) に対応。
    """
    n = theta.shape[0]
    out = np.zeros((n, N_TRIPLES))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        perm_probs = permutation_probs(theta[start:end])  # (b, 720)
        out[start:end] = perm_probs @ TRIFECTA_MASK.T
    return out


def triple_label(triple_idx: int) -> str:
    a, b, c = TRIPLES[triple_idx]
    return f"{a + 1}-{b + 1}-{c + 1}"  # 1-indexed艇番表記
