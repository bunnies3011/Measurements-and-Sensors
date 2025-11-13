# solomon.py — approximate SOLOMON splitter for two equivalent halves
# Usage:
#   from solomon import solomon_split
#   split = solomon_split(X, test_ratio=0.5)
#   train_idx, test_idx = split['idx_A'], split['idx_B']

import numpy as np
from typing import Tuple, Dict, Any

def _safe_corr(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    Z = Xc / std
    return np.corrcoef(Z, rowvar=False)

def _kmo(X: np.ndarray) -> float:
    R = _safe_corr(X)
    invR = np.linalg.pinv(R)
    D = np.diag(1 / np.sqrt(np.diag(invR)))
    P = -D @ (invR - np.diag(np.diag(invR))) @ D
    np.fill_diagonal(P, 0.0)
    r2 = float((R**2).sum() - np.diag(R**2).sum())
    p2 = float((P**2).sum() - np.diag(P**2).sum())
    denom = r2 + p2
    return 0.0 if denom == 0.0 else float(r2 / denom)

def _communalities_pca(X: np.ndarray, n_factors: int | None = None) -> Tuple[np.ndarray, float]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    eigvals = (Svals**2) / max(n - 1, 1)
    loadings = Vt.T * np.sqrt(eigvals)
    p = X.shape[1]
    if n_factors is None:
        k = int(np.sum(eigvals > 1.0))
        if k < 1:
            k = 1
    else:
        k = max(1, min(int(n_factors), p))
    Lk = loadings[:, :k]
    comm = np.sum(Lk**2, axis=1)
    S_ratio = float(comm.sum() / p)
    return comm, S_ratio

def _objective(XA: np.ndarray, XB: np.ndarray, w_corr=1.0, w_kmo=1.0, w_comm=1.0, n_factors=None) -> float:
    RA = _safe_corr(XA); RB = _safe_corr(XB)
    corr_dist = float(np.linalg.norm(RA - RB, ord='fro'))
    kA = _kmo(XA); kB = _kmo(XB)
    kmo_dist = abs(kA - kB)
    _, SA = _communalities_pca(XA, n_factors)
    _, SB = _communalities_pca(XB, n_factors)
    comm_dist = abs(SA - SB)
    return float(w_corr * corr_dist + w_kmo * kmo_dist + w_comm * comm_dist)

def solomon_split(
    X: np.ndarray,
    test_ratio: float = 0.5,
    seed: int = 42,
    max_iter: int = 2000,
    w_corr: float = 1.0,
    w_kmo: float = 1.0,
    w_comm: float = 1.0,
    n_factors: int | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    nB = int(round(n * test_ratio))
    nB = max(1, min(n - 1, nB))
    nA = n - nB

    # Init via 1st PC alternating along the range
    Xc = X - X.mean(axis=0, keepdims=True)
    U, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = (Xc @ Vt.T[:, 0]).ravel()
    order = np.argsort(pc1)
    idx_A = []
    idx_B = []
    for i, idx in enumerate(order):
        if len(idx_B) < nB and (i % 2 == 1 or len(idx_A) >= nA):
            idx_B.append(idx)
        else:
            idx_A.append(idx)
    idx_A = np.array(idx_A, dtype=int)
    idx_B = np.array(idx_B, dtype=int)

    XA, XB = X[idx_A], X[idx_B]
    best = _objective(XA, XB, w_corr, w_kmo, w_comm, n_factors)

    for _ in range(max_iter):
        ia = rng.integers(0, len(idx_A))
        ib = rng.integers(0, len(idx_B))
        a_idx, b_idx = idx_A[ia], idx_B[ib]

        # swap virtual
        XA_new = XA.copy(); XB_new = XB.copy()
        XA_new[np.where(idx_A == a_idx)[0][0]] = X[b_idx]
        XB_new[np.where(idx_B == b_idx)[0][0]] = X[a_idx]

        score_new = _objective(XA_new, XB_new, w_corr, w_kmo, w_comm, n_factors)
        if score_new + 1e-12 < best:
            best = score_new
            idx_A[ia], idx_B[ib] = b_idx, a_idx
            XA, XB = XA_new, XB_new

    RA = _safe_corr(XA); RB = _safe_corr(XB)
    kA = _kmo(XA); kB = _kmo(XB)
    _, SA = _communalities_pca(XA, n_factors); _, SB = _communalities_pca(XB, n_factors)
    return {
        "idx_A": np.sort(idx_A),
        "idx_B": np.sort(idx_B),
        "objective": float(best),
        "kmo_A": float(kA),
        "kmo_B": float(kB),
        "S_A": float(SA),
        "S_B": float(SB),
        "corr_frobenius": float(np.linalg.norm(RA - RB, ord='fro')),
    }
