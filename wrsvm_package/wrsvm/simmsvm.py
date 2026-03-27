"""SimMSVM formulation of WRSVM: one dual variable per sample.

Unlike the Crammer-Singer formulation (which uses N*K dual variables),
SimMSVM uses N dual variables and a specially-constructed Gram matrix:

    G[i, j] = (K / (K-1)) * k(x_i, x_j)     if y_i == y_j
            = -K / (K-1)^2 * k(x_i, x_j)    otherwise

Same prediction structure with fewer variables; typically 3-5x faster
than the Crammer-Singer reduction for the same problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from wrsvm.kernels import rbf_kernel
from wrsvm.solver import _solve_with_fallback


@dataclass
class SimMSVMResult:
    alpha: np.ndarray
    theta: np.ndarray
    beta: np.ndarray
    b: np.ndarray
    X_train: np.ndarray
    classes: np.ndarray
    K_cls: int
    n_c: np.ndarray
    gamma: float
    C: float
    upsilon: float


def solve_simmsvm(X: np.ndarray, y: np.ndarray, C: float, gamma: float,
                  upsilon: float = 0.3,
                  solver: str = "CLARABEL",
                  kernel_backend: str = "numpy") -> SimMSVMResult:
    """Solve the SimMSVM (simultaneous multiclass) WRSVM dual QP.

    Parameters match :func:`wrsvm.solver.solve_crammer_singer`.
    """
    X = np.asarray(X, dtype=np.float64)
    classes = np.sort(np.unique(y))
    K_cls = len(classes)
    N = X.shape[0]
    y_idx = np.searchsorted(classes, y)
    n_c = np.bincount(y_idx, minlength=K_cls).astype(np.float64)

    K_mat = rbf_kernel(X, X, gamma=gamma, backend=kernel_backend)

    same_class = y_idx[:, None] == y_idx[None, :]
    coef_same = K_cls / (K_cls - 1)
    coef_diff = -K_cls / (K_cls - 1) ** 2
    G = np.where(same_class, coef_same * K_mat, coef_diff * K_mat)

    w_diag = n_c[y_idx] / C
    G[np.arange(N), np.arange(N)] += w_diag + 1e-7
    G = 0.5 * (G + G.T)

    alpha_var = cp.Variable(N, nonneg=True)
    obj_expr = -cp.sum(alpha_var) + 0.5 * cp.quad_form(alpha_var, cp.psd_wrap(G))
    constraints = []

    beta_var = cp.Variable(K_cls, nonneg=True)
    obj_expr = obj_expr + n_c @ (upsilon * beta_var)
    for k in range(K_cls):
        idx_k = np.where(y_idx == k)[0]
        if len(idx_k) > 0:
            constraints.append(alpha_var[idx_k.tolist()] <= beta_var[k])

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    _solve_with_fallback(prob, solver, alpha_var)
    if alpha_var.value is None:
        raise RuntimeError("SimMSVM solver failed on primary + SCS fallback.")

    alpha_vals = np.asarray(alpha_var.value).flatten()
    alpha_vals[alpha_vals < 1e-7] = 0.0

    beta_vals = np.asarray(beta_var.value).flatten()

    V = np.full((N, K_cls), -1.0 / (K_cls - 1))
    V[np.arange(N), y_idx] = 1.0
    theta = V * alpha_vals[:, None]

    b = _recover_biases_simmsvm(alpha_vals, y_idx, K_mat, theta, n_c, C,
                                 beta_vals, K_cls)

    return SimMSVMResult(
        alpha=alpha_vals, theta=theta, beta=beta_vals, b=b,
        X_train=X, classes=classes, K_cls=K_cls, n_c=n_c,
        gamma=gamma, C=C, upsilon=upsilon,
    )


def _recover_biases_simmsvm(alpha: np.ndarray, y_idx: np.ndarray,
                             K_mat: np.ndarray, theta: np.ndarray,
                             n_c: np.ndarray, C: float, beta: np.ndarray,
                             K_cls: int) -> np.ndarray:
    K_scores = K_mat @ theta
    sv = np.where(alpha > 1e-6)[0]
    if len(sv) == 0:
        return np.zeros(K_cls)

    eq_rows, rhs = [], []
    for i in sv:
        ci = y_idx[i]
        bc = beta[ci]
        is_free = alpha[i] > 1e-6 and (not np.isfinite(bc) or alpha[i] < bc - 1e-4)
        if not is_free:
            continue
        s_own = K_scores[i, ci]
        s_others = K_scores[i].sum() - s_own
        composite = s_own - s_others / (K_cls - 1)
        xi_i = n_c[ci] * alpha[i] / C
        rhs_i = 1.0 - xi_i - composite
        row = np.full(K_cls, -1.0 / (K_cls - 1))
        row[ci] = 1.0
        eq_rows.append(row)
        rhs.append(rhs_i)

    if len(rhs) == 0:
        return np.zeros(K_cls)

    A = np.vstack(eq_rows)
    A = np.vstack([A, np.ones(K_cls)])
    rhs.append(0.0)
    try:
        b = np.linalg.lstsq(A, np.asarray(rhs), rcond=None)[0]
        if np.any(np.isnan(b)):
            b = np.zeros(K_cls)
    except np.linalg.LinAlgError:
        b = np.zeros(K_cls)
    return b


def predict_simmsvm(result: SimMSVMResult, X_new: np.ndarray,
                    kernel_backend: str = "numpy") -> np.ndarray:
    X_new = np.asarray(X_new, dtype=np.float64)
    K_new = rbf_kernel(X_new, result.X_train, gamma=result.gamma,
                       backend=kernel_backend)
    scores = K_new @ result.theta + result.b.reshape(1, -1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return result.classes[scores.argmax(axis=1)]
