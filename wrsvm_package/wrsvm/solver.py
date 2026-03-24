"""Crammer-Singer multiclass QP solver for SVM / RSVM / WSVM / WRSVM."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from wrsvm.kernels import rbf_kernel


def _solve_with_fallback(prob: cp.Problem, solver: str, witness_var: cp.Variable) -> None:
    """Solve a CVXPY problem with the requested solver and fall back to SCS on failure.

    Supported ``solver`` values:

    - ``"CLARABEL"`` (default, CPU)
    - ``"SCS"`` (CPU)
    - ``"SCS_GPU"`` — requires ``scs`` built with CUDA support
    - ``"GUROBI"`` — requires ``gurobipy`` and a Gurobi license (free academic
      license available). On small problems the restricted license included
      with ``pip install gurobipy`` is sufficient.
    - Any other name passed through to CVXPY's solver registry.
    """
    solver_upper = solver.upper()
    if solver_upper == "SCS_GPU":
        try:
            prob.solve(solver="SCS", use_indirect=True, gpu=True)
        except (cp.SolverError, TypeError, ValueError, RuntimeError):
            prob.solve(solver="SCS", max_iters=50000, eps=1e-7)
    elif solver_upper == "GUROBI":
        try:
            prob.solve(solver="GUROBI")
        except Exception:
            prob.solve(solver="SCS", max_iters=50000, eps=1e-7)
    else:
        try:
            prob.solve(solver=solver)
        except Exception:
            prob.solve(solver="SCS", max_iters=50000, eps=1e-7)
    if witness_var.value is None:
        prob.solve(solver="SCS", max_iters=50000, eps=1e-7)


@dataclass
class SolverResult:
    """Solution of the Crammer-Singer WRSVM dual QP."""

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


def _build_hessian(K_mat: np.ndarray, pos_mask: np.ndarray,
                   w_diag: np.ndarray) -> np.ndarray:
    """Build the (NK, NK) dense Hessian for the Crammer-Singer dual.

    Uses column-major (Fortran) flat indexing: alpha_flat[j*N + i] = alpha[i, j].
    """
    N, K_cls = pos_mask.shape
    NK = N * K_cls
    P = np.zeros((NK, NK), dtype=np.float64)

    for k in range(K_cls):
        p_k = pos_mask[:, k]
        s_k = 2.0 * p_k - 1.0
        pp = np.outer(p_k, p_k) * K_mat
        ps = np.outer(p_k, s_k) * K_mat
        sp = np.outer(s_k, p_k) * K_mat
        ss = np.outer(s_k, s_k) * K_mat

        for j1 in range(K_cls):
            r_idx = slice(j1 * N, (j1 + 1) * N)
            for j2 in range(K_cls):
                c_idx = slice(j2 * N, (j2 + 1) * N)
                if j1 != k and j2 != k:
                    P[r_idx, c_idx] += pp
                elif j1 != k and j2 == k:
                    P[r_idx, c_idx] += ps
                elif j1 == k and j2 != k:
                    P[r_idx, c_idx] += sp
                else:
                    P[r_idx, c_idx] += ss

    diag_add = np.tile(w_diag, K_cls)
    P[np.arange(NK), np.arange(NK)] += diag_add
    P = 0.5 * (P + P.T)
    return P


def _build_equality(pos_mask: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    """Build the (K, NK) equality constraint matrix: A_eq @ alpha_flat = 0."""
    N, K_cls = pos_mask.shape
    NK = N * K_cls
    A_eq = np.zeros((K_cls, NK), dtype=np.float64)

    for k in range(K_cls):
        for j in range(K_cls):
            col_idx = slice(j * N, (j + 1) * N)
            if j != k:
                A_eq[k, col_idx] += pos_mask[:, k]
            else:
                A_eq[k, col_idx] += pos_mask[:, k] - neg_mask[:, k]
    return A_eq


def _recover_biases(alpha: np.ndarray, y: np.ndarray, K_mat: np.ndarray,
                    theta: np.ndarray, n_c: np.ndarray, C: float,
                    beta: np.ndarray, n_passes: int = 3) -> np.ndarray:
    """Recover bias terms b_c via free support vector conditions.

    Iterates a small linear system built from the KKT stationarity
    condition for each free SV (not bounded by beta).
    """
    N, K_cls = alpha.shape
    K_scores = K_mat @ theta
    w_xi = n_c[y] / C

    b = np.zeros(K_cls)
    for _ in range(n_passes):
        diff_sum = np.zeros((K_cls, K_cls))
        diff_count = np.zeros((K_cls, K_cls), dtype=np.int64)

        for i in range(N):
            ci = y[i]
            bc = beta[ci]
            for k in range(K_cls):
                if k == ci:
                    continue
                aik = alpha[i, k]
                is_free = aik > 1e-6 and (
                    not np.isfinite(bc) or aik < bc - 1e-4
                )
                if is_free:
                    xi_ik = w_xi[i] * aik
                    obs = (1.0 - xi_ik) - (K_scores[i, ci] - K_scores[i, k])
                    diff_sum[ci, k] += obs
                    diff_count[ci, k] += 1

        active = np.argwhere(diff_count > 0)
        if len(active) == 0:
            scores_full = K_scores + b.reshape(1, -1)
            for i in range(N):
                ci = y[i]
                if alpha[i].sum() < 1e-6:
                    continue
                sc = scores_full[i].copy()
                sc[ci] = -np.inf
                jstar = int(np.argmax(sc))
                xi_ij = w_xi[i] * alpha[i, jstar]
                obs = (1.0 - xi_ij) - (K_scores[i, ci] - K_scores[i, jstar])
                diff_sum[ci, jstar] += obs
                diff_count[ci, jstar] += 1
            active = np.argwhere(diff_count > 0)
            if len(active) == 0:
                break

        n_eqs = len(active) + 1
        M_sys = np.zeros((n_eqs, K_cls))
        rhs = np.zeros(n_eqs)
        for r, (c_r, j_r) in enumerate(active):
            M_sys[r, c_r] = 1.0
            M_sys[r, j_r] = -1.0
            rhs[r] = diff_sum[c_r, j_r] / diff_count[c_r, j_r]
        M_sys[-1, :] = 1.0
        rhs[-1] = 0.0

        try:
            b = np.linalg.lstsq(M_sys, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            b = np.zeros(K_cls)

    if np.any(np.isnan(b)):
        b = np.zeros(K_cls)
    return b


def solve_crammer_singer(X: np.ndarray, y: np.ndarray, C: float, gamma: float,
                          upsilon: float = 0.3,
                          solver: str = "CLARABEL",
                          kernel_backend: str = "numpy") -> SolverResult:
    """Solve the Crammer-Singer WRSVM dual QP.

    Parameters
    ----------
    X : ndarray of shape (N, d)
    y : ndarray of shape (N,)
    C : float > 0
    gamma : float > 0
    upsilon : float in (0, 1]
    solver : CVXPY solver name ("CLARABEL", "SCS", etc.)
    kernel_backend : {"numpy", "torch"}

    Returns
    -------
    SolverResult
    """
    X = np.asarray(X, dtype=np.float64)
    classes = np.sort(np.unique(y))
    K_cls = len(classes)
    N = X.shape[0]
    y_idx = np.searchsorted(classes, y)
    n_c = np.bincount(y_idx, minlength=K_cls).astype(np.float64)

    K_mat = rbf_kernel(X, X, gamma=gamma, backend=kernel_backend)
    K_mat = K_mat + np.eye(N) * 1e-6

    pos_mask = np.eye(K_cls)[y_idx]
    neg_mask = 1.0 - pos_mask

    w_diag = n_c[y_idx] / C

    P = _build_hessian(K_mat, pos_mask, w_diag)
    A_eq = _build_equality(pos_mask, neg_mask)

    NK = N * K_cls
    zero_idx = [y_idx[i] * N + i for i in range(N)]

    alpha_flat = cp.Variable(NK, nonneg=True)
    obj_expr = -cp.sum(alpha_flat) + 0.5 * cp.quad_form(alpha_flat, cp.psd_wrap(P))
    constraints = [
        alpha_flat[zero_idx] == 0,
        A_eq @ alpha_flat == 0,
    ]

    beta_var = cp.Variable(K_cls, nonneg=True)
    obj_expr = obj_expr + n_c @ ((K_cls - 1) * upsilon * beta_var)
    for k in range(K_cls):
        idx_k = np.where(y_idx == k)[0]
        if len(idx_k) == 0:
            continue
        for j in range(K_cls):
            flat_idx = (idx_k + j * N).tolist()
            constraints.append(alpha_flat[flat_idx] <= beta_var[k])

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    _solve_with_fallback(prob, solver, alpha_flat)
    if alpha_flat.value is None:
        raise RuntimeError("QP solver failed on primary + SCS fallback.")

    alpha_vals = np.asarray(alpha_flat.value).reshape((N, K_cls), order="F").copy()
    alpha_vals[alpha_vals < 1e-7] = 0.0

    beta_vals = np.asarray(beta_var.value).flatten()

    theta = pos_mask * alpha_vals.sum(axis=1, keepdims=True) - neg_mask * alpha_vals

    b = _recover_biases(alpha_vals, y_idx, K_mat, theta, n_c, C, beta_vals)

    return SolverResult(
        alpha=alpha_vals, theta=theta, beta=beta_vals, b=b,
        X_train=X, classes=classes, K_cls=K_cls, n_c=n_c,
        gamma=gamma, C=C, upsilon=upsilon,
    )


def predict(result: SolverResult, X_new: np.ndarray,
            kernel_backend: str = "numpy") -> np.ndarray:
    """Predict class labels for new samples."""
    X_new = np.asarray(X_new, dtype=np.float64)
    K_new = rbf_kernel(X_new, result.X_train, gamma=result.gamma,
                       backend=kernel_backend)
    scores = K_new @ result.theta + result.b.reshape(1, -1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pred_idx = scores.argmax(axis=1)
    return result.classes[pred_idx]
