"""Scikit-learn compatible WRSVM classifier.

Supports four decomposition strategies: Crammer-Singer (``"cs"``),
SimMSVM (``"simmsvm"``), One-vs-One (``"ovo"``), and One-vs-Rest (``"ovr"``).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y

from wrsvm.decomposition import OVOClassifier, OVRClassifier
from wrsvm.simmsvm import predict_simmsvm, solve_simmsvm
from wrsvm.solver import predict as _predict_cs
from wrsvm.solver import solve_crammer_singer


class WRSVMClassifier(BaseEstimator, ClassifierMixin):
    """Weighted Relaxed SVM for multiclass classification.

    Solves one of four decomposition strategies:

    - ``"cs"`` — Crammer-Singer direct formulation (N * K dual variables)
    - ``"simmsvm"`` — Simultaneous multiclass with one dual per sample (N vars)
    - ``"ovo"`` — One-vs-One pairwise decomposition
    - ``"ovr"`` — One-vs-Rest decomposition

    Parameters
    ----------
    strategy : {"cs", "simmsvm", "ovo", "ovr"}, default "cs"
    C : float, default 100.0
    kernel : {"rbf", "linear", "poly", "sigmoid", "laplacian"}, default "rbf"
    gamma : float, default 0.1
        Bandwidth for rbf / laplacian / sigmoid; feature scale for poly.
        Ignored by the linear kernel.
    degree : int, default 3
        Polynomial degree. Only used when ``kernel="poly"``.
    coef0 : float, default 0.0
        Independent term for ``kernel in {"poly", "sigmoid"}``.
    upsilon : float, default 0.2
    solver : str, default "CLARABEL"
        One of ``"CLARABEL"``, ``"SCS"``, ``"SCS_GPU"``, ``"GUROBI"``, or any
        other name recognized by CVXPY. Falls back to SCS on failure.
    kernel_backend : {"numpy", "torch"}, default "numpy"
    """

    def __init__(self, strategy: str = "cs",
                 C: float = 100.0, gamma: float = 0.1, upsilon: float = 0.2,
                 kernel: str = "rbf", degree: int = 3, coef0: float = 0.0,
                 solver: str = "CLARABEL", kernel_backend: str = "numpy"):
        self.strategy = strategy
        self.C = C
        self.gamma = gamma
        self.upsilon = upsilon
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.solver = solver
        self.kernel_backend = kernel_backend

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("At least two classes are required.")

        kw = dict(C=self.C, gamma=self.gamma, upsilon=self.upsilon,
                  kernel=self.kernel, degree=self.degree, coef0=self.coef0,
                  solver=self.solver, kernel_backend=self.kernel_backend)

        if self.strategy == "cs":
            self.result_ = solve_crammer_singer(X, y, **kw)
        elif self.strategy == "simmsvm":
            self.result_ = solve_simmsvm(X, y, **kw)
        elif self.strategy == "ovo":
            self.result_ = OVOClassifier(**kw).fit(X, y)
        elif self.strategy == "ovr":
            self.result_ = OVRClassifier(**kw).fit(X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")
        return self

    def predict(self, X):
        check_is_fitted(self, "result_")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        if self.strategy == "cs":
            return _predict_cs(self.result_, X,
                                kernel_backend=self.kernel_backend)
        if self.strategy == "simmsvm":
            return predict_simmsvm(self.result_, X,
                                    kernel_backend=self.kernel_backend)
        return self.result_.predict(X)

    def decision_function(self, X):
        """Return multiclass scores (N, K). Only supported for 'cs' and 'simmsvm'."""
        check_is_fitted(self, "result_")
        if self.strategy not in ("cs", "simmsvm"):
            raise NotImplementedError(
                f"decision_function is not implemented for strategy={self.strategy!r}; "
                "use predict()."
            )
        from wrsvm.kernels import compute_kernel
        X = np.asarray(X, dtype=np.float64)
        K_new = compute_kernel(X, self.result_.X_train,
                                kernel=self.kernel, gamma=self.gamma,
                                degree=self.degree, coef0=self.coef0,
                                backend=self.kernel_backend)
        return K_new @ self.result_.theta + self.result_.b.reshape(1, -1)
