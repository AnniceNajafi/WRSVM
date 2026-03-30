"""One-vs-One and One-vs-Rest decompositions for WRSVM.

Both decompositions reuse the binary WRSVM solver. OVO trains K*(K-1)/2
pairwise binary classifiers; OVR trains K one-vs-rest binary classifiers.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from wrsvm.solver import SolverResult, predict as _predict, solve_crammer_singer


class OVOClassifier:
    """Multiclass WRSVM via pairwise binary solves + majority vote."""

    def __init__(self, C: float = 100.0, gamma: float = 0.1,
                 upsilon: float = 0.2,
                 solver: str = "CLARABEL", kernel_backend: str = "numpy"):
        self.C = C
        self.gamma = gamma
        self.upsilon = upsilon
        self.solver = solver
        self.kernel_backend = kernel_backend

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        self.pair_models_ = {}
        for ci, cj in combinations(range(self.n_classes_), 2):
            lbl_i, lbl_j = self.classes_[ci], self.classes_[cj]
            mask = (y == lbl_i) | (y == lbl_j)
            X_pair = X[mask]
            y_pair = y[mask]
            res = solve_crammer_singer(
                X_pair, y_pair, C=self.C, gamma=self.gamma,
                upsilon=self.upsilon,
                solver=self.solver, kernel_backend=self.kernel_backend,
            )
            self.pair_models_[(ci, cj)] = res
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=np.int64)
        for (ci, cj), model in self.pair_models_.items():
            preds = _predict(model, X, kernel_backend=self.kernel_backend)
            for idx, p in enumerate(preds):
                winner = ci if p == self.classes_[ci] else cj
                votes[idx, winner] += 1
        return self.classes_[votes.argmax(axis=1)]


class OVRClassifier:
    """Multiclass WRSVM via one-vs-rest binary solves."""

    def __init__(self, C: float = 100.0, gamma: float = 0.1,
                 upsilon: float = 0.2,
                 solver: str = "CLARABEL", kernel_backend: str = "numpy"):
        self.C = C
        self.gamma = gamma
        self.upsilon = upsilon
        self.solver = solver
        self.kernel_backend = kernel_backend

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        self.rest_models_ = []
        for k in range(self.n_classes_):
            y_bin = np.where(y == self.classes_[k], 1, -1)
            res = solve_crammer_singer(
                X, y_bin, C=self.C, gamma=self.gamma,
                upsilon=self.upsilon,
                solver=self.solver, kernel_backend=self.kernel_backend,
            )
            self.rest_models_.append(res)
        return self

    def _binary_score(self, model: SolverResult, X: np.ndarray) -> np.ndarray:
        from wrsvm.kernels import rbf_kernel
        K_new = rbf_kernel(X, model.X_train, gamma=self.gamma,
                            backend=self.kernel_backend)
        scores = K_new @ model.theta + model.b.reshape(1, -1)
        pos_col = int(np.where(model.classes == 1)[0][0])
        neg_col = int(np.where(model.classes == -1)[0][0])
        return scores[:, pos_col] - scores[:, neg_col]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        all_scores = np.zeros((N, self.n_classes_))
        for k, model in enumerate(self.rest_models_):
            all_scores[:, k] = self._binary_score(model, X)
        return self.classes_[all_scores.argmax(axis=1)]
