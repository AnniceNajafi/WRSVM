"""Label noise injection procedures for imbalanced classification."""

from __future__ import annotations

import numpy as np


def inject_outliers_majority(X: np.ndarray, y: np.ndarray,
                              outlier_rate: float = 0.14,
                              seed: int | None = None) -> np.ndarray:
    """Flip labels of majority-class samples farthest from the majority centroid.

    This is the original noise model from the WRSVM paper. Picks
    round(outlier_rate * n) majority samples with largest Euclidean
    distance from the majority centroid, and reassigns each to a
    uniformly random minority class.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    outlier_rate : float in [0, 1]
        Fraction of total samples to flip.
    seed : int, optional

    Returns
    -------
    y_corrupted : ndarray of shape (n,)
    """
    if outlier_rate <= 0:
        return y.copy()

    rng = np.random.default_rng(seed)
    classes, sizes = np.unique(y, return_counts=True)
    maj_class = classes[np.argmax(sizes)]
    min_classes = classes[classes != maj_class]
    if len(min_classes) == 0:
        return y.copy()

    n = len(y)
    n_flip = int(round(n * outlier_rate))
    maj_idx = np.where(y == maj_class)[0]
    centroid = X[maj_idx].mean(axis=0)
    dists = np.sum((X[maj_idx] - centroid) ** 2, axis=1)
    n_flip = min(n_flip, len(maj_idx))
    flip_local = np.argsort(dists)[::-1][:n_flip]
    flip_global = maj_idx[flip_local]

    y_out = y.copy()
    for i in flip_global:
        y_out[i] = rng.choice(min_classes)
    return y_out


def inject_outliers_minority(X: np.ndarray, y: np.ndarray,
                              outlier_rate: float = 0.3,
                              seed: int | None = None) -> np.ndarray:
    """Flip labels of a fraction of each minority class to a random other class.

    For each minority class k, picks round(outlier_rate * n_k) samples
    uniformly at random and reassigns each to a uniformly random class
    drawn from all classes except k. The majority class is left untouched.

    This is the noise model introduced in Najafi & Razzaghi (2026) to
    stress-test WRSVM's per-class alpha cap.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Unused; kept for API symmetry with inject_outliers_majority.
    y : ndarray of shape (n,)
    outlier_rate : float in [0, 1]
        Fraction of each minority class to flip.
    seed : int, optional

    Returns
    -------
    y_corrupted : ndarray of shape (n,)
    """
    if outlier_rate <= 0:
        return y.copy()

    rng = np.random.default_rng(seed)
    classes, sizes = np.unique(y, return_counts=True)
    maj_class = classes[np.argmax(sizes)]
    min_classes = classes[classes != maj_class]
    if len(min_classes) == 0:
        return y.copy()

    y_out = y.copy()
    for mc in min_classes:
        idx = np.where(y == mc)[0]
        n_flip = int(round(len(idx) * outlier_rate))
        if n_flip == 0:
            continue
        flip_idx = rng.choice(idx, size=n_flip, replace=False)
        other_classes = classes[classes != mc]
        for i in flip_idx:
            y_out[i] = rng.choice(other_classes)
    return y_out
