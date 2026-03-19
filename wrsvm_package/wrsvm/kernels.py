"""Kernel functions with optional GPU acceleration via PyTorch."""

from __future__ import annotations

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5,
               backend: str = "numpy") -> np.ndarray:
    """Compute the RBF kernel K(x, y) = exp(-gamma * ||x - y||^2).

    Parameters
    ----------
    X1, X2 : ndarray of shape (n1, d) and (n2, d)
    gamma : float, default 0.5
    backend : {"numpy", "torch"}
        If "torch", uses PyTorch (GPU if available).

    Returns
    -------
    K : ndarray of shape (n1, n2)
    """
    if backend == "torch":
        return _rbf_kernel_torch(X1, X2, gamma)
    return _rbf_kernel_numpy(X1, X2, gamma)


def _rbf_kernel_numpy(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    D2 = sq1 + sq2 - 2.0 * X1 @ X2.T
    np.maximum(D2, 0.0, out=D2)
    return np.exp(-gamma * D2)


def _rbf_kernel_torch(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for backend='torch'. "
            "Install with: pip install wrsvm[gpu]"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t1 = torch.as_tensor(X1, dtype=torch.float64, device=device)
    t2 = torch.as_tensor(X2, dtype=torch.float64, device=device)
    sq1 = (t1 ** 2).sum(dim=1, keepdim=True)
    sq2 = (t2 ** 2).sum(dim=1, keepdim=True).T
    D2 = sq1 + sq2 - 2.0 * t1 @ t2.T
    D2.clamp_(min=0.0)
    K = torch.exp(-gamma * D2)
    return K.cpu().numpy()
