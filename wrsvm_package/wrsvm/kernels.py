"""Kernel functions with optional GPU acceleration via PyTorch.

Supported kernels (all accept two matrices and return an n1-by-n2 Gram matrix):

- ``rbf``        :  K(x, y) = exp(-gamma * ||x - y||^2)
- ``linear``     :  K(x, y) = x . y
- ``poly``       :  K(x, y) = (gamma * x . y + coef0)^degree
- ``sigmoid``    :  K(x, y) = tanh(gamma * x . y + coef0)
- ``laplacian``  :  K(x, y) = exp(-gamma * ||x - y||_1)
"""

from __future__ import annotations

import numpy as np

VALID_KERNELS = ("rbf", "linear", "poly", "sigmoid", "laplacian")


def compute_kernel(X1: np.ndarray, X2: np.ndarray, kernel: str = "rbf",
                   gamma: float = 0.5, degree: int = 3, coef0: float = 0.0,
                   backend: str = "numpy") -> np.ndarray:
    """Unified kernel dispatcher.

    Parameters
    ----------
    X1, X2 : ndarray of shape (n1, d) and (n2, d)
    kernel : {"rbf", "linear", "poly", "sigmoid", "laplacian"}
    gamma  : kernel bandwidth (rbf, poly, sigmoid, laplacian)
    degree : polynomial degree (poly only)
    coef0  : polynomial/sigmoid bias (poly, sigmoid)
    backend : {"numpy", "torch"}; "torch" currently only accelerates rbf.
    """
    if kernel == "rbf":
        return rbf_kernel(X1, X2, gamma=gamma, backend=backend)
    if kernel == "linear":
        return linear_kernel(X1, X2)
    if kernel == "poly":
        return poly_kernel(X1, X2, gamma=gamma, degree=degree, coef0=coef0)
    if kernel == "sigmoid":
        return sigmoid_kernel(X1, X2, gamma=gamma, coef0=coef0)
    if kernel == "laplacian":
        return laplacian_kernel(X1, X2, gamma=gamma)
    raise ValueError(
        f"Unknown kernel '{kernel}'. Expected one of {VALID_KERNELS}."
    )


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5,
               backend: str = "numpy") -> np.ndarray:
    """K(x, y) = exp(-gamma * ||x - y||^2)."""
    if backend == "torch":
        return _rbf_kernel_torch(X1, X2, gamma)
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    D2 = sq1 + sq2 - 2.0 * X1 @ X2.T
    np.maximum(D2, 0.0, out=D2)
    return np.exp(-gamma * D2)


def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """K(x, y) = x . y."""
    return X1 @ X2.T


def poly_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5,
                degree: int = 3, coef0: float = 0.0) -> np.ndarray:
    """K(x, y) = (gamma * x . y + coef0)^degree."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def sigmoid_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5,
                   coef0: float = 0.0) -> np.ndarray:
    """K(x, y) = tanh(gamma * x . y + coef0)."""
    return np.tanh(gamma * (X1 @ X2.T) + coef0)


def laplacian_kernel(X1: np.ndarray, X2: np.ndarray,
                     gamma: float = 0.5) -> np.ndarray:
    """K(x, y) = exp(-gamma * ||x - y||_1)."""
    n1, n2 = X1.shape[0], X2.shape[0]
    D1 = np.abs(X1[:, None, :] - X2[None, :, :]).sum(axis=2)
    return np.exp(-gamma * D1)


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
