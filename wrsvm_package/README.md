# WRSVM

Weighted Relaxed Support Vector Machine for multiclass classification under class imbalance and label noise.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[gpu]"      # CUDA-enabled PyTorch for GPU kernel matrix
pip install -e ".[gurobi]"   # Gurobi QP solver (requires license)
```

### Solver backends

- `solver="CLARABEL"` (default): open-source, works out of the box
- `solver="SCS"`: open-source fallback, first-order method
- `solver="SCS_GPU"`: SCS with CUDA support (requires `scs` built with CUDA)
- `solver="GUROBI"`: commercial, competitive on medium problems. Free restricted license via `pip install gurobipy` (up to 2000 variables); academic license available at no cost from Gurobi.

## Quick start

```python
from wrsvm import WRSVMClassifier, inject_outliers_minority
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

y_noisy = inject_outliers_minority(X, y, outlier_rate=0.3, seed=0)

clf = WRSVMClassifier(C=100.0, gamma=0.1, upsilon=0.2)
clf.fit(X, y_noisy)
print("Accuracy:", clf.score(X, y))
```

## Decomposition strategies

| strategy    | Dual size | Notes |
|-------------|-----------|-------|
| `"cs"`      | N * K     | Crammer-Singer direct formulation |
| `"simmsvm"` | N         | Simultaneous multiclass, ~10x faster than `cs` |
| `"ovo"`     | per pair  | One-vs-One pairwise, best for imbalanced data with K >= 4 |
| `"ovr"`     | per class | One-vs-Rest, K binary solves |

```python
clf = WRSVMClassifier(strategy="simmsvm", C=100, gamma=0.1, upsilon=0.2)
```

## Reference

Najafi, A. & Razzaghi, T. (2026). *Open-Source Python, R, and Julia Toolkit for Robust Multiclass Classification Under Class Imbalance and Label Noise.*
