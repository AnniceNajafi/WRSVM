# WRSVM.jl

Julia implementation of the Weighted Relaxed Support Vector Machine for multiclass classification under class imbalance and label noise.

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/WRSVM.jl")
Pkg.instantiate()
```

## Quick start

```julia
using WRSVM

X = randn(100, 4)
y = rand(1:3, 100)
y_noisy = inject_outliers_minority(X, y; outlier_rate = 0.3, seed = 0)

model = solve_crammer_singer(X, y_noisy; C = 100.0, gamma = 0.1, upsilon = 0.2)
preds = predict_cs(model, X)
accuracy = mean(preds .== y)
```

## Formulations

- `solve_crammer_singer`: Crammer-Singer direct formulation (N x K duals)
- `solve_simmsvm`: SimMSVM simultaneous formulation (N duals, ~10x faster)

## Reference

Najafi, A. & Razzaghi, T. (2026). *Open-Source Python, R, and Julia Toolkit for Robust Multiclass Classification Under Class Imbalance and Label Noise.*
