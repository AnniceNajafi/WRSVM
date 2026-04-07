# WRSVMr

R interface to the `wrsvm` Python package for Weighted Relaxed Support Vector Machine classification under class imbalance and label noise.

## Installation

```r
# 1) install the Python backend (one-time)
reticulate::py_install("wrsvm")           # or from the terminal: pip install wrsvm

# 2) install this R package
remotes::install_local("path/to/WRSVMr")

# 3) if you already have a Python with wrsvm installed system-wide, point
#    reticulate at it BEFORE loading WRSVMr (add to .Renviron or session):
Sys.setenv(RETICULATE_PYTHON = "/path/to/your/python")
library(WRSVMr)
```

## Usage

```r
library(WRSVMr)

data(iris)
X <- scale(as.matrix(iris[, 1:4]))
y <- as.integer(iris$Species)

y_noisy <- inject_outliers_minority(X, y, outlier_rate = 0.3, seed = 0)

fit <- wrsvm_fit(X, y_noisy,
                 strategy = "cs",
                 C = 100, gamma = 0.1, upsilon = 0.2)

preds <- wrsvm_predict(fit, X)
mean(preds == y)
```

## Decomposition strategies

| strategy    | Dual size | Notes |
|-------------|-----------|-------|
| `"cs"`      | N * K     | Crammer-Singer direct formulation |
| `"simmsvm"` | N         | ~10x faster than `cs` |
| `"ovo"`     | per pair  | One-vs-One, best for imbalanced data with K >= 4 |
| `"ovr"`     | per class | One-vs-Rest, K binary solves |
