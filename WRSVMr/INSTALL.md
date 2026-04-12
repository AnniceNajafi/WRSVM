# Installing `WRSVMr` (R)

Detailed installation guide for the R interface to WRSVM. `WRSVMr` is a thin reticulate wrapper around the Python `wrsvm` package — installing R alone is not enough; a working Python with `wrsvm` must be reachable from R.

---

## 1. System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| R           | 4.0     | 4.4+ |
| Python      | 3.9     | 3.11+ (located by reticulate) |
| OS          | Linux, macOS, Windows 10/11 | — |
| RAM         | 4 GB    | 8 GB |

R packages installed automatically:

| Package      | Min version | Purpose |
|--------------|-------------|---------|
| `reticulate` | 1.30        | bridge to Python |

Suggested (for tests):

| Package      | Min version | Purpose |
|--------------|-------------|---------|
| `testthat`   | 3.0         | unit tests |

---

## 2. Two-step install

The package does **not** install Python for you. You must (a) get a Python that has the `wrsvm` package, then (b) install the R wrapper.

### Step A — Python backend

You have three options, in order of simplicity.

**A1. Let reticulate manage a private Python**

```r
install.packages("reticulate")
reticulate::install_python(version = "3.11.9")
reticulate::py_install("wrsvm", pip = TRUE)
```

**A2. Install into an existing Python on the system**

```bash
# from a regular shell
python -m pip install wrsvm
```

Then point reticulate at that interpreter (see Step C).

**A3. Install from the local clone (current state of this repo)**

```bash
cd C:/Users/annic/Documents/WRSVM/wrsvm_package
python -m pip install -e .
```

### Step B — R wrapper

From a local clone:

```r
# from R
install.packages("remotes")
remotes::install_local("C:/Users/annic/Documents/WRSVM/WRSVMr")
```

Or, if you already cloned and are working inside it:

```r
setwd("C:/Users/annic/Documents/WRSVM/WRSVMr")
devtools::install()        # requires `devtools`
```

### Step C — Tell reticulate which Python to use (only if multiple exist)

`reticulate` has a discovery order that often picks the wrong interpreter on Windows. Force it explicitly:

```r
# in ~/.Renviron (persistent across sessions):
RETICULATE_PYTHON=C:/Users/annic/AppData/Local/Programs/Python/Python311/python.exe

# or inside an R session, BEFORE library(WRSVMr):
Sys.setenv(RETICULATE_PYTHON = "C:/path/to/python.exe")
library(WRSVMr)
```

Verify reticulate sees it:

```r
reticulate::py_config()
reticulate::py_module_available("wrsvm")   # TRUE
```

---

## 3. Verifying the install

```r
library(WRSVMr)

data(iris)
X <- scale(as.matrix(iris[, 1:4]))
y <- as.integer(iris$Species)

fit <- wrsvm_fit(X, y, strategy = "cs",
                 C = 100, gamma = 0.1, upsilon = 0.2)

mean(wrsvm_predict(fit, X) == y)      # expect ~0.97+
```

Run the bundled tests:

```r
setwd("C:/Users/annic/Documents/WRSVM/WRSVMr")
testthat::test_local()
```

---

## 4. Solver selection

All Python `solver=` values pass through:

```r
fit <- wrsvm_fit(X, y, solver = "GUROBI")     # requires gurobipy in the Python env
fit <- wrsvm_fit(X, y, solver = "CLARABEL")   # default, no extra setup
```

For GPU kernels:

```r
fit <- wrsvm_fit(X, y, kernel_backend = "torch")   # requires torch in the Python env
```

Install those extras into the same Python that reticulate is using:

```r
reticulate::py_install(c("gurobipy", "torch"), pip = TRUE)
```

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error in py_module_import("wrsvm") : ModuleNotFoundError` | reticulate is using a different Python than the one that has `wrsvm` | set `RETICULATE_PYTHON`, restart R |
| `Error: package 'reticulate' could not be loaded` | reticulate not installed | `install.packages("reticulate")` |
| `pip install wrsvm` fails on Windows ARM | clarabel wheel missing | use Python 3.11 x64 build |
| `wrsvm_fit` hangs on first call | reticulate is downloading miniconda silently | run `reticulate::py_config()` to confirm interpreter; consider `use_python(..., required=TRUE)` |
| `cannot find function 'wrsvm_fit'` after install | package installed but not loaded | `library(WRSVMr)` |
| Different Python in RStudio vs. terminal | RStudio reads project-specific `.Renviron` | put `RETICULATE_PYTHON=...` in the project's `.Renviron`, not just the user one |

---

## 6. Uninstalling

```r
remove.packages("WRSVMr")
```

Optionally remove the Python backend:

```r
reticulate::py_install("wrsvm", pip_ignore_installed = FALSE)   # to upgrade
# or, from a shell:
python -m pip uninstall wrsvm
```
