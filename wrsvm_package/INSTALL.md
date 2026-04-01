# Installing `wrsvm` (Python)

Detailed installation guide for the Python implementation of the Weighted Relaxed Support Vector Machine.

---

## 1. System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python      | 3.9     | 3.11+ |
| OS          | Linux, macOS, Windows 10/11 | — |
| RAM         | 2 GB    | 8 GB (large kernel matrices) |
| Compiler    | not required (wheels for all deps) | — |
| GPU (opt.)  | CUDA 11.8+ for `torch` / `scs-gpu` | — |

Verify your interpreter:

```bash
python --version          # >= 3.9
python -m pip --version
```

---

## 2. Core install (CPU only)

### 2a. From a local clone (current state of the repo)

```bash
cd C:/Users/annic/Documents/WRSVM/wrsvm_package
python -m pip install -e .
```

The `-e` flag installs in *editable* mode so changes to the source tree are picked up without reinstalling.

### 2b. From a wheel (when published)

```bash
python -m pip install wrsvm
```

### 2c. Inside an isolated environment (recommended)

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (bash)
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Or with conda:

```bash
conda create -n wrsvm python=3.11
conda activate wrsvm
python -m pip install -e .
```

---

## 3. Dependencies pulled in automatically

| Package        | Min version | Purpose |
|----------------|-------------|---------|
| `numpy`        | 1.20        | dense linear algebra |
| `scipy`        | 1.7         | sparse ops, stats |
| `scikit-learn` | 1.0         | base classes, preprocessing |
| `cvxpy`        | 1.3         | QP modeling layer |
| `clarabel`     | 0.5         | default open-source QP solver |

---

## 4. Optional extras

### 4a. GPU kernel matrices (PyTorch backend)

```bash
python -m pip install -e ".[gpu]"
```

This installs `torch>=2.0`. For CUDA you must select the matching wheel from <https://pytorch.org/get-started/locally/>; the default extra installs the CPU build.

### 4b. Gurobi solver (commercial, ~1.5–2× faster)

```bash
python -m pip install -e ".[gurobi]"
```

`gurobipy` ships with a free *restricted* license that handles up to 2000 variables — sufficient for problems up to ~N = 600, K = 3. Larger problems require a free academic license (<https://www.gurobi.com/academia/>) or a commercial seat.

### 4c. Developer tools

```bash
python -m pip install -e ".[dev]"
```

Pulls in `pytest` and `pytest-cov`.

### 4d. Combine extras

```bash
python -m pip install -e ".[gpu,gurobi,dev]"
```

---

## 5. Solver backends (pick at call time)

| `solver=` | Bundled? | Notes |
|---|---|---|
| `"CLARABEL"`  | yes (default)         | interior-point, robust |
| `"SCS"`       | comes with cvxpy      | first-order fallback |
| `"SCS_GPU"`   | requires `scs` w/ CUDA | install `scs` from source against your CUDA toolkit |
| `"GUROBI"`    | extra `[gurobi]`      | commercial, fastest |

```python
clf = WRSVMClassifier(C=100, gamma=0.1, solver="GUROBI")
```

---

## 6. Verifying the install

```bash
python -c "import wrsvm; print(wrsvm.__version__)"
# 0.2.0
```

Run the bundled smoke test:

```bash
python -m pytest tests/ -v
```

Run the head-to-head solver benchmark (CLARABEL vs Gurobi):

```bash
cd scripts
python benchmark_solvers.py
```

A 30-second sanity check from the REPL:

```python
from wrsvm import WRSVMClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

clf = WRSVMClassifier(C=100, gamma=0.1, upsilon=0.2)
clf.fit(X, y)
print("train acc:", clf.score(X, y))   # expect ~0.97+
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cvxpy` | install ran in the wrong interpreter | activate the right env, reinstall |
| `SolverError: GUROBI` not installed | extras not selected | `pip install -e ".[gurobi]"` |
| `Model too large for size-limited license` | Gurobi restricted license hit | request academic license, or use CLARABEL |
| Build fails on `clarabel` (Apple Silicon, Python 3.13) | wheel not yet published | upgrade pip; if still failing, use Python 3.11 |
| `RuntimeError: CUDA out of memory` (torch backend) | kernel matrix too large | reduce N, drop to `kernel_backend="numpy"`, or subsample |
| `ImportError: DLL load failed ... torch` (Windows) | mismatched CUDA wheel | reinstall the CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

---

## 8. Uninstalling

```bash
python -m pip uninstall wrsvm
```

Editable installs leave a `.egg-info/` directory in the source tree; remove it manually if desired.
