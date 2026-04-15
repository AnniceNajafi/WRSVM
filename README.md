# WRSVM

Open-source Python, R, and Julia implementations of the **Weighted Relaxed Support Vector Machine** (WRSVM) for robust multiclass classification under class imbalance and label noise.

## Packages

| Language | Directory | Install |
|---|---|---|
| Python | [`wrsvm_package/`](wrsvm_package/) | `pip install ./wrsvm_package` |
| R      | [`WRSVMr/`](WRSVMr/)               | `devtools::install("WRSVMr")` |
| Julia  | [`WRSVM.jl/`](WRSVM.jl/)           | `] add ./WRSVM.jl` |

See each subdirectory's `README.md` and `INSTALL.md` for detailed instructions.

## Quick start (Python)

```python
from wrsvm import WRSVMClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = WRSVMClassifier(strategy="simmsvm", C=100.0, gamma=0.1, upsilon=0.2)
clf.fit(X, y)
print(clf.predict(X[:5]))
```

## Strategies

All three packages expose four decomposition strategies through a single `strategy` argument:

- `cs` — Crammer–Singer native multiclass
- `simmsvm` — simplex-coded multiclass (sim-MCWRSVM): faster, same accuracy
- `ovo` — one-vs-one
- `ovr` — one-vs-rest

## Documentation

Full documentation is hosted at <https://annicenajafi.github.io/WRSVM/>.

## License

MIT — see [`LICENSE`](LICENSE).
