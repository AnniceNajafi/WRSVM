# Installing `WRSVM.jl` (Julia)

Detailed installation guide for the Julia implementation of the Weighted Relaxed Support Vector Machine. Pure Julia — no Python required.

---

## 1. System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Julia       | 1.9     | 1.11+ (matches `Manifest.toml`) |
| OS          | Linux, macOS, Windows 10/11 | — |
| RAM         | 2 GB    | 8 GB (large kernel matrices) |

Verify:

```bash
julia --version       # >= 1.9
```

If you don't have Julia, get it from <https://julialang.org/downloads/> or install with `juliaup` (recommended on all platforms):

```bash
# Linux / macOS
curl -fsSL https://install.julialang.org | sh

# Windows (PowerShell)
winget install julia -s msstore
```

---

## 2. Dependencies

Resolved automatically from `Project.toml` / `Manifest.toml`:

| Package        | Version pin (compat) | Purpose |
|----------------|----------------------|---------|
| `JuMP`         | 1.30                 | optimization modeling DSL |
| `Clarabel`     | 0.11.1               | interior-point conic solver |
| `JSON`         | 1.5                  | result serialization |
| `LinearAlgebra`, `Random`, `Statistics`, `Test` | stdlib | — |

Stdlib packages (`LinearAlgebra`, etc.) ship with Julia and need no install.

---

## 3. Install (three options)

### 3a. Develop from a local clone (current state of the repo)

```julia
using Pkg
Pkg.develop(path = "C:/Users/annic/Documents/WRSVM/WRSVM.jl")
Pkg.instantiate()
```

`develop` symlinks the source so edits are picked up immediately. Use this while iterating on the package.

### 3b. Activate the package's own environment

```julia
using Pkg
Pkg.activate("C:/Users/annic/Documents/WRSVM/WRSVM.jl")
Pkg.instantiate()           # resolve from Manifest.toml (reproducible)
```

This is the most reproducible — it pins exactly the versions in `Manifest.toml`.

### 3c. Add as a dependency in your own project

```julia
using Pkg
Pkg.activate(".")           # your project
Pkg.add(url = "https://github.com/annicenajafi/WRSVM.jl")   # when published
# or, locally:
Pkg.add(path = "C:/Users/annic/Documents/WRSVM/WRSVM.jl")
```

---

## 4. Verifying the install

From inside the activated environment:

```julia
using WRSVM
using Statistics

X = randn(100, 4)
y = rand(1:3, 100)

model = solve_crammer_singer(X, y; C = 100.0, gamma = 0.1, upsilon = 0.2)
preds = predict_cs(model, X)
println("train acc = ", mean(preds .== y))
```

Run the test suite:

```julia
using Pkg
Pkg.activate("C:/Users/annic/Documents/WRSVM/WRSVM.jl")
Pkg.test()
```

Or from the shell:

```bash
cd C:/Users/annic/Documents/WRSVM/WRSVM.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

---

## 5. Optional: alternative solvers

`Clarabel.jl` is the bundled default. To swap in a different JuMP-compatible QP solver, install it and pass it through (modify the relevant `solve_*.jl` if a kwarg isn't already exposed):

```julia
using Pkg
Pkg.add("OSQP")          # open-source ADMM
Pkg.add("Gurobi")        # commercial, requires GUROBI_HOME env var pointing to install
Pkg.add("MosekTools")    # commercial / academic
```

Gurobi requires a license file and `GUROBI_HOME` set before `Pkg.build("Gurobi")`.

---

## 6. Performance tips

- **Precompile time** — first `using WRSVM` triggers JIT compilation of JuMP + Clarabel, ~10–30 s. Subsequent loads in the same session are instant.
- **Threads** — start Julia with `julia -t auto` to use all cores for kernel matrix assembly:
  ```bash
  julia -t auto --project=.
  ```
- **System image** — for repeated short runs, build a sysimage with `PackageCompiler.jl` to skip JIT entirely:
  ```julia
  using PackageCompiler
  create_sysimage([:WRSVM, :Clarabel, :JuMP]; sysimage_path = "wrsvm_sys.so")
  # then: julia --sysimage wrsvm_sys.so
  ```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: ArgumentError: Package WRSVM not found` | environment not activated | `Pkg.activate("path/to/WRSVM.jl")` |
| `Pkg.instantiate()` resolves to wrong versions | global env interference | always work inside an activated project; do not install into the default env |
| Clarabel build fails on Apple Silicon | old Julia | upgrade to Julia 1.10+ (native arm64) |
| `LoadError: UndefVarError: solve_crammer_singer` | stale precompile cache after edits | `Pkg.precompile()` or delete `~/.julia/compiled/v1.x/WRSVM` |
| Slow first call | normal JIT warmup | use a sysimage (see §6) |
| `MethodError: no method matching solve_*` with newer JuMP | breaking JuMP release | pin via Manifest, or update WRSVM.jl |

---

## 8. Uninstalling

```julia
using Pkg
Pkg.rm("WRSVM")
```

For a `develop` install, also remove the dev entry:

```julia
Pkg.free("WRSVM")     # detach the local path
Pkg.rm("WRSVM")
```
