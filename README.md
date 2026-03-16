# Market Probability Tracker — Performance Optimisation

This repository documents a **~7.7× end-to-end speedup** in the Federal Reserve Bank of Atlanta's [Market Probability Tracker](https://www.atlantafed.org/research-and-data/data/market-probability-tracker) (MPT), with full verification that the statistical output is unchanged.

The optimisation comes from three independent improvements:

| Optimisation | Target | Speedup | Methodology change |
|---|---|---|---|
| Precomputed X'X, incremental X·β (v1) | MCMC sampler | 5–6× | None |
| Scalar RNG + uniform-ξ fast path (v2) | MCMC sampler | 1.7× additional | None |
| C++ tanh-sinh quadrature (v2) | X matrix build | 110× | None |

## Contents

```
original/
  market_probability_tracker.R      # Original R code from Atlanta Fed (reference only)

optimized/
  simplexregression.cpp             # Bug-fixed C++ sampler (original algorithm)
  simplexregression_opt.cpp         # Opt v1: precomputed XtX/Xtp, incremental Xb/XtXb
  simplexregression_opt2.cpp        # Opt v2: + scalar RNG, uniform-xi fast path
  xmat_cpp.cpp                      # C++ X matrix (replaces pracma::integral)

benchmarks/
  benchmark_baseline.R              # Builds synthetic data, times original get_bvec_cpp
  benchmark_optimized.R             # Same data + parallel X matrix + optimised sampler
  benchmark_compare.R               # Head-to-head: identical inputs, swap only the sampler
  benchmark_multi_contract.R        # 6-contract production-scale simulation (Jun 2026–Sep 2027)

benchmark_v2.R                      # Full v2 benchmark: C++ X matrix + all MCMC versions

docs/
  DERIVATION.md                     # Full algebraic derivation of the v1 MCMC optimisation
  BUGS.md                           # Three bugs found and fixed in the original source
  VERIFICATION.md                   # Step-by-step reproduction guide with expected output
  EQUIVALENCE.md                    # Proof that no methodology was changed
```

---

## Quick replication

Install R packages once:

```r
install.packages(c("pracma", "Rsolnp", "reshape", "Rcpp", "RcppArmadillo", "parallel"),
                 repos = "https://cran.rstudio.com/")
```

All scripts should be run from the repository root.

### 1. Full v2 benchmark (~30s, recommended)

```bash
Rscript benchmark_v2.R 2>/dev/null
```

Benchmarks all three optimisation levels (original, v1, v2) plus the C++ X matrix against the R version. Validates accuracy at every level.

### 2. Head-to-head MCMC-only comparison (~30s)

```bash
Rscript benchmark_compare.R 2>/dev/null
```

Runs both v0 and v1 MCMC samplers on identical synthetic SOFR options data (Jun 2026 contract, obs date 2026-03-13, n=31 rows, k=80 basis functions, 250 000 MCMC draws). Validates that posterior means agree to floating-point precision.

### 3. Production-scale 6-contract simulation (~5 min)

```bash
Rscript benchmark_multi_contract.R 2>/dev/null
```

Simulates the full production run: 6 SOFR futures contracts from Jun 2026 to Sep 2027.

---

## Results — full v2 pipeline (benchmark_v2.R)

Measured on Apple M-series, obs 2026-03-13, Jun 2026 contract, n=31, k=80:

| Step | Original (s) | Opt v2 (s) | Speedup |
|---|---|---|---|
| X matrix (R → C++) | 5.81 | 0.05 | 110× |
| solnp warm-start | 1.22 | 1.22 | (shared) |
| MCMC sampler | 18.10 | 1.98 | 9.1× |
| **Total** | **25.14** | **3.26** | **7.7×** |

### X matrix accuracy (C++ vs R)

The C++ X matrix uses the same tanh-sinh (double exponential) quadrature algorithm as `pracma::quadinf`, with the same 7-level Richardson extrapolation and variable transformation for semi-infinite domains. It differs only in implementation language and in sharing `pnorm`/`dnorm` evaluations across the 80 basis functions.

| Metric | Value |
|---|---|
| Max absolute error | 2.08 × 10⁻¹⁶ (machine epsilon) |
| Max relative error (cells > 10⁻¹⁰) | 8.44 × 10⁻¹⁵ |

### MCMC accuracy (all versions use same `set.seed(100)`)

| Quantity | v1 vs v2 max \|diff\| | R X vs C++ X max \|diff\| |
|---|---|---|
| α (posterior mean) | 0.00e+00 | 0.00e+00 |
| B | 1.72e-12 | 1.10e-12 |
| σ² | 6.67e-19 | 5.32e-19 |
| log-posterior | 6.24e-08 | 1.72e-08 |
| any β coefficient | 1.56e-08 | 2.36e-08 |

Differences are MCMC stochasticity from floating-point non-associativity, not algorithmic.

---

## Results — v1 MCMC only, 6-contract production scale

Measured on Apple M-series, obs 2026-03-15, spot SOFR 3.65%:

| Contract | FOMC window | n | T (y) | F (%) | Orig (s) | Opt (s) | Speedup |
|---|---|---|---|---|---|---|---|
| SRM26 | Mar + Apr | 41 | 0.258 | 3.65 | 19.8 | 3.5 | 5.69× |
| SRU26 | Jun + Jul | 44 | 0.507 | 3.40 | 20.7 | 3.7 | 5.62× |
| SRZ26 | Sep + Oct | 49 | 0.756 | 3.15 | 21.1 | 3.7 | 5.76× |
| SRH27 | Dec + Q1-27 | 54 | 1.005 | 2.90 | 23.5 | 3.8 | 6.24× |
| SRM27 | Q1-27 | 55 | 1.255 | 2.65 | 23.9 | 3.8 | 6.36× |
| SRU27 | Q2-27 | 57 | 1.504 | 2.40 | 24.0 | 3.8 | 6.36× |
| **Total** | | | | | **133.0 s** | **22.1 s** | **6.01×** |
| **Full run** | | | | | **3.3 min** | **1.4 min** | **2.31×** |

---

## What was optimised

### v1: MCMC sampler — O(n·k²) → O(k²+n·k)

The key optimisation reduces `beta_met` from **O(n·k²)** to **O(k²+n·k)** per call by precomputing `XtX = X'X` and `Xtp = X'·price` once, then maintaining `Xb = X·β` and `XtXb = X'X·β` incrementally (O(n) and O(k) updates per accepted MCMC move, vs full O(n·k) recomputes). See [DERIVATION.md](docs/DERIVATION.md).

### v2: Scalar RNG — eliminate 40M vector allocations

Rcpp sugar's `runif(1)[0]` allocates an R `NumericVector`, fills it, and extracts element 0 on every call. With ~160 RNG draws per MCMC iteration × 250k iterations = 40M calls, the allocation overhead is substantial. Replacing with `R::runif(0.0, 1.0)` calls the same underlying R RNG directly at the C level, producing the same random stream with no allocation.

### v2: Uniform-ξ Dirichlet fast path

When ξ is uniform (all elements = 1/k, the standard MPT configuration), the Dirichlet log-pdf's `Σ_j lgamma(α·ξ_j)` simplifies to `k · lgamma(α/k)` — one `lgamma` call instead of 80. This is evaluated ~500k times per contract (twice per `z_met` call × 250k iterations).

### v2: C++ X matrix — same quadrature, no R dispatch

The X matrix build calls `pracma::integral` (which delegates to `quadinf` for semi-infinite domains) once per (strike, basis function) pair — 31 × 80 = 2,480 calls. Each call evaluates the integrand ~773 times through R function dispatch, for 1.9M R-level function evaluations total.

The C++ implementation uses the same tanh-sinh quadrature with the same nodes, weights, variable transformation, and 7-level Richardson extrapolation. It eliminates two sources of overhead:

1. **R function dispatch**: all `pnorm`, `dnorm`, `dbeta` calls go directly through R's C API
2. **Shared computation**: at each quadrature node, `pnorm(x, m, σ)` and `dnorm(x, m, σ)` are computed once and shared across all 80 basis functions (only `dbeta` varies with j)

This reduces 1.9M R-dispatched evaluations to 24k C-level `pnorm`/`dnorm` calls + 1.9M C-level `dbeta` calls, yielding 110× speedup with machine-epsilon accuracy.

---

## At production scale

| Scenario | Original | Optimised (v2) |
|---|---|---|
| 100 contracts × 1 day | ~42 min | ~5 min |
| 100 contracts × 252 days × 5 years (historical) | ~37 days | ~4.8 days |

---

## See also

- [EQUIVALENCE.md](docs/EQUIVALENCE.md) — proof that no methodology was changed
- [DERIVATION.md](docs/DERIVATION.md) — algebra showing v1 optimised ≡ original
- [BUGS.md](docs/BUGS.md) — three pre-existing bugs fixed
- [VERIFICATION.md](docs/VERIFICATION.md) — full reproduction guide with expected output
