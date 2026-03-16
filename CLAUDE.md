# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An optimised re-implementation of the Federal Reserve Bank of Atlanta's [Market Probability Tracker](https://www.atlantafed.org/research-and-data/data/market-probability-tracker) MCMC sampler and X matrix build. End-to-end runtime is ~7.7× faster (MCMC sampler 9.1×, X matrix 110×). No change to statistical methodology.

The remote GitHub repo is at `https://github.com/adriangilliam/mpt-optimizer`. The local working directory is `~/Dev/fed`; the git-tracked copy synced to GitHub lives at `/tmp/mpt-optimizer`.

## Running benchmarks

```bash
# Full v2 benchmark: C++ X matrix + all MCMC versions (~30s, recommended)
Rscript benchmark_v2.R 2>/dev/null

# Single-contract MCMC-only head-to-head (~30s)
Rscript benchmarks/benchmark_compare.R 2>/dev/null

# 6-contract production-scale simulation (~5 min)
Rscript benchmarks/benchmark_multi_contract.R 2>/dev/null

# Stress-test: n × draws scaling grid (σ=1–7%, draws=250k–1M)
Rscript benchmarks/benchmark_stress.R 2>/dev/null
```

Suppress the `For infinite domains Gauss integration is applied!` noise with `2>/dev/null`.

## Key pipeline (per contract)

Every benchmark follows the same four-phase sequence:

1. **`update_data()`** → synthetic SOFR options data frame + futures + Treasury curve
2. **`get_transformparms()`** → OLS-implied F, B, σ via put-call parity regression
3. **`get_xmat()` / `get_xmat_cpp()`** → n×80 matrix X of Beta-mixture basis integrals
4. **`get_bvec_cpp` / `get_bvec_cpp_opt` / `get_bvec_cpp_opt2`** → MCMC posterior draws (250k draws, 100k burn, thin=150)

## C++ source files

| File | Purpose |
|---|---|
| `optimized/simplexregression.cpp` | Bug-fixed original sampler; exports `get_bvec_cpp` |
| `optimized/simplexregression_opt.cpp` | Opt v1: precomputed XtX/Xtp, incremental Xb/XtXb; exports `get_bvec_cpp_opt` |
| `optimized/simplexregression_opt2.cpp` | Opt v2: + scalar RNG, uniform-ξ fast path; exports `get_bvec_cpp_opt2` |
| `optimized/xmat_cpp.cpp` | C++ X matrix (tanh-sinh quadrature); exports `get_xmat_cpp` |

Also mirrored in `mpt_source/` for local development. The `optimized/` directory is the canonical location for the public repo.

All are loaded via `sourceCpp(...)` at the top of each benchmark script. After recompilation, Rcpp caches the `.so` in `~/.cache/R/sourceCpp/`; clear it if you suspect a stale build.

### v1 MCMC optimisation

Reduces `beta_met` from O(n·k²) to O(k²+n·k) per iteration by precomputing `XtX=X'X` and `Xtp=X'·price`, then maintaining `Xb=X·β` and `XtXb=XtX·β` incrementally on each accepted MCMC move.

### v2 MCMC optimisation

Replaces Rcpp sugar RNG calls (`runif(1)[0]`, `rlogis(1,...)[0]`, `rgamma(1,...)[0]`) with scalar C API calls (`R::runif`, `R::rlogis`, `R::rgamma`). Eliminates 40M vector allocations per contract. Same R RNG stream.

Adds uniform-ξ fast path for Dirichlet log-pdf: `k * lgamma(α/k)` instead of `Σ lgamma(α·ξ_j)` when ξ is uniform (1.7× additional MCMC speedup).

### C++ X matrix

Same tanh-sinh quadrature as `pracma::quadinf` (same nodes, weights, variable transformation, 7-level Richardson extrapolation). Key difference: shares `pnorm`/`dnorm` across all k=80 basis functions at each quadrature node. Machine-epsilon accuracy (max error 2×10⁻¹⁶).

## Critical option convention

The convention is counter-intuitive and easy to swap:

- `put_call='P'` = **put on futures price** = **cap on rate** → `option_flag=1`, uses `f_c` integrand
- `put_call='C'` = **call on futures price** = **floor on rate** → `option_flag=0`, uses `f_p` integrand

Put-call parity: `P_price − C_price = B·(F_rate − K_rate)`, so the OLS slope is `−B`, giving `B_ols = −coef[2] > 0`. Swapping P/C causes B to come out negative and σ ≈ 0%.

## Units

`get_transformparms` stores `F_ols` and `sigma_ols` in **percent** (e.g., 3.65, 1.00). All downstream helpers (`get_xmat`, `get_xmat_cpp`, `norm_solvenlm`, `b_gibbs`, etc.) divide by 100 to convert back to decimal. Strikes and prices are always in **decimal rate space**.

The C++ X matrix takes `m` and `sig` in decimal space (already converted): `m = F_ols/100`, `sig = 3 * sigma_ols/100`.

## MCMC hyperparameters (fixed across all benchmarks)

```r
k=80; draws=250000; burn=100000; thin=150
alpha=10; alpha_step=0.1; tau=1; zeta=1
xi=rep(1/k, k); B_m=1; B_v=1e12; sigsq=1
```

Use `set.seed(100)` immediately before each `get_bvec_cpp` / `get_bvec_cpp_opt` / `get_bvec_cpp_opt2` call to ensure reproducible, comparable draws.

## Synthetic data parameters

`update_data()` generates data for the Jun 2026 contract (obs 2026-03-13). `benchmark_multi_contract.R` extends this to 6 contracts (Jun 2026 – Sep 2027, obs 2026-03-15) using:
- Spot SOFR 3.65% (FRED, 2026-03-12)
- Forward curve: 25bp cut per ~2 FOMC meetings (5 cuts to Sep 2027)
- Normal vol term structure: 1.00% (3M) → 1.25% (18M)
- Strike grid: 25bp, range = max(3.5·σ·√T, 3%)
- Data seeds: `set.seed(42)` + contract index; MCMC seed: `set.seed(100)`

## Bugs fixed in original

Four bugs are documented in `docs/BUGS.md`. Three are in the Atlanta Fed's original `simplexregression.cpp` (fixed in `optimized/simplexregression.cpp`); one is a floating-point drift issue in the optimised sampler (fixed by periodic recomputation every 10,000 iterations). The most impactful original bug: `beta_prior()` assigned to undeclared `like` instead of `prior`, making the log-posterior diagnostic column meaningless (sampling itself was unaffected).
