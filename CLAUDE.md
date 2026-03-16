# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An optimised re-implementation of the Federal Reserve Bank of Atlanta's [Market Probability Tracker](https://www.atlantafed.org/research-and-data/data/market-probability-tracker) MCMC sampler. The MCMC sampler itself runs 5–6× faster; end-to-end total runtime is ~2.3× faster (the X matrix build is unchanged and accounts for roughly half the total runtime). No change to statistical methodology.

The remote GitHub repo is at `https://github.com/adriangilliam/mpt-optimizer`. The local working directory is `~/Dev/fed`; the git-tracked copy synced to GitHub lives at `/tmp/mpt-optimizer`.

## Running benchmarks

```bash
# Single-contract head-to-head (fastest, ~30s, validates correctness)
Rscript benchmark_compare.R 2>/dev/null

# 6-contract production-scale simulation (~5 min)
Rscript benchmark_multi_contract.R 2>/dev/null

# Baseline only (original sampler, ~25s)
Rscript benchmark.R 2>/dev/null
```

Suppress the `For infinite domains Gauss integration is applied!` noise with `2>/dev/null`.

## Key pipeline (per contract)

Every benchmark follows the same four-phase sequence:

1. **`update_data()`** → synthetic SOFR options data frame + futures + Treasury curve
2. **`get_transformparms()`** → OLS-implied F, B, σ via put-call parity regression
3. **`get_xmat()`** → n×80 matrix X of Beta-mixture basis integrals (slowest step, O(n·k) integrals)
4. **`get_bvec_cpp` / `get_bvec_cpp_opt`** → MCMC posterior draws (250k draws, 100k burn, thin=150)

## C++ source files

| File | Purpose |
|---|---|
| `mpt_source/simplexregression.cpp` | Bug-fixed original sampler; exports `get_bvec_cpp` |
| `mpt_source/simplexregression_opt.cpp` | Optimised sampler; exports `get_bvec_cpp_opt` |

Both are loaded via `sourceCpp(...)` at the top of each benchmark script. Paths are currently hardcoded to `~/Dev/fed/mpt_source/`. After recompilation, Rcpp caches the `.so` in `~/.cache/R/sourceCpp/`; clear it if you suspect a stale build.

The optimised sampler reduces `beta_met` from O(n·k²) to O(k²+n·k) per iteration by precomputing `XtX=X'X` and `Xtp=X'·price`, then maintaining `Xb=X·β` and `XtXb=XtX·β` incrementally on each accepted MCMC move. No other part of the algorithm changes.

## Critical option convention

The convention is counter-intuitive and easy to swap:

- `put_call='P'` = **put on futures price** = **cap on rate** → `option_flag=1`, uses `f_c` integrand
- `put_call='C'` = **call on futures price** = **floor on rate** → `option_flag=0`, uses `f_p` integrand

Put-call parity: `P_price − C_price = B·(F_rate − K_rate)`, so the OLS slope is `−B`, giving `B_ols = −coef[2] > 0`. Swapping P/C causes B to come out negative and σ ≈ 0%.

## Units

`get_transformparms` stores `F_ols` and `sigma_ols` in **percent** (e.g., 3.65, 1.00). All downstream helpers (`get_xmat`, `norm_solvenlm`, `b_gibbs`, etc.) divide by 100 to convert back to decimal. Strikes and prices are always in **decimal rate space**.

## MCMC hyperparameters (fixed across all benchmarks)

```r
k=80; draws=250000; burn=100000; thin=150
alpha=10; alpha_step=0.1; tau=1; zeta=1
xi=rep(1/k, k); B_m=1; B_v=1e12; sigsq=1
```

Use `set.seed(100)` immediately before each `get_bvec_cpp` / `get_bvec_cpp_opt` call to ensure reproducible, comparable draws.

## Synthetic data parameters

`update_data()` generates data for the Jun 2026 contract (obs 2026-03-13). `benchmark_multi_contract.R` extends this to 6 contracts (Jun 2026 – Sep 2027, obs 2026-03-15) using:
- Spot SOFR 3.65% (FRED, 2026-03-12)
- Forward curve: 25bp cut per ~2 FOMC meetings (5 cuts to Sep 2027)
- Normal vol term structure: 1.00% (3M) → 1.25% (18M)
- Strike grid: 25bp, range = max(3.5·σ·√T, 3%)
- Data seeds: `set.seed(42)` + contract index; MCMC seed: `set.seed(100)`

## Bugs fixed in original

Three bugs in the Atlanta Fed's `simplexregression.cpp` are fixed in `mpt_source/simplexregression.cpp`. See `docs/BUGS.md` in the GitHub repo for details. The most impactful: `beta_prior()` assigned to undeclared `like` instead of `prior`, making the log-posterior diagnostic column meaningless (sampling itself was unaffected).
