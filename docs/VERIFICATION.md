# Independent Verification Guide

This document explains exactly how to reproduce the benchmark numbers from
scratch, what to check, and what constitutes a valid replication.

---

## 1. Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| R | ≥ 4.4 | Tested on 4.5.3 |
| C++ compiler | clang ≥ 17 or g++ ≥ 12 | Must support C++17 |
| R packages | see below | Install once |

Install R packages:
```r
install.packages(c(
  "pracma",         # integral()
  "Rsolnp",         # solnp() — nonlinear optimiser for warm-start
  "reshape",        # cast() — wide/long pivot
  "Rcpp",           # sourceCpp()
  "RcppArmadillo",  # arma:: — linear algebra
  "parallel"        # makeCluster / parLapply — for parallel X matrix
), repos = "https://cran.rstudio.com/")
```

---

## 2. Data

The benchmarks use **synthetic** SOFR options data generated inside R — no
external data source is required.

The synthetic dataset models the CME June 2026 3M SOFR futures contract as of
2026-03-13:

| Parameter | Value | Source |
|---|---|---|
| Observation date | 2026-03-13 | — |
| Contract expiry | 2026-06-17 | CME IMM date |
| Forward rate F | 3.60% | ~SOFR on 2026-03-12 |
| Normal vol σ | 1.00% | Representative |
| Discount factor B | exp(−3.72% × 96/365) | Treasury.gov 3M yield |
| Strike increment | 25 bp | Representative CME grid |
| Pricing model | Bachelier (normal) | Same model used by MPT |
| Treasury yields | 1M–30Y curve | Treasury.gov, 2026-03-13 |

The resulting dataset has 31 option rows (puts + calls passing the minimum
price filter) across 16 strike prices, representative of a single FOMC window.

The random seed for price noise is `set.seed(42)`.  The MCMC seed is
`set.seed(100)`.

---

## 3. Running the head-to-head comparison

```bash
cd <repo_root>
Rscript benchmarks/benchmark_compare.R 2>/dev/null
```

Expected output structure:
```
=== SHARED SETUP ===
Options: 31 rows  F=3.60xx%  B=0.9901xx  sigma=0.9988%
X matrix (31x80): ~6s
solnp: ~1s

--- ORIGINAL get_bvec_cpp ---
MCMC time (original): ~19s

--- OPTIMISED get_bvec_cpp_opt ---
MCMC time (optimised): ~3-4s

=== RESULTS SUMMARY ===
...  MCMC sampler ...  ~5x
...  TOTAL        ...  ~2.4x

=== POSTERIOR MEAN COMPARISON ===
  alpha      orig=X.XXX   opt=X.XXX   diff=0.00e+00
  B          orig=X.XXX   opt=X.XXX   diff<1e-10
  sigma^2    orig=X.XXX   opt=X.XXX   diff<1e-15
  log-post   orig=X.XXX   opt=X.XXX   diff<1e-06
  beta[1:k]  max |diff| < 1e-07,  mean |diff| < 1e-08
```

---

## 4. What to verify

### 4a. Statistical equivalence

The two samplers use `set.seed(100)` before each run, so they consume the same
sequence of random numbers and produce **identical** draws (not just draws from
the same distribution).

A valid replication shows:

| Quantity | Expected max absolute difference |
|---|---|
| alpha (posterior mean) | 0.00e+00 (bit-for-bit) |
| B | < 1e-10 |
| sigma^2 | < 1e-14 |
| log-posterior | < 1e-06 |
| any beta coefficient | < 1e-07 |

Larger differences indicate an algorithmic discrepancy, not mere floating-point
rounding.  If you see them, check whether `XtXb` is being updated correctly
after each accepted move.

### 4b. Speedup

The speedup ratio depends on:
- **n** (number of options) — more options = larger benefit
- **k** (number of basis functions, always 80) — fixed
- Hardware SIMD/BLAS capabilities — already help the original

Expected MCMC speedup range: **3× – 8×** across typical machines.

If the speedup is < 2×, confirm that `sourceCpp` compiled the optimised file
(not cached from a previous run).  Delete `~/.cache/R/sourceCpp/` or the
equivalent and re-run.

### 4c. Correctness of the maintained quantities

To verify that `Xb` and `XtXb` remain consistent throughout, add the following
assertion inside the MCMC loop in `simplexregression_opt.cpp` (debug build only):

```cpp
// After beta_met_opt returns:
assert(norm(Xb   - X * beta_vec,   2) < 1e-10);
assert(norm(XtXb - XtX * beta_vec, 2) < 1e-10);
```

These should never trigger.  If they do, the incremental update logic has a bug.

---

## 5. Scaling: six-contract production simulation

Run `benchmarks/benchmark_multi_contract.R` for a full production-scale test.
It uses observation date 2026-03-15 (spot SOFR = 3.65%, FRED 2026-03-12),
six SOFR futures contracts (Jun 2026 – Sep 2027), a 25 bp strike grid, and
draws a forward curve consistent with 5–6 Fed cuts through Sep 2027.

```bash
Rscript benchmarks/benchmark_multi_contract.R 2>/dev/null
```

Expected output (Apple M-series, measured):

```
Contract FOMC-covered     n  T(y)    F(%) Xmat(s) solnp(s) Orig(s)  Opt(s)
------------------------------------------------------------------------
SRM26    Mar+Apr         41 0.258    3.65     7.9      1.1    19.8     3.5  ( 5.69x)
SRU26    Jun+Jul         44 0.507    3.40     8.8      1.2    20.7     3.7  ( 5.62x)
SRZ26    Sep+Oct         49 0.756    3.15     9.2      1.0    21.1     3.7  ( 5.76x)
SRH27    Dec+Q1-27       54 1.005    2.90    10.3      0.6    23.5     3.8  ( 6.24x)
SRM27    Q1-27           55 1.255    2.65    10.5      0.6    23.9     3.8  ( 6.36x)
SRU27    Q2-27           57 1.504    2.40    10.8      0.6    24.0     3.8  ( 6.36x)

Step                     Original  Optimised   Speedup
X matrix (all)               57.5s       57.5s  (shared)
solnp (all)                   5.1s        5.1s  (shared)
MCMC (all)                  133.0s       22.1s     6.01x
TOTAL                       195.6s       84.7s     2.31x

Total production run: orig=3.3min  opt=1.4min
```

### Observed speedup scaling by contract size

| Contract | n (rows) | Theory n·k/(k+n) | Measured |
|---|---|---|---|
| SRM26 | 41 | 27× | 5.69× |
| SRU26 | 44 | 28× | 5.62× |
| SRZ26 | 49 | 30× | 5.76× |
| SRH27 | 54 | 32× | 6.24× |
| SRM27 | 55 | 33× | 6.36× |
| SRU27 | 57 | 33× | 6.36× |

The MCMC speedup grows with n (larger options chains = bigger win), because the
original's O(n·k²) bottleneck scales linearly in n while the optimised O(k²+n·k)
scales much more slowly.  The gap between theoretical and measured speedup is
explained by BLAS vectorisation, which disproportionately benefits the original's
large matrix-vector products.

---

## 6. Algorithmic proof (brief)

See [DERIVATION.md](DERIVATION.md) for full algebra.  The core claim:

> `X_kj' · X_k · β`  (O(n·k) in original)
>
> equals
>
> `XtXb[j] − XtXb[k] − XtX[j,k] + XtX[k,k]`  (O(1) in optimised)

To verify by hand: pick any iteration, print `beta_vec`, and confirm both
expressions evaluate to the same scalar for a few (j, k_index) pairs.

---

## 7. Files changed and rationale

| File | Change | Reason |
|---|---|---|
| `optimized/simplexregression.cpp` | `#` → `//` comments; `like` → `prior` in `beta_prior`; moved comment before `[[Rcpp::export]]` | Bug fixes — original would not compile or had UB |
| `optimized/simplexregression_opt.cpp` | New file | Optimised sampler; identical interface to `get_bvec_cpp` |
| `benchmarks/benchmark_baseline.R` | New file | Generates synthetic data, runs original sampler |
| `benchmarks/benchmark_optimized.R` | New file | Same data + parallel X matrix + optimised sampler |
| `benchmarks/benchmark_compare.R` | New file | Strict head-to-head: shared inputs, both samplers, output diff |
| `benchmarks/benchmark_multi_contract.R` | New file | 6-contract production-scale simulation (Jun 2026 – Sep 2027) |
| `original/market_probability_tracker.R` | Unchanged from Atlanta Fed | Reference only |
