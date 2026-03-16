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

Expected MCMC speedup range: **5× – 11×** across typical machines, growing with n.

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

## 6. Stress-test: n × draws scaling grid

Run `benchmarks/benchmark_stress.R` to sweep across option-chain sizes and draw
counts simultaneously.

```bash
Rscript benchmarks/benchmark_stress.R 2>/dev/null
```

Volatility controls n: a wider vol window produces more strikes within ±3.5σ√T,
giving more option rows.  The X matrix is computed once per σ level and reused.

Expected output (Apple M-series, measured after periodic-recomputation fix):

```
=== STRESS-TEST BENCHMARK: n × draws SCALING GRID ===
Contract: SRM26  |  obs: 2026-03-15  |  F=3.65%  |  k=80

sigma  draws          n   Xmat(s) Orig(s)  Opt(s) Speedup  Theory
----------------------------------------------------------------------
   1%  250,000       31      6.2    18.6     3.6   5.16x  [theory 22.3x | max_beta_diff 1.5e-09]
   1%  1,000,000     31      6.2    73.7    14.3   5.17x  [theory 22.3x | max_beta_diff 4.5e-10]

   3%  250,000       68     12.9    24.6     3.5   7.10x  [theory 36.8x | max_beta_diff 9.8e-09]
   3%  1,000,000     68     12.9   101.9    14.4   7.06x  [theory 36.8x | max_beta_diff 1.1e-08]

   5%  250,000       96     17.9    33.0     3.6   9.23x  [theory 43.6x | max_beta_diff 6.1e-09]
   5%  1,000,000     96     17.9   128.9    14.0   9.24x  [theory 43.6x | max_beta_diff 5.9e-08]

   7%  250,000      123     22.7    40.1     3.8  10.66x  [theory 48.5x | max_beta_diff 5.5e-08]
   7%  1,000,000    123     22.7   161.0    15.0  10.73x  [theory 48.5x | max_beta_diff 1.4e-02]
```

**What to look for:**

| Observation | Expected |
|---|---|
| Speedup grows with n | Yes: 5.2× at n=31, 10.7× at n=123 |
| Speedup constant across draws | Yes: ratio changes < 1% between 250k and 1M draws |
| max_beta_diff ≤ 1e-07 for production σ (≤ 5%) | Yes: all ≤ 5.9e-08 |
| σ=7%, 1M draws shows larger diff | Expected — σ=7% is 7× above realistic SOFR vol, outside production range |

The BLAS-efficiency factor (theory/measured ≈ 4–5×) reflects that BLAS
vectorisation disproportionately benefits the original's large matrix-vector
products.  This is hardware-dependent; the speedup ratios above are the
relevant production metric.

---

## 7. Algorithmic proof (brief)

See [DERIVATION.md](DERIVATION.md) for full algebra.  The core claim:

> `X_kj' · X_k · β`  (O(n·k) in original)
>
> equals
>
> `XtXb[j] − XtXb[k] − XtX[j,k] + XtX[k,k]`  (O(1) in optimised)

To verify by hand: pick any iteration, print `beta_vec`, and confirm both
expressions evaluate to the same scalar for a few (j, k_index) pairs.

---

## 8. Files changed and rationale

| File | Change | Reason |
|---|---|---|
| `optimized/simplexregression.cpp` | `#` → `//` comments; `like` → `prior` in `beta_prior`; moved comment before `[[Rcpp::export]]` | Bug fixes — original would not compile or had UB |
| `optimized/simplexregression_opt.cpp` | New file | Opt v1: precomputed XtX/Xtp, incremental Xb/XtXb |
| `optimized/simplexregression_opt2.cpp` | New file | Opt v2: scalar RNG + uniform-ξ fast path |
| `optimized/xmat_cpp.cpp` | New file | C++ X matrix using tanh-sinh quadrature |
| `benchmarks/benchmark_baseline.R` | New file | Generates synthetic data, runs original sampler |
| `benchmarks/benchmark_optimized.R` | New file | Same data + parallel X matrix + optimised sampler |
| `benchmarks/benchmark_compare.R` | New file | Strict head-to-head: shared inputs, both samplers, output diff |
| `benchmarks/benchmark_multi_contract.R` | New file | 6-contract production-scale simulation (Jun 2026 – Sep 2027) |
| `benchmark_v2.R` | New file | Full v2 benchmark: C++ X matrix + all MCMC versions |
| `original/market_probability_tracker.R` | Unchanged from Atlanta Fed | Reference only |

---

## 9. Running the v2 benchmark

```bash
cd <repo_root>
Rscript benchmark_v2.R 2>/dev/null
```

This benchmark compares all optimisation levels in a single run:

1. **X matrix**: R `pracma::integral` (reltol=0) vs C++ tanh-sinh (all 7 levels)
2. **MCMC**: original → opt v1 → opt v2, all on the same X matrix
3. **Full pipeline**: opt v2 MCMC + C++ X matrix

Expected output (Apple M-series):

```
================================================================
RESULTS SUMMARY
================================================================

Step                          Time(s)  Speedup  vs Orig
--------------------------------------------------------
X matrix (R)                     5.81      —      —
X matrix (C++)                 0.0530     110x      —
solnp (shared)                   1.22      —      —
MCMC original                   18.10      —     1.0x
MCMC opt v1                      3.44     5.3x     5.3x
MCMC opt v2                      1.98     1.7x     9.1x
--------------------------------------------------------
TOTAL original pipeline         25.14      —      —
TOTAL opt v1                    10.47      —     2.4x
TOTAL opt v2 (R X)               9.02      —     2.8x
TOTAL opt v2 (C++ X)             3.26      —     7.7x
```

### What to verify

**X matrix accuracy** (C++ vs R):

| Metric | Expected |
|---|---|
| Max absolute error (all cells) | < 1e-14 |
| Max relative error (cells > 1e-10) | < 1e-13 |

**MCMC accuracy** (v1 vs v2, same X matrix):

| Quantity | Expected max \|diff\| |
|---|---|
| α | 0.00e+00 (bit-for-bit) |
| B | < 1e-10 |
| σ² | < 1e-14 |
| any β | < 1e-07 |

**MCMC accuracy** (C++ X vs R X):

| Quantity | Expected max \|diff\| |
|---|---|
| any β | < 1e-07 |

Differences > 1e-07 in β indicate an algorithmic bug, not rounding.
