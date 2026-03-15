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

## 5. Scaling: why the author's production run takes ~3 minutes

The benchmark above runs a **single contract**.  The production MPT processes
**multiple contracts** (one per upcoming FOMC meeting window, typically 3–7)
with real CME options chains that have more strikes (25 bp grid → ~50–100
option rows per contract).

Approximate scaling:

```
T_total ≈ N_contracts × (T_xmat(n) + T_solnp + T_mcmc(n, k))
```

At 5 contracts × ~50 options each with k=80:
- Serial X matrix: ~5 × 18s ≈ 90s
- MCMC (original): ~5 × 60s ≈ 300s  → ~390s total ≈ 6.5 min
- MCMC (optimised): ~5 × 12s ≈ 60s  → ~150s total ≈ 2.5 min

The speedup holds at scale because the O(n·k²) bottleneck is per-call and
n grows linearly with the options chain width.

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
| `original/market_probability_tracker.R` | Unchanged from Atlanta Fed | Reference only |
