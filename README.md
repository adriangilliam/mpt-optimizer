# Market Probability Tracker — MCMC Optimisation

This repository documents a **~2.3× end-to-end speedup** in the Federal Reserve Bank of Atlanta's [Market Probability Tracker](https://www.atlantafed.org/research-and-data/data/market-probability-tracker) (MPT), driven by a **5–6× speedup in the MCMC sampler** alone, with full verification that the statistical output is unchanged.

## Contents

```
original/
  market_probability_tracker.R    # Original R code from Atlanta Fed (reference only)

optimized/
  simplexregression.cpp           # Bug-fixed C++ sampler (original algorithm)
  simplexregression_opt.cpp       # Optimised C++ sampler (same algorithm, less work)

benchmarks/
  benchmark_baseline.R            # Builds synthetic data, times original get_bvec_cpp
  benchmark_optimized.R           # Same data + parallel X matrix + optimised sampler
  benchmark_compare.R             # Head-to-head: identical inputs, swap only the sampler
  benchmark_multi_contract.R      # 6-contract production-scale simulation (Jun 2026–Sep 2027)

docs/
  DERIVATION.md                   # Full algebraic derivation of the optimisation
  BUGS.md                         # Three bugs found and fixed in the original source
  VERIFICATION.md                 # Step-by-step reproduction guide with expected output
```

---

## Quick replication

Install R packages once:

```r
install.packages(c("pracma", "Rsolnp", "reshape", "Rcpp", "RcppArmadillo", "parallel"),
                 repos = "https://cran.rstudio.com/")
```

### 1. Head-to-head single-contract comparison (fastest, ~30s)

```bash
Rscript benchmarks/benchmark_compare.R 2>/dev/null
```

Runs both samplers on identical synthetic SOFR options data (Jun 2026 contract,
obs date 2026-03-13, n=31 rows, k=80 basis functions, 250 000 MCMC draws).
Prints a timing table and validates that posterior means agree to floating-point
precision.

### 2. Production-scale 6-contract simulation (~5 min)

```bash
Rscript benchmarks/benchmark_multi_contract.R 2>/dev/null
```

Simulates the full production run: 6 SOFR futures contracts from Jun 2026 to
Sep 2027, observation date 2026-03-15, spot SOFR 3.65% (FRED 2026-03-12), 25 bp
strike grid, ~40–57 option rows per contract, 250 000 MCMC draws each.

---

## Results — single contract (benchmark_compare.R)

Measured on Apple M-series, obs 2026-03-13, Jun 2026 contract, n=31, k=80:

| Step | Original (s) | Optimised (s) | Speedup |
|---|---|---|---|
| X matrix | 6.2 | 6.2 (shared) | — |
| solnp warm-start | 1.3 | 1.3 (shared) | — |
| **MCMC sampler** | **18.75** | **3.56** | **5.26×** |
| **Total** | **26.2** | **11.0** | **2.38×** |

Posterior means for α, B, σ², and all 80 β coefficients agree to floating-point
precision between the two samplers.

---

## Results — 6-contract production scale (benchmark_multi_contract.R)

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

The MCMC speedup grows with n (larger options chains = bigger gain), because the
original's O(n·k²) bottleneck scales linearly in n while the optimised O(k²+n·k)
scales much more slowly.

---

## Background

The MPT fits a mixture-of-Betas probability distribution to CME 3-month SOFR
options using Bayesian MCMC (Metropolis-Hastings + Gibbs) on 80 simplex-constrained
mixture weights.  The sampler is documented in Mark Fisher's
[Simplex Regression](http://www.markfisher.net/~mefisher/papers/simplex%20regression.pdf)
paper.

The key optimisation reduces `beta_met` from **O(n·k²)** to **O(k²+n·k)** per
call by precomputing `XtX = X'X` and maintaining `Xb = X·β` and `XtXb = X'X·β`
incrementally (O(n) and O(k) updates per accepted MCMC move, vs full recomputes).
The statistical output is bit-for-bit identical for α, and agrees to < 1e-7 for
all β coefficients (differences are floating-point rounding, not algorithmic).

The end-to-end speedup (~2.3×) is more modest than the MCMC-only speedup (5–6×)
because the X matrix build — an O(n·k) numerical integration step that is
unchanged — accounts for roughly half the total runtime.

---

## See also

- [EQUIVALENCE.md](docs/EQUIVALENCE.md) — proof that no methodology was changed (proposal distributions, acceptance criteria, priors, RNG consumption all identical)
- [DERIVATION.md](docs/DERIVATION.md) — algebra showing optimised ≡ original
- [BUGS.md](docs/BUGS.md) — three pre-existing bugs fixed
- [VERIFICATION.md](docs/VERIFICATION.md) — full reproduction guide with expected output ranges
