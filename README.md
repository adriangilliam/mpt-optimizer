# Market Probability Tracker — MCMC Optimisation

This repository documents a **5.26× speedup** in the MCMC sampler of the Federal Reserve Bank of Atlanta's [Market Probability Tracker](https://www.atlantafed.org/research-and-data/data/market-probability-tracker) (MPT), with full verification that the statistical output is unchanged.

## Contents

```
original/
  market_probability_tracker.R   # Original R code (stub + helpers), with two bug fixes

optimized/
  simplexregression.cpp          # Bug-fixed C++ sampler (original algorithm)
  simplexregression_opt.cpp      # Optimised C++ sampler (same algorithm, less work)

benchmarks/
  benchmark_baseline.R           # Builds data, times original get_bvec_cpp
  benchmark_optimized.R          # Times get_bvec_cpp_opt (parallel X matrix too)
  benchmark_compare.R            # Head-to-head: identical inputs, swap only the sampler

docs/
  DERIVATION.md                  # Full algebraic derivation of the optimisation
  BUGS.md                        # Two bugs found and fixed in the original source
  VERIFICATION.md                # Step-by-step reproduction guide
```

## Quick results

Measured on Apple M-series (10 logical cores), synthetic SOFR options data for 2026-03-13,
June 2026 contract (31 option rows, k = 80 basis functions, 250 000 MCMC draws):

| Step | Original (s) | Optimised (s) | Speedup |
|------|-------------|---------------|---------|
| X matrix | 6.2 | 6.2 (shared) | — |
| solnp warm-start | 1.3 | 1.3 (shared) | — |
| **MCMC sampler** | **18.75** | **3.56** | **5.26×** |
| **Total** | **26.2** | **11.0** | **2.38×** |

Posterior means for α, B, σ², and all 80 β coefficients agree to floating-point
precision between the two samplers (see `benchmarks/benchmark_compare.R` output).

## Background

The MPT fits a mixture-of-Betas probability distribution to CME 3-month SOFR options,
using Bayesian MCMC (Metropolis-Hastings + Gibbs) on 80 simplex-constrained mixture weights.
The sampler is documented in Mark Fisher's
[Simplex Regression](http://www.markfisher.net/~mefisher/papers/simplex%20regression.pdf) paper.

## See also

- [DERIVATION.md](docs/DERIVATION.md) — algebra showing optimised ≡ original
- [BUGS.md](docs/BUGS.md) — two pre-existing bugs fixed
- [VERIFICATION.md](docs/VERIFICATION.md) — how to reproduce results independently
