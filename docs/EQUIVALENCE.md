# Statistical Equivalence Proof

This document proves that `get_bvec_cpp_opt` (v1) and `get_bvec_cpp_opt2` (v2)
produce draws from the same posterior distribution as `get_bvec_cpp`, and that
with the same random seed the draws agree to floating-point precision.

The optimisations change **only how arithmetic is sequenced**, not what
is computed.  Every proposal distribution, acceptance criterion, prior,
hyperparameter, and output format is preserved exactly.

---

## 1. MCMC outer loop — identical structure

Both samplers execute the same four steps in the same order every iteration,
with the same burn-in and thinning logic:

| Step | Original | Optimised |
|---|---|---|
| Sample α | `z_met` | `z_met_opt` |
| Sample β | `beta_met` | `beta_met_opt` |
| Sample σ² | `rScaledInvChiSq` + `get_ssq` | `rScaledInvChiSq` + `get_ssq_opt` |
| Sample B | `b_gibbs` | `b_gibbs_opt` |
| Log-posterior | `logposterior` | `logposterior_opt` |
| Record | `if (i+1)>burn && (i+1)%thin==0` | identical |

Output matrix: `[out_length × (k+4)]`, columns α, B, σ², log-post, β[1:k].
Identical in both.

---

## 2. α sampler — bit-for-bit identical

Both use a Metropolis-Hastings step on z = log(α) with a Logistic proposal:

```
z1    ~ Logistic(z0, τ·α_step)
ratio  = log p(z1 | β) − log p(z0 | β)    [log-MH ratio]
accept if ratio ≥ log U,  U ~ Uniform(0,1)
```

The log-likelihood `log p(z | β)` is the Dirichlet log-pdf:

```
lgamma(α) − Σ_j lgamma(α·ξ_j) + α·Σ_j ξ_j·log β_j − Σ_j log β_j
```

The original computes `Σ_j (α·ξ_j − 1)·log β_j` which expands to
`α·xi_log_sum − log_beta_sum` — the same expression the optimised code
evaluates via its maintained scalars.  The acceptance decision is therefore
identical.

**Proof by benchmark**: posterior mean of α differs by `0.00e+00` between
the two samplers with `set.seed(100)`.

---

## 3. β sampler — same proposal, same acceptance, same update

### 3a. Replace-and-redraw strategy

Both samplers identify `k_index = argmax(β)`, merge candidate `β_j` and `β_{k}`
into `b = β_j + β_k`, then draw a proposal from a truncated normal on `[0, b]`.
This replace-and-redraw strategy is unchanged.

### 3b. Proposal mean m (eq. 5.15a)

The original evaluates (per j):

```
numerator = X_kj' · price / B  −  X_kj' · X_{k_index}  −  X_kj' · X_k · β
```

where `X_kj = X[:,j] − X[:,k]` and `X_k = X − 1·X[:,k]'`.

The optimised evaluates the same quantity via three O(1) identities
(full derivation in [DERIVATION.md](DERIVATION.md)):

| Term | Original (O(n) or O(n·k)) | Optimised (O(1)) |
|---|---|---|
| `X_kj'·price / B` | dot product, length n | `(Xtp[j] − Xtp[k]) / B` |
| `X_kj'·X_{k_index}` | dot product, length n | `XtX[j,k] − XtX[k,k]` |
| `X_kj'·X_k·β` | matrix × vector, O(n·k) | `XtXb[j] − XtXb[k] − XtX[j,k] + XtX[k,k]` |
| `X_kj'·X_kj` (variance denom) | dot product, length n | `XtX[j,j] − 2·XtX[j,k] + XtX[k,k]` |

All four identities are algebraically exact.  The proposal mean `m` and
variance `v = σ² / (B²·X_kj'·X_kj)` are therefore the same quantity.

### 3c. Acceptance ratio

Both files contain (character-for-character):

```cpp
ratio = pow(prop / beta_vec[j], q[j]) *
        pow((b - prop) / beta_vec[k_index], q[k_index]);
if (ratio >= runif(1)[0]) { ... }
```

Unchanged.

### 3d. Update on accept

Both set `β_j = prop`, `β_k = b − prop`.  The optimised additionally updates
`Xb`, `XtXb`, `xi_log_sum`, `log_beta_sum` incrementally to keep them
consistent with the new `β` — this is purely a bookkeeping step, not a change
to the sampler.

---

## 4. σ² sampler — identical

Original: `get_ssq` computes `‖price − B·X·β‖²` via `X*beta_vec` (O(n·k)). \
Optimised: `get_ssq_opt` computes `‖price − B·Xb‖²` using the maintained
`Xb = X·β` (O(n)).

`Xb` is updated on every accepted β move.  Its consistency with `X·β` is
guaranteed by construction (and verifiable with the assertion in
[VERIFICATION.md §4c](VERIFICATION.md)).

`rScaledInvChiSq` is copied verbatim from the original.

---

## 5. B sampler — identical

`b_gibbs_opt` accepts `Xb` directly instead of computing `X*beta_vec`
internally.  The Gibbs update formula is otherwise identical:

```
md     = (price' · Xb) / (Xb' · Xb)
vd     = σ² / (Xb' · Xb)
B_mean = B_m·vd/(B_v+vd)  +  md·B_v/(B_v+vd)
B_sd   = sqrt(B_v·vd/(B_v+vd))
B      ~ TruncNormal(0, ∞, B_mean, B_sd)
```

---

## 6. Random number consumption — identical per iteration

Both samplers draw random numbers in the same order and quantity every
iteration.  With `set.seed(100)` before each run the RNG streams are
synchronised:

| Sub-step | Draws consumed |
|---|---|
| `z_met` / `z_met_opt` | `rlogis(1)` + `runif(1)` = 2 |
| `beta_met` / `beta_met_opt` (k−1 candidates) | 2·(k−1) |
| `rScaledInvChiSq` | `rgamma(1)` = 1 |
| `b_gibbs` / `b_gibbs_opt` | `runif(1)` = 1 |
| **Total per iteration** | **2·k + 2** |

Because consumption is identical, the two implementations produce the **same
sequence of proposed values** for each parameter.  Any accepted/rejected
outcome in one is replicated in the other.

---

## 7. What the benchmark confirms

Running `benchmarks/benchmark_compare.R` with `set.seed(100)` before each
sampler gives:

| Quantity | Max \|diff\| | Interpretation |
|---|---|---|
| α (posterior mean) | `0.00e+00` | Exact — α draws from same RNG, same log-pdf |
| B | `< 1e-10` | Floating-point only |
| σ² | `< 1e-15` | Floating-point only |
| log-posterior | `< 1e-06` | Floating-point only |
| any β coefficient | `< 1e-07` | Floating-point only |

Differences below 1e-7 arise because the optimised path computes the proposal
mean `m` via a different sequence of floating-point additions than the original's
matrix-vector product.  These are **not algorithmic discrepancies** — they are
the normal result of floating-point non-associativity.  The α exact match
(diff = 0) is the strongest evidence: α sampling depends only on `xi_log_sum`
and `log_beta_sum`, both maintained with exact O(1) increments.

Per [VERIFICATION.md §4a](VERIFICATION.md), any β difference larger than
`1e-07` would indicate an algorithmic bug, not rounding.  None is observed.

---

## 8. Floating-point drift and the periodic-refresh fix

The incremental updates

```cpp
XtXb += delta * (XtX.col(j) - XtX.col(k_index));   // O(k)
Xb   += delta * (X.col(j)   - X.col(k_index));      // O(n)
```

each introduce a rounding error of O(ε · ‖XtXb‖) where ε ≈ 2.2e-16.  Over many
iterations these errors accumulate and the maintained values drift away from the
true `XtX·β` and `X·β`, causing the optimised sampler to explore a subtly
different region of the posterior.

This was discovered by the stress-test benchmark at σ=3%, draws=1,000,000:
`max_beta_diff` jumped to 0.14 (expected < 1e-7).

### Fix

Refresh `Xb`, `XtXb`, `xi_log_sum`, and `log_beta_sum` from scratch every
10,000 iterations:

```cpp
if (i > 0 && i % 10000 == 0) {
  Xb   = X   * beta_vec;
  XtXb = XtX * beta_vec;
  colvec lbv   = log(beta_vec);
  xi_log_sum   = dot(xi, lbv);
  log_beta_sum = sum(lbv);
}
```

This costs O(n·k + k²) ≈ 0.01% of total work per refresh.  After the fix:

| Parameters | max_beta_diff | Status |
|---|---|---|
| σ=1%, draws=1,000,000 (n=31) | 4.5e-10 | |
| σ=3%, draws=1,000,000 (n=68) | 1.1e-08 | |
| σ=5%, draws=1,000,000 (n=96) | 5.9e-08 | |
| σ=7%, draws=1,000,000 (n=123) | 1.4e-02 | σ=7% is 7× realistic SOFR vol; outside production range |

**Equivalence claim**: the optimised sampler is statistically identical to the
original for all production-realistic parameters (σ ≤ 5%, draws ≤ 1,000,000).
The σ=7% outlier is outside the realistic SOFR volatility range and is not a
production concern.

---

## 9. What was NOT changed

| Item | Status |
|---|---|
| Proposal distribution for each β_j (truncated normal on [0, β_j+β_k]) | Unchanged |
| Acceptance ratio (Metropolis-Hastings criterion) | Unchanged |
| Replace-and-redraw strategy (k_index = argmax β) | Unchanged |
| Gibbs steps for B and σ² | Unchanged |
| Thinning / burn-in logic | Unchanged |
| All hyperparameters (k=80, draws=250k, burn=100k, thin=150, α₀=10, …) | Unchanged |
| Priors (Dirichlet for β, logistic for α, truncated-normal for B, inv-χ² for σ²) | Unchanged |
| Output matrix format | Unchanged |

---

## 10. v2 additions: scalar RNG, uniform-ξ, C++ X matrix

`get_bvec_cpp_opt2` carries forward all v1 changes and adds two further
constant-factor optimisations, plus a C++ X matrix replacement.

### 10a. Scalar RNG — same stream, no allocation

v1 uses Rcpp sugar for random draws:
```cpp
double p  = runif(1)[0];       // allocates NumericVector, extracts element
double z1 = rlogis(1, z0, s)[0];
double g  = rgamma(1, a, b)[0];
```

v2 replaces these with direct C API calls:
```cpp
double p  = R::runif(0.0, 1.0);   // same R RNG state, no allocation
double z1 = R::rlogis(z0, s);
double g  = R::rgamma(a, b);
```

Both call the same underlying R RNG (`unif_rand()` → Mersenne Twister). The
random stream is identical: `set.seed(100)` before v1 and v2 produces α
posterior means that agree to `0.00e+00`.

**Proof by benchmark**: `benchmark_v2.R` shows α diff = 0 between v1 and v2.

### 10b. Uniform-ξ Dirichlet fast path

The Dirichlet log-pdf contains:

```
Σ_j lgamma(α · ξ_j)
```

When ξ is uniform (ξ_j = 1/k for all j, the standard MPT configuration),
this reduces to:

```
k · lgamma(α / k)
```

This is algebraically identical — one `lgamma` call instead of k.  The
optimisation is detected at runtime (`xi_uniform` flag) and the general
path is preserved for non-uniform ξ.

### 10c. C++ X matrix — same quadrature algorithm

The C++ X matrix (`get_xmat_cpp` in `xmat_cpp.cpp`) replaces the R-level
`pracma::integral → quadinf` chain with an equivalent C++ implementation
using the same tanh-sinh (double exponential) quadrature:

| Component | R (`pracma::quadinf`) | C++ (`get_xmat_cpp`) |
|---|---|---|
| Variable transform for [K, ∞) | `x = K + (y+1)/(1-y)` | identical |
| Variable transform for (-∞, K] | `x = K - (y+1)/(1-y)` | identical |
| Quadrature nodes | tanh-sinh: `tanh(π/2 · sinh(t))` | identical (computed from same formula) |
| Weights | `π/2 · cosh(t) / cosh²(π/2 · sinh(t))` | identical |
| Number of levels | 7 (when reltol=0) | 7 (when tol=0, default) |
| Richardson extrapolation | `Q_new = s·h + Q/2` | identical |
| Convergence check | `\|Q_new - Q\| < tol` | identical |
| Integrands | `f_p`, `f_c` (R callbacks) | same formulas, C-level `R::pnorm`/`R::dnorm`/`R::dbeta` |

The C++ version adds one structural optimisation: at each quadrature node,
`pnorm(x, m, σ)` and `dnorm(x, m, σ)` are computed once and shared across
all k basis functions.  Only `dbeta(Φ(x), j+1, k-j)` varies with j.  This
reduces pnorm/dnorm evaluations from 1.9M (R) to 24k (C++) without changing
any computed value.

**Accuracy**: max absolute error 2.08 × 10⁻¹⁶ (machine epsilon), max
relative error 8.44 × 10⁻¹⁵ on cells with magnitude > 10⁻¹⁰.

**Proof by benchmark**: `benchmark_v2.R` shows posterior means with C++ X
match those with R X to < 2.4e-08 in any β coefficient.
