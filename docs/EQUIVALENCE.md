# Statistical Equivalence Proof

This document proves that `get_bvec_cpp_opt` produces draws from the same
posterior distribution as `get_bvec_cpp`, and that with the same random seed
the draws agree to floating-point precision.

The optimisation changes **only how arithmetic is sequenced**, not what
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

## 8. What was NOT changed

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
