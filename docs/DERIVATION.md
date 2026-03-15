# Algebraic Derivation of the Optimisation

## 1. What `beta_met` computes (original)

Section 5.10–5.19 of Fisher's *Simplex Regression* paper gives a
Metropolis-Hastings sampler for the simplex-constrained weight vector β.

At each sweep the element with the largest weight, `β_k` (index `k_index`), is
temporarily merged with each candidate `β_j` (j ≠ k) to open up parameter space
in [0, β_j + β_k].  For each j the proposal mean is (eq. 5.15a):

```
m_j = β_j  +  [X_kj' · price / B  −  X_kj' · X_kindex  −  X_kj' · X_k · β]
              ─────────────────────────────────────────────────────────────────
                                    X_kj' · X_kj
```

where `X_kj = X[:,j] − X[:,k]`  (column difference),
      `X_k  = X − 1·X[:,k]'`    (every column minus column k),
      all of dimension n × 1 or n × k.

The proposal variance is `v_j = σ² / (B² · X_kj'·X_kj)`.

**Per-j cost in the original code:**
- `X_k·β` is an n-vector recomputed by `X_kj.t()*X_k*beta_vec` → **O(n·k)**
- Three further dot products of length n → O(n) each
- Total per j: **O(n·k)**; over 79 j's: **O(n·k²)** per `beta_met` call

---

## 2. Key identity used by the optimisation

Define the maintained quantities:

| Symbol | Value | Update when β_j += δ, β_k −= δ |
|--------|-------|--------------------------------|
| `Xb`   | `X · β`   (n-vector) | `Xb += δ · (X[:,j] − X[:,k])` — O(n) |
| `XtXb` | `X'X · β` (k-vector) | `XtXb += δ · (XtX[:,j] − XtX[:,k])` — O(k) |

Both are precomputed once before the MCMC loop (O(n·k) and O(k²) respectively)
and updated in O(n) and O(k) on each accepted move.

**Expanding X_k·β:**

Because sum(β) = 1 (simplex constraint):

```
X_k · β  =  (X − 1·X[:,k]') · β
          =  X·β − X[:,k]·sum(β)
          =  Xb  − X[:,k]                           (sum(β) = 1)
```

**Expanding X_kj'·X_k·β:**

```
X_kj'·X_k·β  =  (X[:,j] − X[:,k])' · (Xb − X[:,k])
              =  X[:,j]'·Xb  −  X[:,j]'·X[:,k]
               − X[:,k]'·Xb  +  X[:,k]'·X[:,k]
              =  XtXb[j]  −  XtX[j,k]
               − XtXb[k]  +  XtX[k,k]              (O(1) lookups)
```

**Expanding X_kj'·X_kindex:**

```
X_kj'·X_kindex  =  (X[:,j] − X[:,k])' · X[:,k]
                 =  XtX[j,k] − XtX[k,k]             (O(1))
```

**Expanding X_kj'·price:**

```
X_kj'·price  =  (X[:,j] − X[:,k])' · price
             =  Xtp[j] − Xtp[k]                     (O(1), Xtp = X'·price)
```

**Expanding X_kj'·X_kj:**

```
X_kj'·X_kj  =  (X[:,j]−X[:,k])' · (X[:,j]−X[:,k])
             =  XtX[j,j] − 2·XtX[j,k] + XtX[k,k]  (O(1))
```

**Full numerator, simplified:**

```
Xtp_kj / B  −  (XtX[j,k]−XtX[k,k])  −  (XtXb[j]−XtX[j,k]−XtXb[k]+XtX[k,k])
  =  (Xtp[j]−Xtp[k]) / B  −  XtXb[j] + XtXb[k]        ← O(1) per j
```

This is the numerator that appears in `beta_met_opt`.

---

## 3. Per-call complexity comparison

| | Original | Optimised |
|---|---|---|
| One-time setup | — | O(n·k) for Xb, O(k²) for XtX, XtXb |
| Per `beta_met` call (79 j's) | O(n·k²) | O(k²) for XtXb updates + O(n·k) for Xb updates |
| Per `get_ssq` / `b_gibbs` / `logposterior` | O(n·k) each × 3 | O(n) each × 3 (use maintained Xb) |
| **Total per MCMC iteration** | **O(n·k²)** | **O(k² + n·k)** |

For the production values n ≈ 31–100, k = 80:

```
Speedup  ≈  n·k² / (k² + n·k)  =  n·k / (k + n)
```

At n = 31: speedup ≈ 31·80 / (80+31) ≈ **22×** theoretical.
Measured: **5.26×** on real hardware (BLAS already vectorises the original too).

---

## 4. Maintained quantities for the Dirichlet log-pdf

`z_met` and `logposterior` both evaluate the Dirichlet log-pdf:

```
log p(β | α, ξ)  =  lgamma(α)  −  Σ_j lgamma(α·ξ_j)
                  +  α · Σ_j ξ_j·log β_j  −  Σ_j log β_j
                  =  lgamma(α)  −  lgam_sum(α)
                  +  α·xi_log_sum  −  log_beta_sum
```

`lgam_sum(α)` is O(k) and recomputed whenever α changes (unavoidable).
`xi_log_sum` and `log_beta_sum` are maintained in O(1) per accepted move:

```
xi_log_sum   += ξ_j·(log β_j_new − log β_j_old) + ξ_k·(log β_k_new − log β_k_old)
log_beta_sum +=     (log β_j_new − log β_j_old) +     (log β_k_new − log β_k_old)
```

**Important:** in `z_met` the acceptance ratio is `log p(z1|β) − log p(z0|β)`.
`log_beta_sum` is constant with respect to α, so it cancels in the ratio — meaning
the *sampling* is correct even if `log_beta_sum` were wrong.  However, having it
correct ensures the `logposterior` diagnostic column is also accurate.

---

## 5. What is NOT changed

- The proposal distribution for each β_j (truncated normal on [0, β_j+β_k])
- The acceptance ratio (MH criterion)
- The replace-and-redraw strategy (k_index = argmax β, β_k = b − prop)
- The Gibbs step for B and σ²
- The thinning / burn-in logic
- All hyperparameters (k=80, draws=250k, burn=100k, thin=150, α₀=10, …)
