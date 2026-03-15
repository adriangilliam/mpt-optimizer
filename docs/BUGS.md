# Bugs Fixed in the Original Source

The original `simplexregression.cpp` downloaded from the Atlanta Fed website
(last updated 2024-04-30) contains two bugs that prevent compilation and
produce incorrect output.  Both are fixed in `optimized/simplexregression.cpp`.

---

## Bug 1 — R-style `#` comments in C++ source

### Symptom
`sourceCpp()` fails immediately:
```
simplexregression.cpp:1:1: error: expected unqualified-id
    1 | ###########################################
```

### Cause
The file uses `#` for all comment lines (R convention).  In C++, `#` begins
a preprocessor directive; lines like `# Market Probability Tracker R Code`
are not valid directives and cause a parse error.

### Fix
Replace every `#`-comment line with a `//`-comment.  The `#include` lines at
the top are genuine preprocessor directives and are left unchanged.

### Affected lines (original file)
Lines 1–15 (file header), lines 27–28, 35–36, 43–46, 54–55, 62–64,
72–73, 81–83, 98–99, 112–115, 124–130, 157–158, 167–168, 175–177,
190–192, 204–208.

---

## Bug 2 — `beta_prior` assigns to undeclared `like` instead of `prior`

### Symptom
Undefined behaviour / compiler error: `like` is not declared in `beta_prior`.

### Cause
Copy-paste error.  The function body was copied from `z_like` and the
assignment target was not updated:

```cpp
// ORIGINAL (buggy)
double beta_prior(double alpha, colvec xi, colvec beta_vec){
  double prior;
  like = lgamma(alpha) - ...;   // ← assigns to undeclared 'like'
  return prior;                 // ← returns uninitialised 'prior'
}
```

### Fix
```cpp
// FIXED
double beta_prior(double alpha, colvec xi, colvec beta_vec){
  double prior;
  prior = lgamma(alpha) - ...;  // ← correct target
  return prior;
}
```

### Impact
`beta_prior` is called from `logposterior`, which is computed every iteration
and stored as column 4 of the output matrix.  The bug makes the log-posterior
diagnostic column meaningless (undefined behaviour).  The *sampling* steps
(alpha, beta, B, sigma^2) are not affected because they do not call
`beta_prior` directly.

---

## Bug 3 (minor, pre-existing) — `// [[Rcpp::export]]` followed by comments

The original file places `//` comment lines between the `// [[Rcpp::export]]`
attribute and the function declaration.  Rcpp's attribute parser tries to
interpret those comment tokens as export parameters, generating the warning:

```
Invalid parameter: '1/k' for Rcpp::export attribute
```

In some Rcpp versions this prevents the function from being exported at all.

### Fix
Move all descriptive comments to *before* the `// [[Rcpp::export]]` line so
the attribute is the last thing before the function signature.
