///////////////////////////////////////////
// Market Probability Tracker - Optimized//
// Original: Brian Robertson, FRBA       //
// Optimizations (no methodology change)://
//  1. Precompute XtX=X'X, Xtp=X'price  //
//     once before the MCMC loop.        //
//  2. Maintain Xb=X*beta and            //
//     XtXb=X'X*beta incrementally       //
//     (O(n) and O(k) updates) instead   //
//     of recomputing O(n*k) each iter.  //
//  3. Maintain xi_log_sum=sum(xi*logB)  //
//     and log_beta_sum=sum(log(beta))   //
//     for O(1) Dirichlet log-pdf.       //
// These reduce per-iteration cost from  //
// O(n*k^2) to O(k^2+n*k), an ~n-fold   //
// speedup in the MCMC sampler.          //
///////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rmath.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Unchanged helpers from original
// ---------------------------------------------------------------------------

static double trunc_norm_invcdf(double a, double b, double m, double sd) {
  double p    = runif(1)[0];
  double cd_a = R::pnorm(a, m, sd, 1, 0);
  double cd_b = R::pnorm(b, m, sd, 1, 0);
  return R::qnorm(cd_a + p * (cd_b - cd_a), m, sd, 1, 0);
}

static double rScaledInvChiSq(double nu, double tau) {
  return 1.0 / rgamma(1, nu / 2.0, 1.0 / ((nu * tau) / 2.0))[0];
}

// ---------------------------------------------------------------------------
// Dirichlet log-likelihood (same as original z_like / beta_prior).
//
// log p(beta | alpha, xi) =
//   lgamma(alpha) - sum_j lgamma(alpha*xi[j])
//   + alpha*xi_log_sum - log_beta_sum
//
// where xi_log_sum = sum_j xi[j]*log(beta[j])    (maintained incrementally)
//       log_beta_sum = sum_j log(beta[j])          (maintained incrementally)
//
// The "alpha - 1" form in the original is:
//   sum_j (alpha*xi[j]-1)*log(beta[j])
//   = alpha*xi_log_sum - log_beta_sum
// ---------------------------------------------------------------------------
static double dirichlet_lp(double alpha, const colvec& xi,
                            double xi_log_sum, double log_beta_sum) {
  double lgam_sum = 0.0;
  for (uword j = 0; j < xi.n_rows; j++) lgam_sum += lgamma(alpha * xi[j]);
  return lgamma(alpha) - lgam_sum + alpha * xi_log_sum - log_beta_sum;
}

// ---------------------------------------------------------------------------
// z_prior, z_posterior, z_met — same algorithm
// ---------------------------------------------------------------------------
static double z_prior(double z, double zeta, double tau) {
  return log(pow(2.0 * tau * (1.0 + cosh((z - log(zeta)) / tau)), -1.0));
}

static double z_posterior_opt(double z, const colvec& xi,
                               double xi_log_sum, double log_beta_sum,
                               double zeta, double tau) {
  double alpha = exp(z);
  return dirichlet_lp(alpha, xi, xi_log_sum, log_beta_sum)
         + z_prior(z, zeta, tau);
}

static double z_met_opt(double alpha, const colvec& xi,
                         double xi_log_sum, double log_beta_sum,
                         double zeta, double tau, double alpha_step) {
  double z0    = log(alpha);
  double z1    = rlogis(1, z0, tau * alpha_step)[0];
  double v0    = z_posterior_opt(z0, xi, xi_log_sum, log_beta_sum, zeta, tau);
  double v1    = z_posterior_opt(z1, xi, xi_log_sum, log_beta_sum, zeta, tau);
  double ratio = v1 - v0;
  double test  = log(runif(1)[0]);
  if (ratio >= test) alpha = exp(z1);
  return alpha;
}

// ---------------------------------------------------------------------------
// beta_met_opt — same algorithm as original, O(k^2+n*k) per call vs O(n*k^2)
//
// Uses precomputed XtX (k×k) and Xtp (k-vector), and maintains in-place:
//   Xb         = X * beta_vec           (n-vector, updated O(n) per accept)
//   XtXb       = XtX * beta_vec         (k-vector, updated O(k) per accept)
//   xi_log_sum = sum_j xi[j]*log(b[j])  (scalar,   updated O(1) per accept)
//   log_beta_sum = sum_j log(b[j])       (scalar,   updated O(1) per accept)
// ---------------------------------------------------------------------------
static colvec beta_met_opt(const mat& X, const mat& XtX, const colvec& Xtp,
                            colvec beta_vec, const colvec& q,
                            double B, double sigsq, const colvec& xi,
                            colvec& Xb, colvec& XtXb,
                            double& xi_log_sum, double& log_beta_sum) {
  uword k_index;
  beta_vec.max(k_index);

  int k = (int)XtX.n_cols;
  double XtX_kk = XtX(k_index, k_index);

  for (int j = 0; j < k; j++) {
    if (j == (int)k_index) continue;

    double b     = beta_vec[j] + beta_vec[k_index];
    double XtX_jk = XtX(j, k_index);

    // --- Proposal mean m (eq 5.15a) using precomputed quantities: O(1) ---
    //
    // Expanded from original: X_kj'*price/B - X_kj'*X_kindex - X_kj'*X_k*beta
    // = (Xtp[j]-Xtp[k])/B  - (XtXb[j]-XtXb[k]) + constant
    // (full derivation in comments of simplexregression_opt.cpp header)
    double XtX_kj2  = XtX(j,j) - 2.0*XtX_jk + XtX_kk;
    if (XtX_kj2 <= 0.0) continue;

    // numerator = X_kj'*price/B - X_kj'*X_k*beta
    //           = (Xtp[j]-Xtp[k])/B - (XtXb[j]-XtXb[k]-XtX[j,k]+XtX[k,k])
    //             - (XtX[j,k]-XtX[k,k])           [= -X_kj'*X_kindex]
    // simplifies to: (Xtp[j]-Xtp[k])/B - (XtXb[j]-XtXb[k])
    double num = (Xtp[j] - Xtp[k_index]) / B
                 - (XtXb[j] - XtXb[k_index] - XtX_jk + XtX_kk)
                 - (XtX_jk  - XtX_kk);
    double m   = beta_vec[j] + num / XtX_kj2;
    double v   = sigsq / (B * B * XtX_kj2);

    double prop  = trunc_norm_invcdf(0.0, b, m, sqrt(v));
    double ratio = pow(prop / beta_vec[j], q[j]) *
                   pow((b - prop) / beta_vec[k_index], q[k_index]);

    if (ratio >= runif(1)[0]) {
      double delta  = prop - beta_vec[j];
      double old_j  = beta_vec[j];
      double old_k  = beta_vec[k_index];

      beta_vec[j]       = prop;
      beta_vec[k_index] = b - prop;

      // Incremental updates — O(k) and O(n) each
      XtXb       += delta * (XtX.col(j) - XtX.col(k_index));
      Xb         += delta * (X.col(j)   - X.col(k_index));

      // O(1) scalar updates for the Dirichlet log-pdf
      double dlj = log(beta_vec[j])       - log(old_j);
      double dlk = log(beta_vec[k_index]) - log(old_k);
      xi_log_sum  += xi[j] * dlj + xi[k_index] * dlk;
      log_beta_sum += dlj + dlk;
    }
  }
  return beta_vec;
}

// ---------------------------------------------------------------------------
// b_gibbs_opt — accepts maintained Xb instead of recomputing X*beta: O(n)
// ---------------------------------------------------------------------------
static double b_gibbs_opt(const colvec& price, const colvec& Xb,
                           double sigsq, double B_m, double B_v) {
  double Xb2    = dot(Xb, Xb);
  double md     = dot(price, Xb) / Xb2;
  double vd     = sigsq / Xb2;
  double b_mean = B_m * vd / (B_v + vd) + md * B_v / (B_v + vd);
  double b_sd   = sqrt(B_v * vd / (B_v + vd));
  return trunc_norm_invcdf(0.0, 1e10, b_mean, b_sd);
}

// ---------------------------------------------------------------------------
// get_ssq_opt — O(n) using maintained Xb
// ---------------------------------------------------------------------------
static double get_ssq_opt(const colvec& price, const colvec& Xb, double B) {
  colvec r = price - B * Xb;
  return dot(r, r);
}

// ---------------------------------------------------------------------------
// logposterior_opt — same formula as original, all O(k) or O(n)
// ---------------------------------------------------------------------------
static double logposterior_opt(double alpha, double B, double sigsq,
                                double zeta, double tau, const colvec& xi,
                                double xi_log_sum, double log_beta_sum,
                                const colvec& price, const colvec& Xb,
                                double B_m, double B_v) {
  int n = (int)price.n_rows;
  double ssq    = get_ssq_opt(price, Xb, B);
  double sig_lp = -0.5 * n * log(sigsq) - ssq / (2.0 * sigsq);
  double B_lp   = -0.5 * pow(B - B_m, 2.0) / B_v;
  double bet_lp = dirichlet_lp(alpha, xi, xi_log_sum, log_beta_sum);
  double alp_lp = (1.0 / tau - 1.0) * log(alpha)
                  - 2.0 * log(pow(alpha, 1.0/tau) + pow(zeta, 1.0/tau));
  return sig_lp + B_lp + bet_lp + alp_lp;
}

// ---------------------------------------------------------------------------
// get_bvec_cpp_opt — identical interface to get_bvec_cpp; same statistical
// algorithm; precomputes XtX/Xtp once, maintains Xb/XtXb/xi_log_sum
// incrementally throughout the MCMC loop.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
mat get_bvec_cpp_opt(int draws, int burn, int thin, int k, int out_length,
                     double alpha, double zeta, double tau, double alpha_step,
                     double B, double sigsq, double B_m, double B_v,
                     colvec price, colvec beta_vec, colvec xi, mat X) {
  // One-time precomputations
  mat    XtX  = X.t() * X;
  colvec Xtp  = X.t() * price;
  colvec Xb   = X * beta_vec;
  colvec XtXb = XtX * beta_vec;

  // Running scalars for Dirichlet log-pdf
  colvec log_bv    = log(beta_vec);
  double xi_log_sum  = dot(xi, log_bv);
  double log_beta_sum = sum(log_bv);

  double lp;
  int    ii = 0;
  colvec q;
  mat output_mat(out_length, k + 4);

  for (int i = 0; i < draws; i++) {
    // Periodic recomputation to prevent floating-point drift in the maintained
    // quantities. Incremental updates (O(n) and O(k) each) accumulate rounding
    // error over very long runs; refreshing every 10 000 iterations costs
    // O(n*k + k^2) ≈ 0.01% of total work and keeps errors at machine epsilon.
    if (i > 0 && i % 10000 == 0) {
      Xb   = X   * beta_vec;
      XtXb = XtX * beta_vec;
      colvec lbv   = log(beta_vec);
      xi_log_sum   = dot(xi, lbv);
      log_beta_sum = sum(lbv);
    }

    // Sample alpha via MH (O(k) for lgamma sums — unavoidable per iteration)
    alpha = z_met_opt(alpha, xi, xi_log_sum, log_beta_sum, zeta, tau, alpha_step);
    q     = alpha * xi - 1.0;

    // Sample beta via MH — O(k^2 + n*k) per call
    beta_vec = beta_met_opt(X, XtX, Xtp, beta_vec, q, B, sigsq, xi,
                            Xb, XtXb, xi_log_sum, log_beta_sum);

    // Sample sigma^2 via Gibbs — O(n)
    double ssq = get_ssq_opt(price, Xb, B);
    sigsq = rScaledInvChiSq((double)price.n_rows, ssq / (double)price.n_rows);

    // Sample B via Gibbs — O(n)
    B = b_gibbs_opt(price, Xb, sigsq, B_m, B_v);

    // Log-posterior diagnostic — O(n)
    lp = logposterior_opt(alpha, B, sigsq, zeta, tau, xi,
                          xi_log_sum, log_beta_sum, price, Xb, B_m, B_v);

    if ((i + 1) > burn && (i + 1) % thin == 0) {
      output_mat(ii, 0) = alpha;
      output_mat(ii, 1) = B;
      output_mat(ii, 2) = sigsq;
      output_mat(ii, 3) = lp;
      for (int j = 0; j < k; j++) output_mat(ii, j + 4) = beta_vec[j];
      ii++;
    }
  }
  return output_mat;
}
