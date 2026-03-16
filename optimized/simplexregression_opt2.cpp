///////////////////////////////////////////
// Market Probability Tracker - Opt v2  //
// Original: Brian Robertson, FRBA      //
//                                      //
// Carries forward all opt1 changes:    //
//   Precomputed XtX/Xtp, incremental   //
//   Xb/XtXb, O(1) Dirichlet log-pdf.  //
//                                      //
// Additional v2 optimisations:         //
//  1. Scalar RNG: R::runif/rlogis/     //
//     rgamma instead of Rcpp sugar     //
//     (eliminates 40M vector allocs).  //
//  2. Uniform-xi fast path: single     //
//     lgamma call instead of k, for    //
//     the standard xi=rep(1/k,k) case. //
// Same RNG stream as v1 (same R state, //
// same set.seed reproducibility).      //
///////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rmath.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Scalar RNG wrappers — same R RNG state as Rcpp sugar, no vector allocation
// ---------------------------------------------------------------------------
static inline double scalar_runif() {
    return R::runif(0.0, 1.0);
}

static inline double scalar_rlogis(double location, double scale) {
    return R::rlogis(location, scale);
}

static inline double scalar_rgamma(double shape, double scale) {
    return R::rgamma(shape, scale);
}

// ---------------------------------------------------------------------------
// Truncated normal draw via inverse CDF — scalar RNG
// ---------------------------------------------------------------------------
static double trunc_norm_invcdf(double a, double b, double m, double sd) {
    double p    = scalar_runif();
    double cd_a = R::pnorm(a, m, sd, 1, 0);
    double cd_b = R::pnorm(b, m, sd, 1, 0);
    return R::qnorm(cd_a + p * (cd_b - cd_a), m, sd, 1, 0);
}

// ---------------------------------------------------------------------------
// Scaled inverse chi-squared — scalar RNG
// ---------------------------------------------------------------------------
static double rScaledInvChiSq(double nu, double tau) {
    return 1.0 / scalar_rgamma(nu / 2.0, 1.0 / ((nu * tau) / 2.0));
}

// ---------------------------------------------------------------------------
// Dirichlet log-likelihood with uniform-xi fast path
//
// When xi is uniform (all elements = 1/k), the lgamma sum simplifies:
//   sum_j lgamma(alpha * xi[j]) = k * lgamma(alpha / k)
// This replaces k lgamma calls with 1.
// ---------------------------------------------------------------------------
static double dirichlet_lp_v2(double alpha, int k, bool xi_uniform,
                                const colvec& xi,
                                double xi_log_sum, double log_beta_sum) {
    double lgam_sum;
    if (xi_uniform) {
        lgam_sum = k * lgamma(alpha / k);
    } else {
        lgam_sum = 0.0;
        for (int j = 0; j < k; j++) lgam_sum += lgamma(alpha * xi[j]);
    }
    return lgamma(alpha) - lgam_sum + alpha * xi_log_sum - log_beta_sum;
}

// ---------------------------------------------------------------------------
// z_prior, z_posterior, z_met — same algorithm, scalar RNG + uniform-xi
// ---------------------------------------------------------------------------
static double z_prior(double z, double zeta, double tau) {
    return log(pow(2.0 * tau * (1.0 + cosh((z - log(zeta)) / tau)), -1.0));
}

static double z_posterior_v2(double z, int k, bool xi_uniform,
                              const colvec& xi,
                              double xi_log_sum, double log_beta_sum,
                              double zeta, double tau) {
    double alpha = exp(z);
    return dirichlet_lp_v2(alpha, k, xi_uniform, xi, xi_log_sum, log_beta_sum)
           + z_prior(z, zeta, tau);
}

static double z_met_v2(double alpha, int k, bool xi_uniform,
                        const colvec& xi,
                        double xi_log_sum, double log_beta_sum,
                        double zeta, double tau, double alpha_step) {
    double z0    = log(alpha);
    double z1    = scalar_rlogis(z0, tau * alpha_step);
    double v0    = z_posterior_v2(z0, k, xi_uniform, xi, xi_log_sum, log_beta_sum, zeta, tau);
    double v1    = z_posterior_v2(z1, k, xi_uniform, xi, xi_log_sum, log_beta_sum, zeta, tau);
    double ratio = v1 - v0;
    double test  = log(scalar_runif());
    if (ratio >= test) alpha = exp(z1);
    return alpha;
}

// ---------------------------------------------------------------------------
// beta_met_v2 — scalar RNG, otherwise identical to beta_met_opt
// ---------------------------------------------------------------------------
static colvec beta_met_v2(const mat& X, const mat& XtX, const colvec& Xtp,
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

        double b      = beta_vec[j] + beta_vec[k_index];
        double XtX_jk = XtX(j, k_index);

        double XtX_kj2 = XtX(j, j) - 2.0 * XtX_jk + XtX_kk;
        if (XtX_kj2 <= 0.0) continue;

        double num = (Xtp[j] - Xtp[k_index]) / B
                     - (XtXb[j] - XtXb[k_index] - XtX_jk + XtX_kk)
                     - (XtX_jk  - XtX_kk);
        double m   = beta_vec[j] + num / XtX_kj2;
        double v   = sigsq / (B * B * XtX_kj2);

        double prop  = trunc_norm_invcdf(0.0, b, m, sqrt(v));
        double ratio = pow(prop / beta_vec[j], q[j]) *
                       pow((b - prop) / beta_vec[k_index], q[k_index]);

        if (ratio >= scalar_runif()) {
            double delta = prop - beta_vec[j];
            double old_j = beta_vec[j];
            double old_k = beta_vec[k_index];

            beta_vec[j]       = prop;
            beta_vec[k_index] = b - prop;

            // Incremental updates — O(k) and O(n), Armadillo SIMD
            XtXb += delta * (XtX.col(j) - XtX.col(k_index));
            Xb   += delta * (X.col(j)   - X.col(k_index));

            // O(1) scalar updates
            double dlj = log(beta_vec[j])       - log(old_j);
            double dlk = log(beta_vec[k_index]) - log(old_k);
            xi_log_sum   += xi[j] * dlj + xi[k_index] * dlk;
            log_beta_sum += dlj + dlk;
        }
    }
    return beta_vec;
}

// ---------------------------------------------------------------------------
// b_gibbs_v2 — scalar RNG
// ---------------------------------------------------------------------------
static double b_gibbs_v2(const colvec& price, const colvec& Xb,
                          double sigsq, double B_m, double B_v) {
    double Xb2    = dot(Xb, Xb);
    double md     = dot(price, Xb) / Xb2;
    double vd     = sigsq / Xb2;
    double b_mean = B_m * vd / (B_v + vd) + md * B_v / (B_v + vd);
    double b_sd   = sqrt(B_v * vd / (B_v + vd));
    return trunc_norm_invcdf(0.0, 1e10, b_mean, b_sd);
}

// ---------------------------------------------------------------------------
// get_ssq
// ---------------------------------------------------------------------------
static double get_ssq_v2(const colvec& price, const colvec& Xb, double B) {
    colvec r = price - B * Xb;
    return dot(r, r);
}

// ---------------------------------------------------------------------------
// logposterior_v2 — uniform-xi fast path
// ---------------------------------------------------------------------------
static double logposterior_v2(double alpha, double B, double sigsq,
                               double zeta, double tau, int k, bool xi_uniform,
                               const colvec& xi,
                               double xi_log_sum, double log_beta_sum,
                               const colvec& price, const colvec& Xb,
                               double B_m, double B_v) {
    int n = (int)price.n_rows;
    double ssq    = get_ssq_v2(price, Xb, B);
    double sig_lp = -0.5 * n * log(sigsq) - ssq / (2.0 * sigsq);
    double B_lp   = -0.5 * pow(B - B_m, 2.0) / B_v;
    double bet_lp = dirichlet_lp_v2(alpha, k, xi_uniform, xi, xi_log_sum, log_beta_sum);
    double alp_lp = (1.0 / tau - 1.0) * log(alpha)
                    - 2.0 * log(pow(alpha, 1.0 / tau) + pow(zeta, 1.0 / tau));
    return sig_lp + B_lp + bet_lp + alp_lp;
}

// ---------------------------------------------------------------------------
// get_bvec_cpp_opt2 — same interface and statistical algorithm as opt/opt1.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
mat get_bvec_cpp_opt2(int draws, int burn, int thin, int k, int out_length,
                       double alpha, double zeta, double tau, double alpha_step,
                       double B, double sigsq, double B_m, double B_v,
                       colvec price, colvec beta_vec, colvec xi, mat X) {
    // Detect uniform xi
    bool xi_uniform = true;
    double xi0 = xi[0];
    for (int j = 1; j < k; j++) {
        if (std::fabs(xi[j] - xi0) > 1e-15) { xi_uniform = false; break; }
    }

    // One-time precomputations
    mat    XtX  = X.t() * X;
    colvec Xtp  = X.t() * price;
    colvec Xb   = X * beta_vec;
    colvec XtXb = XtX * beta_vec;

    // Running scalars for Dirichlet log-pdf
    colvec log_bv      = log(beta_vec);
    double xi_log_sum  = dot(xi, log_bv);
    double log_beta_sum = sum(log_bv);

    double lp;
    int    ii = 0;
    colvec q;
    mat output_mat(out_length, k + 4);

    for (int i = 0; i < draws; i++) {
        // Sample alpha via MH
        alpha = z_met_v2(alpha, k, xi_uniform, xi, xi_log_sum, log_beta_sum,
                          zeta, tau, alpha_step);
        q = alpha * xi - 1.0;

        // Sample beta via MH
        beta_vec = beta_met_v2(X, XtX, Xtp, beta_vec, q, B, sigsq, xi,
                                Xb, XtXb, xi_log_sum, log_beta_sum);

        // Sample sigma^2 via Gibbs
        double ssq = get_ssq_v2(price, Xb, B);
        sigsq = rScaledInvChiSq((double)price.n_rows, ssq / (double)price.n_rows);

        // Sample B via Gibbs
        B = b_gibbs_v2(price, Xb, sigsq, B_m, B_v);

        // Log-posterior diagnostic
        lp = logposterior_v2(alpha, B, sigsq, zeta, tau, k, xi_uniform, xi,
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
