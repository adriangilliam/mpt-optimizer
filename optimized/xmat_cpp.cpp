///////////////////////////////////////////
// C++ X matrix integration using the   //
// same tanh-sinh (double exponential)   //
// quadrature as pracma::quadinf.        //
//                                       //
// Key optimisations:                    //
//  1. Eliminate R function dispatch:    //
//     all pnorm/dnorm/dbeta calls go   //
//     directly through the C API.      //
//  2. Share pnorm/dnorm across the k   //
//     basis functions: at each quad    //
//     node, compute pnorm and dnorm    //
//     once, then loop over k dbeta     //
//     evaluations.  Saves 79/80 of     //
//     the pnorm/dnorm calls.           //
//  3. No methodology change: same      //
//     integrands, same variable        //
//     transform, same 7-level          //
//     Richardson extrapolation.        //
///////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rmath.h>
#include <cmath>
#include <vector>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Tanh-sinh quadrature tables
//
// pracma::quadinf uses the double-exponential / tanh-sinh transform:
//   x(t) = tanh(pi/2 * sinh(t))
//   w(t) = pi/2 * cosh(t) / cosh^2(pi/2 * sinh(t))
//
// Level 0 (quadinf "level 1"): step h=0.5, nodes at t = 0, 0.5, 1.0, ..., 3.0
// Level m>0 (quadinf "level m+1"): new interleaved nodes at odd multiples
//   of h_m = 0.5/2^m, i.e. t = h_m, 3*h_m, 5*h_m, ...
//
// The Richardson accumulation is:
//   Level 0: Q = sum * 0.5
//   Level m: Q_new = sum_m * h_m + Q_old / 2
// where h halves at each level (0.5, 0.25, 0.125, ..., 1/256).
// ---------------------------------------------------------------------------
struct TSLevel {
    std::vector<double> x;    // tanh-sinh nodes on (0, 1)
    std::vector<double> w;    // weights (WITHOUT the h factor)
};

static std::vector<TSLevel> precompute_ts() {
    const double pi2 = M_PI / 2.0;
    std::vector<TSLevel> levels(7);

    // Level 0: t = 0, 0.5, 1.0, ..., 3.0  (7 nodes including center)
    for (int j = 0; j <= 6; j++) {
        double t = j * 0.5;
        double u = pi2 * std::sinh(t);
        double cu = std::cosh(u);
        levels[0].x.push_back(std::tanh(u));
        levels[0].w.push_back(pi2 * std::cosh(t) / (cu * cu));
    }

    // Levels 1-6: interleaved nodes
    for (int lev = 1; lev < 7; lev++) {
        double step = 0.5 / (1 << lev);            // 0.25, 0.125, ...
        int n_new = 6 * (1 << lev);                // 6, 12, 24, 48, 96, 192
        for (int j = 0; j < n_new; j++) {
            double t = (2 * j + 1) * step;
            double u = pi2 * std::sinh(t);
            double cu = std::cosh(u);
            double wt = pi2 * std::cosh(t) / (cu * cu);
            if (wt < 1e-30) break;                 // negligible weight, stop
            levels[lev].x.push_back(std::tanh(u));
            levels[lev].w.push_back(wt);
        }
    }
    return levels;
}

// Initialised once when the shared library is loaded
static const std::vector<TSLevel> ts_levels = precompute_ts();

// ---------------------------------------------------------------------------
// Helper: evaluate the integrand contribution for one quadrature node
// and accumulate into the sum vector s[0..k-1].
//
// For cap  (flag=1): f(x) = (x-K) * dbeta(Phi(x), j+1, k-j) * phi(x)
// For floor(flag=0): f(x) = (K-x) * dbeta(Phi(x), j+1, k-j) * phi(x)
//
// The caller passes the physical-space x, the Jacobian, and the quadrature
// weight.  pnorm and dnorm are computed here (once) and shared across all
// k dbeta calls.
// ---------------------------------------------------------------------------
static inline void eval_node(double phys_x, double K, int flag,
                              double m, double sig, int k,
                              double jac_w,               // jacobian * weight
                              std::vector<double>& s) {
    double dn = R::dnorm(phys_x, m, sig, 0);
    if (dn < 1e-300) return;                            // negligible contribution

    double pn  = R::pnorm(phys_x, m, sig, 1, 0);
    double pay = (flag == 1) ? (phys_x - K) : (K - phys_x);
    double common = jac_w * pay * dn;

    for (int b = 0; b < k; b++) {
        double db = R::dbeta(pn, (double)(b + 1), (double)(k - b), 0);
        s[b] += common * db;
    }
}

// ---------------------------------------------------------------------------
// get_xmat_cpp — drop-in replacement for the R get_xmat / get_xmat_serial
//
// Arguments:
//   m         — forward rate (decimal, e.g. 0.036)
//   sig       — 3 * sigma_ols (decimal), same as the R code's sig = 3*tp$sigma_ols/100
//   strikes   — vector of strike rates (decimal)
//   opt_flags — 1 = cap (integrates [K, inf)), 0 = floor (integrates (-inf, K])
//   k         — number of basis functions (80)
//   tol       — convergence tolerance; 0 = run all 7 levels (matches reltol=0)
//
// Returns an (n x k) matrix identical to the R version.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat get_xmat_cpp(double m, double sig,
                        arma::vec strikes, arma::ivec opt_flags,
                        int k, double tol = 0.0) {
    int n = (int)strikes.n_elem;
    mat X(n, k, fill::zeros);

    for (int i = 0; i < n; i++) {
        double K    = strikes[i];
        int    flag = opt_flags[i];

        std::vector<double> Q(k, 0.0);
        double h = 0.5;

        // ---- Level 0 (quadinf level 1): center node + 6 symmetric pairs ---
        {
            const TSLevel& lev = ts_levels[0];
            std::vector<double> s(k, 0.0);

            // Center node (index 0, y=0)
            {
                double phys_x = (flag == 1) ? K + 1.0 : K - 1.0;
                double jac_w  = lev.w[0] * 2.0;        // jac = 2/(1-0)^2 = 2
                eval_node(phys_x, K, flag, m, sig, k, jac_w, s);
            }

            // Symmetric pairs (indices 1..6)
            for (int j = 1; j < (int)lev.x.size(); j++) {
                double y = lev.x[j];

                // +y  →  r = (y+1)/(1-y),  jac = 2/(1-y)^2
                double omy  = 1.0 - y;
                double r_p  = (y + 1.0) / omy;
                double jac_p = 2.0 / (omy * omy);
                double px_p = (flag == 1) ? K + r_p : K - r_p;

                // -y  →  r = (1-y)/(1+y),  jac = 2/(1+y)^2
                double opy  = 1.0 + y;
                double r_m  = omy / opy;
                double jac_m = 2.0 / (opy * opy);
                double px_m = (flag == 1) ? K + r_m : K - r_m;

                eval_node(px_p, K, flag, m, sig, k, lev.w[j] * jac_p, s);
                eval_node(px_m, K, flag, m, sig, k, lev.w[j] * jac_m, s);
            }

            for (int b = 0; b < k; b++) Q[b] = s[b] * h;
        }

        // ---- Levels 1-6 (quadinf levels 2-7) ------------------------------
        for (int lev_idx = 1; lev_idx < 7; lev_idx++) {
            const TSLevel& lev = ts_levels[lev_idx];
            h /= 2.0;
            std::vector<double> s(k, 0.0);

            for (int j = 0; j < (int)lev.x.size(); j++) {
                double y   = lev.x[j];
                double omy = 1.0 - y;
                double opy = 1.0 + y;

                // +y
                if (omy > 1e-18) {
                    double r_p  = (y + 1.0) / omy;
                    double jac_p = 2.0 / (omy * omy);
                    double px_p = (flag == 1) ? K + r_p : K - r_p;
                    eval_node(px_p, K, flag, m, sig, k, lev.w[j] * jac_p, s);
                }

                // -y
                {
                    double r_m  = omy / opy;
                    double jac_m = 2.0 / (opy * opy);
                    double px_m = (flag == 1) ? K + r_m : K - r_m;
                    eval_node(px_m, K, flag, m, sig, k, lev.w[j] * jac_m, s);
                }
            }

            // Richardson extrapolation: Q_new = s*h + Q_old/2
            double max_delta = 0.0;
            for (int b = 0; b < k; b++) {
                double newQ = s[b] * h + Q[b] / 2.0;
                double delta = std::fabs(newQ - Q[b]);
                if (delta > max_delta) max_delta = delta;
                Q[b] = newQ;
            }

            if (tol > 0.0 && max_delta < tol) break;
        }

        for (int b = 0; b < k; b++) X(i, b) = Q[b];
    }

    return X;
}
