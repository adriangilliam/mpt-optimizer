################################################################################
# Multi-contract production-scale benchmark
#
# Simulates the full MPT production run:
#   6 SOFR futures contracts: Jun 2026 – Sep 2027 (one per ~2 FOMC windows)
#   Observation date: 2026-03-15
#   Spot SOFR: 3.65% (FRED, 2026-03-12)
#   FOMC schedule (federalreserve.gov): Mar 17-18, Apr 28-29, Jun 16-17,
#     Jul 28-29, Sep 15-16, Oct 27-28, Dec 8-9 (2026); Mar/Jun/Sep 2027
#
# Forward curve: gradual 25bp cut per ~2 meetings, consistent with "hold" at
#   Jan 2026 FOMC and current pricing in 5-6 cuts through Sep 2027.
# Normal vol term structure: 1.00% (3M) -> 1.25% (18M), upward sloping.
# Strike grid: 25bp, range = max(3.5*sigma*sqrt(T), 3%), ~30-70 rows/contract.
#
# Both original (get_bvec_cpp) and optimised (get_bvec_cpp_opt) samplers run
# on identical inputs for each contract. Per-contract and aggregate timings
# and the n*k/(k+n) theoretical speedup heuristic are reported.
################################################################################

suppressPackageStartupMessages({
  library(pracma); library(Rsolnp); library(reshape)
  library(Rcpp);   library(RcppArmadillo)
})

sourceCpp(file.path(dirname(sys.frame(1)$ofile), '..', 'optimized', 'simplexregression.cpp'))
sourceCpp(file.path(dirname(sys.frame(1)$ofile), '..', 'optimized', 'simplexregression_opt.cpp'))

# ---- helpers ----------------------------------------------------------------
get_rate = function(tr, mat) approx(tr$mat, tr$rate, xout=mat, rule=2)$y

bachelier_price = function(F, K, T, sig, flag) {
  d = (F - K) / (sig * sqrt(T))
  if (flag == 1) (F-K)*pnorm(d)  + sig*sqrt(T)*dnorm(d)   # cap  (put on futures)
  else           (K-F)*pnorm(-d) + sig*sqrt(T)*dnorm(d)   # floor (call on futures)
}

norm_solvenlm = function(strikes, F, prices, B, flags, T_exp) {
  obj = function(sig) {
    if (sig <= 0) return(1e10)
    phat = B * sapply(seq_along(strikes), function(i)
             bachelier_price(F, strikes[i], T_exp, sig, flags[i]))
    sum((prices - phat)^2)
  }
  tryCatch(optimise(obj, interval=c(1e-5, 0.10))$minimum, error=function(e) 0.01)
}

f_p = function(x,parms) (parms[1]-x)*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])
f_c = function(x,parms) (x-parms[1])*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])

get_transformparms = function(options, futures, treas, bs, T_exp) {
  dys=unique(options$d_y); date=as.Date(unique(options$date))
  contract=unique(options$contract); expy_date=as.Date(unique(options$expy))
  lm.out=data.frame(d_y=dys, date=date, contract=contract,
    days_to_expy=as.numeric(expy_date-date), F_ols=NA, B_ols=NA, sigma_ols=NA,
    basis_spread=NA, est_type=NA, yield_curve=NA, option_count=0L, option_overlap=0L,
    stringsAsFactors=FALSE)
  lm.out$basis_spread = bs
  x     = cast(options[options$settle > 0, ], rate_strike ~ put_call, value='settle')
  x_ov  = 0L; x_cnt = nrow(x)
  lm.out$yield_curve = ifelse(unique(options$date)==unique(treas$date),'current','lagged')
  if (length(names(x)) == 3) { x = x[!is.na(x$C) & !is.na(x$P), ]; x_ov = nrow(x) }
  if (x_ov < 2) {
    lm.out$F_ols = futures$rate_settle
    lm.out$B_ols = exp(-get_rate(treas, futures$mat) * futures$mat)
    lm.out$est_type = 'data'
  } else {
    x$y = x$P - x$C; tmp = lm(x$y ~ x$rate_strike)
    lm.out$B_ols = -tmp$coefficients[2]
    lm.out$F_ols =  tmp$coefficients[1] / -tmp$coefficients[2]
    lm.out$est_type = 'ols'
  }
  lm.out$option_count = x_cnt; lm.out$option_overlap = x_ov
  if (!is.na(lm.out$F_ols))
    lm.out$sigma_ols = norm_solvenlm(options$rate_strike, lm.out$F_ols, options$settle,
                                     lm.out$B_ols, options$option_flag, T_exp)
  lm.out$F_ols     = lm.out$F_ols     * 100   # store as %
  lm.out$sigma_ols = lm.out$sigma_ols * 100
  lm.out
}

get_xmat_serial = function(tp, options, k) {
  m=tp$F_ols/100; sig=3*tp$sigma_ols/100
  s=options$rate_strike; opt=options$option_flag; sl=length(s)
  X = matrix(0.0, nrow=sl, ncol=k)
  for (j in 1:k) for (i in 1:sl) {
    parms = c(s[i], j, k, m, sig)
    X[i,j] = suppressMessages(
      if (opt[i]==0L) integral(f_p, -Inf, s[i], parms=parms, reltol=0)
      else            integral(f_c,  s[i],  Inf, parms=parms, reltol=0)
    )
  }
  X
}

get_ssq_r = function(beta,B,X,y) sum((y - B*X%*%beta)^2)
eqfn      = function(beta,B,X,y) sum(beta)
ineqgfn   = function(beta,B,X,y) beta

# ---- Contract specifications ------------------------------------------------
# 2026-03-15 forward curve built from:
#   - SOFR spot 3.65% (FRED 2026-03-12)
#   - Hold at Mar 17-18 meeting priced as near-certain
#   - ~25bp cuts at Jun, Sep, Dec, Mar, Jun, Sep meetings (5 cuts total)
# IMM expiry dates: 3rd Wednesday of Mar/Jun/Sep/Dec
# Normal vol term structure: 1.00% for 3M, +5bp per additional quarter
obs_date = as.Date('2026-03-15')

# ---- Treasury yield curve (2026-03-13, one business day prior) -------------
treas = data.frame(
  date = obs_date,
  mat  = c(1/12,2/12,3/12,4/12,6/12,1,2,3,5,7,10,20,30),
  rate = c(0.0375,0.0371,0.0372,0.0369,0.0370,0.0366,0.0373,0.0374,0.0387,0.0407,0.0428,0.0489,0.0490),
  stringsAsFactors=FALSE
)

contracts = data.frame(
  name      = c('SRM26',       'SRU26',       'SRZ26',       'SRH27',       'SRM27',       'SRU27'),
  expy      = as.Date(c('2026-06-17','2026-09-16','2026-12-16','2027-03-17','2027-06-16','2027-09-15')),
  F_fwd     = c(0.0365,         0.0340,         0.0315,         0.0290,         0.0265,         0.0240),
  sigma     = c(0.0100,         0.0105,         0.0110,         0.0115,         0.0120,         0.0125),
  fomc_cvrd = c('Mar+Apr',      'Jun+Jul',      'Sep+Oct',      'Dec+Q1-27',   'Q1-27',        'Q2-27'),
  seed_data = c(42L, 43L, 44L, 45L, 46L, 47L),
  stringsAsFactors=FALSE
)

# ---- MCMC hyperparameters (production settings) ----------------------------
k=80; draws=250000; burn=100000; thin=150
alpha=10; alpha_step=0.1; tau=1; zeta=1
xi=rep(1/k,k); B_m=1; B_v=1e12; sigsq=1

# ---- Storage ----------------------------------------------------------------
results = data.frame(
  contract    = contracts$name,
  n_rows      = 0L,
  T_exp       = 0.0,
  t_xmat      = 0.0,
  t_solnp     = 0.0,
  t_mcmc_orig = 0.0,
  t_mcmc_opt  = 0.0,
  speedup     = 0.0,
  stringsAsFactors=FALSE
)

# ---- Per-contract loop ------------------------------------------------------
cat('=== MULTI-CONTRACT PRODUCTION-SCALE BENCHMARK ===\n')
cat(sprintf('Observation: %s  |  k=%d  |  draws=%dk  burn=%dk  thin=%d\n\n',
            obs_date, k, draws/1e3, burn/1e3, thin))
cat(sprintf('%-8s %-12s %5s %5s %7s %7s %8s %7s %7s\n',
            'Contract','FOMC-covered','n','T(y)','F(%)','Xmat(s)','solnp(s)','Orig(s)','Opt(s)'))
cat(strrep('-', 80), '\n')

for (ci in seq_len(nrow(contracts))) {
  cr = contracts[ci, ]

  # generate synthetic options data
  days_exp = as.numeric(cr$expy - obs_date)
  T_exp    = days_exp / 365
  B_true   = exp(-get_rate(treas, T_exp) * T_exp)
  sig_adj  = cr$sigma * sqrt(T_exp)

  # strike range: wider of (3.5*sigma_adj, 3%) for realistic CME chain depth
  rng   = max(0.030, 3.5 * sig_adj)
  K_min = max(0.005, round((cr$F_fwd - rng) / 0.0025) * 0.0025)
  K_max =            round((cr$F_fwd + rng) / 0.0025) * 0.0025
  strikes = seq(K_min, K_max, by=0.0025)

  set.seed(cr$seed_data)
  rows = do.call(rbind, lapply(strikes, function(K) {
    cap_p   = B_true * bachelier_price(cr$F_fwd, K, T_exp, cr$sigma, 1)
    floor_p = B_true * bachelier_price(cr$F_fwd, K, T_exp, cr$sigma, 0)
    noise   = rnorm(2, 0, 1e-5)
    data.frame(
      date=obs_date, contract=cr$name, expy=cr$expy, d_y=days_exp,
      rate_strike=K,
      settle=c(max(cap_p+noise[1], 1e-8), max(floor_p+noise[2], 1e-8)),
      put_call=c('P','C'), option_flag=c(1L,0L),
      stringsAsFactors=FALSE
    )
  }))
  rows = rows[rows$settle > 1e-6, ]

  futures = data.frame(date=obs_date, contract=cr$name,
                       rate_settle=cr$F_fwd, mat=T_exp, stringsAsFactors=FALSE)
  tp = get_transformparms(rows, futures, treas, 0, T_exp)

  # X matrix
  t0  = proc.time()
  X   = get_xmat_serial(tp, rows, k)
  t_x = (proc.time() - t0)['elapsed']

  price = rows$settle

  # warm-start
  t0      = proc.time()
  beta0   = solnp(rep(1/k, k), get_ssq_r, eqfun=eqfn, eqB=1,
                  ineqfun=ineqgfn, ineqLB=rep(0,k), ineqUB=rep(1,k),
                  B=tp$B_ols, X=X, y=price, control=list(trace=0))$pars
  t_solnp = (proc.time() - t0)['elapsed']

  out_len = (draws - burn) / thin

  # original
  set.seed(100)
  t0     = proc.time()
  b_orig = get_bvec_cpp(draws, burn, thin, k, out_len,
                         alpha, zeta, tau, alpha_step,
                         tp$B_ols, sigsq, B_m, B_v,
                         price, beta0, xi, X)
  t_orig = (proc.time() - t0)['elapsed']

  # optimised
  set.seed(100)
  t0    = proc.time()
  b_opt = get_bvec_cpp_opt(draws, burn, thin, k, out_len,
                            alpha, zeta, tau, alpha_step,
                            tp$B_ols, sigsq, B_m, B_v,
                            price, beta0, xi, X)
  t_opt = (proc.time() - t0)['elapsed']

  spdup = t_orig / t_opt
  results[ci, ] = list(cr$name, nrow(rows), T_exp, t_x, t_solnp, t_orig, t_opt, spdup)

  cat(sprintf('%-8s %-12s %5d %5.3f %7.2f %7.1f %8.1f %7.1f %7.1f  (%5.2fx)\n',
              cr$name, cr$fomc_cvrd, nrow(rows), T_exp, cr$F_fwd*100,
              t_x, t_solnp, t_orig, t_opt, spdup))
}

# ---- Aggregate summary ------------------------------------------------------
cat(strrep('-', 80), '\n')
cat('\n=== AGGREGATE RESULTS ===\n\n')

tot_x     = sum(results$t_xmat)
tot_solnp = sum(results$t_solnp)
tot_orig  = sum(results$t_mcmc_orig)
tot_opt   = sum(results$t_mcmc_opt)

cat(sprintf('%-22s  %9s  %9s  %8s\n', 'Step', 'Original', 'Optimised', 'Speedup'))
cat(sprintf('%-22s  %9.1fs  %9.1fs  %8s\n',   'X matrix (all)', tot_x, tot_x, '(shared)'))
cat(sprintf('%-22s  %9.1fs  %9.1fs  %8s\n',   'solnp (all)',    tot_solnp, tot_solnp, '(shared)'))
cat(sprintf('%-22s  %9.1fs  %9.1fs  %7.2fx\n','MCMC (all)',     tot_orig, tot_opt, tot_orig/tot_opt))
cat(sprintf('%-22s  %9.1fs  %9.1fs  %7.2fx\n','TOTAL',
            tot_x+tot_solnp+tot_orig, tot_x+tot_solnp+tot_opt,
            (tot_x+tot_solnp+tot_orig)/(tot_x+tot_solnp+tot_opt)))

cat('\n=== SPEEDUP SCALING (n*k/(k+n) heuristic vs measured) ===\n')
for (ci in seq_len(nrow(results))) {
  n       = results$n_rows[ci]
  theory  = n * k / (k + n)
  cat(sprintf('  %-8s  n=%3d  theory=%5.1fx  measured=%5.2fx\n',
              results$contract[ci], n, theory, results$speedup[ci]))
}

cat(sprintf('\nTotal production run: orig=%.1fmin  opt=%.1fmin\n',
            (tot_x+tot_solnp+tot_orig)/60,
            (tot_x+tot_solnp+tot_opt)/60))
