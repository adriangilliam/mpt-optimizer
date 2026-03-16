################################################################################
# MPT Benchmark Script - Optimized
# Same methodology, same data as baseline; optimizations:
#   1. get_xmat: parallelised over basis functions (mclapply)
#   2. MCMC: precomputed XtX/Xtp, incremental Xb/XtXb updates (C++ opt)
################################################################################

suppressPackageStartupMessages({
  library(pracma)
  library(Rsolnp)
  library(reshape)
  library(Rcpp)
  library(RcppArmadillo)
  library(parallel)
})

sourceCpp(file.path(dirname(sys.frame(1)$ofile), '..', 'optimized', 'simplexregression.cpp'))
sourceCpp(file.path(dirname(sys.frame(1)$ofile), '..', 'optimized', 'simplexregression_opt.cpp'))

n_cores = max(1L, detectCores() - 1L)
cat(sprintf('Parallel cores available: %d\n', n_cores))

# ---------------------------------------------------------------------------
# (Copy helpers from benchmark.R — identical, no changes)
# ---------------------------------------------------------------------------
get_rate = function(treas_rates, mat)
  approx(treas_rates$mat, treas_rates$rate, xout = mat, rule = 2)$y

bachelier_price = function(F, K, T, sig, flag) {
  d = (F - K) / (sig * sqrt(T))
  if (flag == 1) (F-K)*pnorm(d)  + sig*sqrt(T)*dnorm(d)
  else           (K-F)*pnorm(-d) + sig*sqrt(T)*dnorm(d)
}

norm_solvenlm = function(strikes, F, prices, B, flags, T_exp = 0.263) {
  obj = function(sig) {
    if (sig <= 0) return(1e10)
    phat = B * sapply(seq_along(strikes),
                      function(i) bachelier_price(F, strikes[i], T_exp, sig, flags[i]))
    sum((prices - phat)^2)
  }
  tryCatch(optimise(obj, interval = c(1e-5, 0.10))$minimum,
           error = function(e) 0.01)
}

update_data = function() {
  obs_date  = as.Date('2026-03-13')
  expy_date = as.Date('2026-06-17')
  days_exp  = as.numeric(expy_date - obs_date)
  T_exp     = days_exp / 365
  contract  = 'SRM26'
  F_true   = 0.0360
  sig_true = 0.0100
  B_true   = exp(-0.0372 * T_exp)
  sig_adj  = sig_true * sqrt(T_exp)
  K_min    = max(0.005, round((F_true - 3.5*sig_adj) / 0.0025) * 0.0025)
  K_max    = round((F_true + 3.5*sig_adj) / 0.0025) * 0.0025
  strikes  = seq(K_min, K_max, by = 0.0025)
  set.seed(42)
  rows = do.call(rbind, lapply(strikes, function(K) {
    cap_p   = B_true * bachelier_price(F_true, K, T_exp, sig_true, 1)
    floor_p = B_true * bachelier_price(F_true, K, T_exp, sig_true, 0)
    noise   = rnorm(2, 0, 1e-5)
    data.frame(date=obs_date, contract=contract, expy=expy_date, d_y=days_exp,
               rate_strike=K,
               settle=c(pmax(cap_p+noise[1],1e-8), pmax(floor_p+noise[2],1e-8)),
               put_call=c('P','C'), option_flag=c(1L,0L), stringsAsFactors=FALSE)
  }))
  rows = rows[rows$settle > 1e-6, ]
  futures = data.frame(date=obs_date, contract=contract,
                       rate_settle=F_true, mat=T_exp, stringsAsFactors=FALSE)
  treas_rates = data.frame(date=obs_date,
    mat  = c(1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30),
    rate = c(0.0375,0.0371,0.0372,0.0369,0.0370,
             0.0366,0.0373,0.0374,0.0387,0.0407,0.0428,0.0489,0.0490),
    stringsAsFactors=FALSE)
  list(options=rows, futures=futures, treas_rates=treas_rates,
       basis_spread=0, T_exp=T_exp)
}

f_p = function(x, parms)
  (parms[1]-x)*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*
  dnorm(x,mean=parms[4],sd=parms[5])

f_c = function(x, parms)
  (x-parms[1])*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*
  dnorm(x,mean=parms[4],sd=parms[5])

get_transformparms = function(options, futures, treas_rates, basis_spread, T_exp) {
  dys = unique(options$d_y); date = as.Date(unique(options$date))
  contract = unique(options$contract); expy_date = as.Date(unique(options$expy))
  days_to_expy = as.numeric(expy_date - date)
  lm.output = data.frame(d_y=dys, date=date, contract=contract,
    days_to_expy=days_to_expy, F_ols=NA, B_ols=NA, sigma_ols=NA,
    basis_spread=NA, est_type=NA, yield_curve=NA,
    option_count=0, option_overlap=0, stringsAsFactors=FALSE)
  lm.output$basis_spread = basis_spread
  x = cast(options[options$settle>0,], rate_strike~put_call, value='settle')
  x_overlap = 0; x_count = nrow(x); x_namelen = length(names(x))
  lm.output$yield_curve = ifelse(unique(options$date)==unique(treas_rates$date),
                                 'current','lagged')
  if (x_namelen == 3) { x = x[!is.na(x$C) & !is.na(x$P),]; x_overlap = nrow(x) }
  if (x_overlap < 2) {
    lm.output$F_ols = futures$rate_settle
    lm.output$B_ols = exp(-get_rate(treas_rates,futures$mat)*futures$mat)
    lm.output$est_type = 'data'
  } else {
    x$y = x$P - x$C; tmp.lm = lm(x$y ~ x$rate_strike)
    lm.output$B_ols = -tmp.lm$coefficients[2]
    lm.output$F_ols = tmp.lm$coefficients[1]/-tmp.lm$coefficients[2]
    lm.output$est_type = 'ols'
  }
  lm.output$option_count = x_count; lm.output$option_overlap = x_overlap
  if (!is.na(lm.output$F_ols))
    lm.output$sigma_ols = norm_solvenlm(options$rate_strike, lm.output$F_ols,
                                         options$settle, lm.output$B_ols,
                                         options$option_flag, T_exp)
  lm.output$F_ols = lm.output$F_ols * 100
  lm.output$sigma_ols = lm.output$sigma_ols * 100
  lm.output
}

# ---------------------------------------------------------------------------
# OPTIMISATION 1: parallel get_xmat (mclapply over basis functions)
# Same integrand functions f_p / f_c — no methodology change.
# ---------------------------------------------------------------------------
get_xmat_parallel = function(transformparms, options, k) {
  if (is.na(transformparms$F_ols)) return(NA)
  m      = transformparms$F_ols / 100
  sig    = 3 * transformparms$sigma_ols / 100
  s      = options$rate_strike
  option = options$option_flag
  s_len  = length(s)

  cl = makeCluster(n_cores)
  on.exit(stopCluster(cl), add=TRUE)
  clusterExport(cl, c('f_p','f_c','s','option','s_len','k','m','sig'), envir=environment())
  clusterEvalQ(cl, suppressPackageStartupMessages(library(pracma)))

  cols = parLapply(cl, 1:k, function(j) {
    col = numeric(s_len)
    for (i in 1:s_len) {
      parms = c(s[i], j, k, m, sig)
      col[i] = suppressMessages(
        if (option[i] == 0L) integral(f_p, -Inf, s[i], parms=parms, reltol=0)
        else                  integral(f_c,  s[i], Inf, parms=parms, reltol=0)
      )
    }
    col
  })
  do.call(cbind, cols)
}

get_ssq_r  = function(beta, B, X, y) sum((y - B * X %*% beta)^2)
eqfn       = function(beta, B, X, y) sum(beta)
ineqgfn    = function(beta, B, X, y) beta

# ---------------------------------------------------------------------------
# OPTIMIZED estimate_parms
# ---------------------------------------------------------------------------
estimate_parms_opt = function() {
  k          = 80
  draws      = 250000
  burn       = 100000
  thin       = 150
  alpha      = 10
  alpha_step = 0.1
  tau        = 1
  B          = 1
  zeta       = 1
  xi         = rep(1/k, k)
  B_m        = 1
  B_v        = 1e12
  sigsq      = 1

  new_data       = update_data()
  transformparms = get_transformparms(new_data$options, new_data$futures,
                                      new_data$treas_rates, new_data$basis_spread,
                                      new_data$T_exp)
  cat(sprintf('Options: %d rows, overlap: %d\n',
              transformparms$option_count, transformparms$option_overlap))

  # Opt 1: parallel X matrix
  cat('Building X matrix (parallel)...\n')
  t0    = proc.time()
  x_mat = get_xmat_parallel(transformparms, new_data$options, k)
  t_xmat = (proc.time() - t0)['elapsed']
  cat(sprintf('  X matrix (%dx%d): %.1fs\n', nrow(x_mat), ncol(x_mat), t_xmat))

  price = new_data$options$settle

  cat('Running solnp warm-start...\n')
  t1 = proc.time()
  beta_vec = solnp(rep(1/k, k), get_ssq_r,
                   eqfun=eqfn, eqB=1,
                   ineqfun=ineqgfn, ineqLB=rep(0,k), ineqUB=rep(1,k),
                   B=transformparms$B_ols, X=x_mat, y=price,
                   control=list(trace=0))$pars
  t_solnp = (proc.time() - t1)['elapsed']
  cat(sprintf('  solnp: %.1fs\n', t_solnp))

  rows = (draws - burn) / thin
  cat(sprintf('Running MCMC optimised (%d draws → %d samples)...\n', draws, rows))

  # Opt 2: optimised MCMC (precomputed XtX/Xtp, incremental Xb/XtXb)
  t2 = proc.time()
  tmp_bvec = get_bvec_cpp_opt(draws, burn, thin, k, rows,
                               alpha, zeta, tau, alpha_step,
                               transformparms$B_ols, sigsq, B_m, B_v,
                               price, beta_vec, xi, x_mat)
  t_mcmc = (proc.time() - t2)['elapsed']
  cat(sprintf('  MCMC: %.1fs\n', t_mcmc))

  total = t_xmat + t_solnp + t_mcmc
  cat(sprintf('\nTotal: %.1fs  (xmat=%.1f, solnp=%.1f, mcmc=%.1f)\n',
              total, t_xmat, t_solnp, t_mcmc))

  list(transformparms = transformparms,
       bvec_means    = apply(tmp_bvec, 2, mean),
       timing        = list(xmat=t_xmat, solnp=t_solnp, mcmc=t_mcmc, total=total))
}

cat('=== MPT OPTIMISED BENCHMARK (2026-03-13) ===\n')
t_wall = proc.time()
result = estimate_parms_opt()
cat(sprintf('Wall clock: %.1fs\n', (proc.time() - t_wall)['elapsed']))
