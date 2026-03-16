################################################################################
# Stress-test benchmark: n × draws scaling grid
#
# Tests two dimensions simultaneously:
#   n     — number of option rows, controlled by varying sigma
#             sigma=1%  →  n≈30   (realistic baseline)
#             sigma=3%  →  n≈70   (wider chain)
#             sigma=5%  →  n≈98   (production-scale wide)
#             sigma=7%  →  n≈126  (stress: hundreds of options)
#
#   draws — MCMC draw count
#             250 000  (standard)
#             1 000 000 (4× — tests linear draw-scaling)
#
# X matrix is computed ONCE per sigma level and reused across draws levels.
# burn = 40% of draws; thin chosen so each run produces ~1 000 posterior samples.
# set.seed(100) before every MCMC call for reproducibility.
#
# Expected results:
#   (1) Speedup grows with n, tracking the n*k/(k+n) heuristic
#   (2) Speedup is constant across draws (both samplers scale O(draws))
#   (3) At n≈100, measured speedup approaches ~10x in the MCMC step
#
# Observation date: 2026-03-15  |  Contract: Jun 2026 SRM26  |  k=80
################################################################################

suppressPackageStartupMessages({
  library(pracma); library(Rsolnp); library(reshape)
  library(Rcpp);   library(RcppArmadillo)
})

# Locate the optimized/ directory relative to this script, works with Rscript
get_script_dir = function() {
  args = commandArgs(trailingOnly = FALSE)
  f = grep("--file=", args, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("--file=", "", f[1])))
  else tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
}
opt_dir = file.path(get_script_dir(), '..', 'optimized')

sourceCpp(file.path(opt_dir, 'simplexregression.cpp'))
sourceCpp(file.path(opt_dir, 'simplexregression_opt.cpp'))

# ---- helpers ----------------------------------------------------------------
get_rate     = function(tr, mat) approx(tr$mat, tr$rate, xout=mat, rule=2)$y

bachelier_price = function(F, K, T, sig, flag) {
  d = (F - K) / (sig * sqrt(T))
  if (flag == 1) (F-K)*pnorm(d)  + sig*sqrt(T)*dnorm(d)
  else           (K-F)*pnorm(-d) + sig*sqrt(T)*dnorm(d)
}

norm_solvenlm = function(strikes, F, prices, B, flags, T_exp) {
  obj = function(sig) {
    if (sig <= 0) return(1e10)
    phat = B * sapply(seq_along(strikes), function(i)
                      bachelier_price(F, strikes[i], T_exp, sig, flags[i]))
    sum((prices - phat)^2)
  }
  tryCatch(optimise(obj, interval=c(1e-5, 0.20))$minimum, error=function(e) sig)
}

f_p = function(x,parms) (parms[1]-x)*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])
f_c = function(x,parms) (x-parms[1])*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])

get_transformparms = function(options, futures, treas, bs, T_exp) {
  dys=unique(options$d_y); date=as.Date(unique(options$date))
  contract=unique(options$contract); expy_date=as.Date(unique(options$expy))
  lm.out=data.frame(d_y=dys,date=date,contract=contract,
    days_to_expy=as.numeric(expy_date-date),F_ols=NA,B_ols=NA,sigma_ols=NA,
    basis_spread=NA,est_type=NA,yield_curve=NA,option_count=0L,option_overlap=0L,
    stringsAsFactors=FALSE)
  lm.out$basis_spread=bs
  x=cast(options[options$settle>0,],rate_strike~put_call,value='settle')
  x_ov=0L; x_cnt=nrow(x)
  lm.out$yield_curve=ifelse(unique(options$date)==unique(treas$date),'current','lagged')
  if(length(names(x))==3){x=x[!is.na(x$C)&!is.na(x$P),];x_ov=nrow(x)}
  if(x_ov<2){
    lm.out$F_ols=futures$rate_settle
    lm.out$B_ols=exp(-get_rate(treas,futures$mat)*futures$mat)
    lm.out$est_type='data'
  } else {
    x$y=x$P-x$C; tmp=lm(x$y~x$rate_strike)
    lm.out$B_ols=-tmp$coefficients[2]
    lm.out$F_ols=tmp$coefficients[1]/-tmp$coefficients[2]
    lm.out$est_type='ols'
  }
  lm.out$option_count=x_cnt; lm.out$option_overlap=x_ov
  if(!is.na(lm.out$F_ols))
    lm.out$sigma_ols=norm_solvenlm(options$rate_strike,lm.out$F_ols,options$settle,
                                    lm.out$B_ols,options$option_flag,T_exp)
  lm.out$F_ols=lm.out$F_ols*100; lm.out$sigma_ols=lm.out$sigma_ols*100
  lm.out
}

get_xmat_serial = function(tp, options, k) {
  m=tp$F_ols/100; sig=3*tp$sigma_ols/100
  s=options$rate_strike; opt=options$option_flag; sl=length(s)
  X=matrix(0.0,nrow=sl,ncol=k)
  for(j in 1:k) for(i in 1:sl) {
    parms=c(s[i],j,k,m,sig)
    X[i,j]=suppressMessages(
      if(opt[i]==0L) integral(f_p,-Inf,s[i],parms=parms,reltol=0)
      else           integral(f_c, s[i], Inf,parms=parms,reltol=0))
  }
  X
}

get_ssq_r = function(beta,B,X,y) sum((y-B*X%*%beta)^2)
eqfn      = function(beta,B,X,y) sum(beta)
ineqgfn   = function(beta,B,X,y) beta

# ---- fixed contract and yield curve ----------------------------------------
obs_date = as.Date('2026-03-15')
expy     = as.Date('2026-06-17')
days_exp = as.numeric(expy - obs_date)   # 94 days
T_exp    = days_exp / 365
F_fwd    = 0.0365

treas = data.frame(
  date = obs_date,
  mat  = c(1/12,2/12,3/12,4/12,6/12,1,2,3,5,7,10,20,30),
  rate = c(0.0375,0.0371,0.0372,0.0369,0.0370,0.0366,0.0373,0.0374,0.0387,0.0407,0.0428,0.0489,0.0490),
  stringsAsFactors=FALSE
)
B_true = exp(-get_rate(treas, T_exp) * T_exp)

# ---- stress-test dimensions ------------------------------------------------
# sigma controls n: wider vol → more strikes within ±3.5*sigma*sqrt(T)
sigma_levels = c(0.010, 0.030, 0.050, 0.070)   # 1%, 3%, 5%, 7%
draws_levels = c(250000L, 1000000L)

k     = 80L
alpha = 10; alpha_step = 0.1; tau = 1; zeta = 1
xi    = rep(1/k, k); B_m = 1; B_v = 1e12; sigsq = 1

# ---- storage ----------------------------------------------------------------
results = data.frame()

cat('=== STRESS-TEST BENCHMARK: n × draws SCALING GRID ===\n')
cat(sprintf('Contract: SRM26  |  obs: %s  |  F=%.2f%%  |  k=%d\n\n',
            obs_date, F_fwd*100, k))

cat(sprintf('%-6s %-8s %5s %9s %7s %7s %7s %7s\n',
            'sigma','draws','n','Xmat(s)','Orig(s)','Opt(s)','Speedup','Theory'))
cat(strrep('-', 70), '\n')

for (sig in sigma_levels) {

  # generate options for this sigma level
  sig_adj = sig * sqrt(T_exp)
  rng     = 3.5 * sig_adj
  K_min   = max(0.005, round((F_fwd - rng) / 0.0025) * 0.0025)
  K_max   =            round((F_fwd + rng) / 0.0025) * 0.0025
  strikes = seq(K_min, K_max, by=0.0025)

  set.seed(42)
  rows = do.call(rbind, lapply(strikes, function(K) {
    cap_p   = B_true * bachelier_price(F_fwd, K, T_exp, sig, 1)
    floor_p = B_true * bachelier_price(F_fwd, K, T_exp, sig, 0)
    noise   = rnorm(2, 0, 1e-5)
    data.frame(date=obs_date, contract='SRM26', expy=expy, d_y=days_exp,
               rate_strike=K,
               settle=c(max(cap_p+noise[1],1e-8), max(floor_p+noise[2],1e-8)),
               put_call=c('P','C'), option_flag=c(1L,0L),
               stringsAsFactors=FALSE)
  }))
  rows   = rows[rows$settle > 1e-6, ]
  n_rows = nrow(rows)
  price  = rows$settle
  futures = data.frame(date=obs_date,contract='SRM26',rate_settle=F_fwd,mat=T_exp,
                       stringsAsFactors=FALSE)
  tp = get_transformparms(rows, futures, treas, 0, T_exp)

  # X matrix — computed once per sigma level
  t0  = proc.time()
  X   = get_xmat_serial(tp, rows, k)
  t_x = (proc.time()-t0)['elapsed']

  # solnp warm-start — computed once per sigma level
  beta0 = solnp(rep(1/k,k), get_ssq_r, eqfun=eqfn, eqB=1,
                ineqfun=ineqgfn, ineqLB=rep(0,k), ineqUB=rep(1,k),
                B=tp$B_ols, X=X, y=price, control=list(trace=0))$pars

  theory = n_rows * k / (k + n_rows)

  for (draws in draws_levels) {
    burn    = as.integer(0.4 * draws)
    out_len = as.integer((draws - burn) / max(1L, as.integer((draws - burn) / 1000L)))
    thin    = as.integer((draws - burn) / out_len)

    # original
    set.seed(100)
    t0     = proc.time()
    b_orig = get_bvec_cpp(draws, burn, thin, k, out_len,
                           alpha, zeta, tau, alpha_step,
                           tp$B_ols, sigsq, B_m, B_v,
                           price, beta0, xi, X)
    t_orig = (proc.time()-t0)['elapsed']

    # optimised
    set.seed(100)
    t0    = proc.time()
    b_opt = get_bvec_cpp_opt(draws, burn, thin, k, out_len,
                              alpha, zeta, tau, alpha_step,
                              tp$B_ols, sigsq, B_m, B_v,
                              price, beta0, xi, X)
    t_opt = (proc.time()-t0)['elapsed']

    spdup = t_orig / t_opt

    # posterior mean max diff (sanity check)
    m_orig = apply(b_orig, 2, mean)
    m_opt  = apply(b_opt,  2, mean)
    max_beta_diff = max(abs(m_orig[5:(k+4)] - m_opt[5:(k+4)]))

    cat(sprintf('  %4.0f%%  %7s %5d %9.1f %7.1f %7.1f %6.2fx  [theory %5.1fx | max_beta_diff %.1e]\n',
                sig*100,
                formatC(draws, format='d', big.mark=','),
                n_rows, t_x, t_orig, t_opt, spdup, theory, max_beta_diff))

    results = rbind(results, data.frame(
      sigma=sig, draws=draws, n=n_rows, t_xmat=t_x,
      t_orig=t_orig, t_opt=t_opt, speedup=spdup, theory=theory,
      stringsAsFactors=FALSE
    ))
  }
  cat('\n')
}

cat(strrep('=', 70), '\n')

# ---- n-scaling summary (at draws=250k) -------------------------------------
cat('\n--- n-SCALING (draws=250k) ---\n')
d1 = results[results$draws==250000, ]
cat(sprintf('%-5s %5s %8s %8s %8s %8s\n',
            'sigma','n','Orig(s)','Opt(s)','Measured','Theory'))
for (i in seq_len(nrow(d1)))
  cat(sprintf(' %4.0f%% %5d %8.1f %8.1f %7.2fx  %7.1fx\n',
              d1$sigma[i]*100, d1$n[i], d1$t_orig[i], d1$t_opt[i],
              d1$speedup[i], d1$theory[i]))

# ---- draws-scaling summary (at each sigma) ----------------------------------
cat('\n--- DRAWS-SCALING (speedup should be constant across draws) ---\n')
cat(sprintf('%-5s %5s %8s %8s %8s\n', 'sigma','n','draws','Speedup','Ratio-per-draw'))
for (sig in sigma_levels) {
  sub = results[results$sigma==sig, ]
  base_t = sub$t_orig[1] / sub$draws[1]
  for (i in seq_len(nrow(sub))) {
    cat(sprintf(' %4.0f%% %5d %8s %7.2fx  %.2e s/draw\n',
                sig*100, sub$n[i],
                formatC(sub$draws[i], format='d', big.mark=','),
                sub$speedup[i],
                sub$t_orig[i] / sub$draws[i]))
  }
}

# ---- aggregate ---------------------------------------------------------------
cat(sprintf('\n--- BLAS efficiency factor (theory/measured at 250k draws) ---\n'))
cat('  (gap between theory and measured explained by BLAS vectorisation\n')
cat('   benefiting the original\'s O(n*k^2) more than the optimised)\n\n')
d1 = results[results$draws==250000, ]
for (i in seq_len(nrow(d1)))
  cat(sprintf('  n=%3d  theory=%5.1fx  measured=%5.2fx  BLAS-factor=%.1fx\n',
              d1$n[i], d1$theory[i], d1$speedup[i], d1$theory[i]/d1$speedup[i]))
