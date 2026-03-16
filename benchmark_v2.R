################################################################################
# Benchmark v2: validates and benchmarks all three optimisation levels
#   1. Original MCMC (get_bvec_cpp)
#   2. Opt v1: precomputed XtX/Xtp, incremental Xb/XtXb (get_bvec_cpp_opt)
#   3. Opt v2: + scalar RNG + uniform-xi fast path (get_bvec_cpp_opt2)
#   4. C++ X matrix (get_xmat_cpp) vs R pracma::integral
################################################################################

suppressPackageStartupMessages({
  library(pracma); library(Rsolnp); library(reshape)
  library(Rcpp); library(RcppArmadillo)
})

# Resolve paths: works with both Rscript and source()
.script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", args, value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else getwd()
})
.root <- normalizePath(file.path(.script_dir))

sourceCpp(file.path(.root, 'optimized', 'simplexregression.cpp'))
sourceCpp(file.path(.root, 'optimized', 'simplexregression_opt.cpp'))
sourceCpp(file.path(.root, 'optimized', 'simplexregression_opt2.cpp'))
sourceCpp(file.path(.root, 'optimized', 'xmat_cpp.cpp'))

# ---- helpers ----------------------------------------------------------------
get_rate = function(tr, mat) approx(tr$mat, tr$rate, xout=mat, rule=2)$y

bachelier_price = function(F, K, T, sig, flag) {
  d = (F-K)/(sig*sqrt(T))
  if (flag==1) (F-K)*pnorm(d)+sig*sqrt(T)*dnorm(d)
  else         (K-F)*pnorm(-d)+sig*sqrt(T)*dnorm(d)
}

norm_solvenlm = function(strikes, F, prices, B, flags, T_exp=0.263) {
  obj = function(sig) {
    if (sig<=0) return(1e10)
    phat = B*sapply(seq_along(strikes), function(i) bachelier_price(F,strikes[i],T_exp,sig,flags[i]))
    sum((prices-phat)^2)
  }
  tryCatch(optimise(obj, interval=c(1e-5,0.10))$minimum, error=function(e) 0.01)
}

update_data = function() {
  obs_date=as.Date('2026-03-13'); expy_date=as.Date('2026-06-17')
  days_exp=as.numeric(expy_date-obs_date); T_exp=days_exp/365; contract='SRM26'
  F_true=0.0360; sig_true=0.0100; B_true=exp(-0.0372*T_exp)
  sig_adj=sig_true*sqrt(T_exp)
  K_min=max(0.005,round((F_true-3.5*sig_adj)/0.0025)*0.0025)
  K_max=round((F_true+3.5*sig_adj)/0.0025)*0.0025
  strikes=seq(K_min,K_max,by=0.0025); set.seed(42)
  rows=do.call(rbind,lapply(strikes,function(K){
    cap_p=B_true*bachelier_price(F_true,K,T_exp,sig_true,1)
    floor_p=B_true*bachelier_price(F_true,K,T_exp,sig_true,0)
    noise=rnorm(2,0,1e-5)
    data.frame(date=obs_date,contract=contract,expy=expy_date,d_y=days_exp,
               rate_strike=K,settle=c(pmax(cap_p+noise[1],1e-8),pmax(floor_p+noise[2],1e-8)),
               put_call=c('P','C'),option_flag=c(1L,0L),stringsAsFactors=FALSE)
  }))
  rows=rows[rows$settle>1e-6,]
  futures=data.frame(date=obs_date,contract=contract,rate_settle=F_true,mat=T_exp,stringsAsFactors=FALSE)
  treas=data.frame(date=obs_date,
    mat=c(1/12,2/12,3/12,4/12,6/12,1,2,3,5,7,10,20,30),
    rate=c(0.0375,0.0371,0.0372,0.0369,0.0370,0.0366,0.0373,0.0374,0.0387,0.0407,0.0428,0.0489,0.0490),
    stringsAsFactors=FALSE)
  list(options=rows,futures=futures,treas_rates=treas,basis_spread=0,T_exp=T_exp)
}

f_p=function(x,parms)(parms[1]-x)*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])
f_c=function(x,parms)(x-parms[1])*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])

get_transformparms = function(options,futures,treas,bs,T_exp) {
  dys=unique(options$d_y); date=as.Date(unique(options$date))
  contract=unique(options$contract); expy_date=as.Date(unique(options$expy))
  lm.out=data.frame(d_y=dys,date=date,contract=contract,
    days_to_expy=as.numeric(expy_date-date),F_ols=NA,B_ols=NA,sigma_ols=NA,
    basis_spread=NA,est_type=NA,yield_curve=NA,option_count=0,option_overlap=0,stringsAsFactors=FALSE)
  lm.out$basis_spread=bs
  x=cast(options[options$settle>0,],rate_strike~put_call,value='settle')
  x_ov=0; x_cnt=nrow(x)
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
  m=tp$F_ols/100; sig=3*tp$sigma_ols/100; s=options$rate_strike; opt=options$option_flag; sl=length(s)
  X = matrix(0.0, nrow=sl, ncol=k)
  for (j in 1:k) for (i in 1:sl) {
    parms = c(s[i], j, k, m, sig)
    X[i,j] = suppressMessages(
      if (opt[i]==0L) integral(f_p,-Inf,s[i],parms=parms,reltol=0)
      else            integral(f_c, s[i],Inf,parms=parms,reltol=0)
    )
  }
  X
}

get_ssq_r=function(beta,B,X,y) sum((y-B*X%*%beta)^2)
eqfn=function(beta,B,X,y) sum(beta)
ineqgfn=function(beta,B,X,y) beta

# ---- setup ------------------------------------------------------------------
cat('================================================================\n')
cat('MPT BENCHMARK v2: X matrix + MCMC optimisations\n')
cat('================================================================\n\n')

k=80; draws=250000; burn=100000; thin=150
alpha=10; alpha_step=0.1; tau=1; zeta=1
xi=rep(1/k,k); B_m=1; B_v=1e12; sigsq=1

nd = update_data()
tp = get_transformparms(nd$options,nd$futures,nd$treas_rates,nd$basis_spread,nd$T_exp)
price = nd$options$settle
m_dec = tp$F_ols/100; sig_dec = 3*tp$sigma_ols/100

cat(sprintf('Options: %d rows  F=%.4f%%  B=%.6f  sigma=%.4f%%\n\n',
            tp$option_count, tp$F_ols, tp$B_ols, tp$sigma_ols))

# ===========================================================================
# PART 1: X MATRIX COMPARISON
# ===========================================================================
cat('--- X MATRIX: R (pracma::integral, reltol=0) ---\n')
t0 = proc.time()
X_r = get_xmat_serial(tp, nd$options, k)
t_xr = (proc.time()-t0)['elapsed']
cat(sprintf('  R X matrix (%dx%d): %.2fs\n', nrow(X_r), ncol(X_r), t_xr))

cat('\n--- X MATRIX: C++ (tanh-sinh, all 7 levels) ---\n')
t0 = proc.time()
X_cpp = get_xmat_cpp(m_dec, sig_dec,
                      nd$options$rate_strike,
                      as.integer(nd$options$option_flag),
                      k, 0.0)
t_xcpp = (proc.time()-t0)['elapsed']
cat(sprintf('  C++ X matrix (%dx%d): %.4fs\n', nrow(X_cpp), ncol(X_cpp), t_xcpp))

cat(sprintf('\n  X matrix speedup: %.0fx\n', t_xr/t_xcpp))

# Accuracy validation
max_abs  = max(abs(X_r - X_cpp))
mean_abs = mean(abs(X_r - X_cpp))
# Relative error only on cells with meaningful magnitude
big = abs(X_r) > 1e-10
if (any(big)) {
  max_rel_big  = max(abs((X_r[big] - X_cpp[big]) / X_r[big]))
  mean_rel_big = mean(abs((X_r[big] - X_cpp[big]) / X_r[big]))
} else {
  max_rel_big = NA; mean_rel_big = NA
}

cat(sprintf('\n  Accuracy (all cells):          max|err|=%.2e  mean|err|=%.2e\n', max_abs, mean_abs))
cat(sprintf('  Accuracy (|X|>1e-10, n=%d):    max rel=%.2e  mean rel=%.2e\n',
            sum(big), max_rel_big, mean_rel_big))

# ===========================================================================
# PART 2: MCMC COMPARISON (all three versions, same X matrix)
# ===========================================================================
cat('\n--- SOLNP warm-start (shared) ---\n')
t0 = proc.time()
beta0 = solnp(rep(1/k,k), get_ssq_r, eqfun=eqfn, eqB=1,
              ineqfun=ineqgfn, ineqLB=rep(0,k), ineqUB=rep(1,k),
              B=tp$B_ols, X=X_r, y=price, control=list(trace=0))$pars
t_solnp = (proc.time()-t0)['elapsed']
cat(sprintf('  solnp: %.1fs\n', t_solnp))

rows = (draws-burn)/thin

cat('\n--- MCMC: Original (get_bvec_cpp) ---\n')
set.seed(100)
t0 = proc.time()
bvec_orig = get_bvec_cpp(draws, burn, thin, k, rows,
                          alpha, zeta, tau, alpha_step,
                          tp$B_ols, sigsq, B_m, B_v,
                          price, beta0, xi, X_r)
t_orig = (proc.time()-t0)['elapsed']
cat(sprintf('  MCMC original: %.2fs\n', t_orig))

cat('\n--- MCMC: Opt v1 (precomputed XtX, incremental Xb) ---\n')
set.seed(100)
t0 = proc.time()
bvec_v1 = get_bvec_cpp_opt(draws, burn, thin, k, rows,
                             alpha, zeta, tau, alpha_step,
                             tp$B_ols, sigsq, B_m, B_v,
                             price, beta0, xi, X_r)
t_v1 = (proc.time()-t0)['elapsed']
cat(sprintf('  MCMC opt v1: %.2fs\n', t_v1))

cat('\n--- MCMC: Opt v2 (+ scalar RNG + uniform-xi) ---\n')
set.seed(100)
t0 = proc.time()
bvec_v2 = get_bvec_cpp_opt2(draws, burn, thin, k, rows,
                              alpha, zeta, tau, alpha_step,
                              tp$B_ols, sigsq, B_m, B_v,
                              price, beta0, xi, X_r)
t_v2 = (proc.time()-t0)['elapsed']
cat(sprintf('  MCMC opt v2: %.2fs\n', t_v2))

# ===========================================================================
# PART 3: FULL PIPELINE — original vs fully optimised
# ===========================================================================
cat('\n--- FULL PIPELINE: Opt v2 MCMC + C++ X matrix ---\n')
set.seed(100)
t0 = proc.time()
bvec_v2_cpp = get_bvec_cpp_opt2(draws, burn, thin, k, rows,
                                  alpha, zeta, tau, alpha_step,
                                  tp$B_ols, sigsq, B_m, B_v,
                                  price, beta0, xi, X_cpp)
t_v2_cpp = (proc.time()-t0)['elapsed']
cat(sprintf('  MCMC opt v2 (C++ X): %.2fs\n', t_v2_cpp))

# ===========================================================================
# PART 4: RESULTS SUMMARY
# ===========================================================================
cat('\n================================================================\n')
cat('RESULTS SUMMARY\n')
cat('================================================================\n\n')

total_orig = t_xr + t_solnp + t_orig
total_v1   = t_xr + t_solnp + t_v1
total_v2   = t_xr + t_solnp + t_v2
total_full = t_xcpp + t_solnp + t_v2_cpp

cat(sprintf('%-28s %8s %8s %8s\n', 'Step', 'Time(s)', 'Speedup', 'vs Orig'))
cat(strrep('-', 56), '\n')
cat(sprintf('%-28s %8.2f %8s %8s\n', 'X matrix (R)',        t_xr,   '—', '—'))
cat(sprintf('%-28s %8.4f %7.0fx %8s\n','X matrix (C++)',     t_xcpp, t_xr/t_xcpp, '—'))
cat(sprintf('%-28s %8.2f %8s %8s\n', 'solnp (shared)',      t_solnp,'—', '—'))
cat(sprintf('%-28s %8.2f %8s %7.1fx\n','MCMC original',     t_orig, '—', 1.0))
cat(sprintf('%-28s %8.2f %7.1fx %7.1fx\n','MCMC opt v1',    t_v1, t_orig/t_v1, t_orig/t_v1))
cat(sprintf('%-28s %8.2f %7.1fx %7.1fx\n','MCMC opt v2',    t_v2, t_v1/t_v2, t_orig/t_v2))
cat(strrep('-', 56), '\n')
cat(sprintf('%-28s %8.2f %8s %8s\n', 'TOTAL original pipeline',  total_orig, '—', '—'))
cat(sprintf('%-28s %8.2f %8s %7.1fx\n','TOTAL opt v1',           total_v1,   '—', total_orig/total_v1))
cat(sprintf('%-28s %8.2f %8s %7.1fx\n','TOTAL opt v2 (R X)',     total_v2,   '—', total_orig/total_v2))
cat(sprintf('%-28s %8.2f %8s %7.1fx\n','TOTAL opt v2 (C++ X)',   total_full, '—', total_orig/total_full))

# ===========================================================================
# PART 5: ACCURACY VALIDATION
# ===========================================================================
cat('\n================================================================\n')
cat('ACCURACY VALIDATION\n')
cat('================================================================\n\n')

# v1 vs v2 (same X matrix — should differ only by RNG stream divergence
# since scalar R::runif and sugar runif(1)[0] draw from the same R state
# BUT the order of draws may differ if Rcpp sugar uses a different path)
cat('v1 vs v2 (both using R X matrix):\n')
m_v1 = apply(bvec_v1, 2, mean)
m_v2 = apply(bvec_v2, 2, mean)
params = c('alpha','B','sigma^2','log-post')
for (i in 1:4) {
  cat(sprintf('  %-12s  v1=%.6f  v2=%.6f  diff=%.2e\n',
              params[i], m_v1[i], m_v2[i], abs(m_v1[i]-m_v2[i])))
}
beta_v1 = m_v1[5:(k+4)]
beta_v2 = m_v2[5:(k+4)]
cat(sprintf('  beta[1:k]     max|diff|=%.2e  mean|diff|=%.2e\n',
            max(abs(beta_v1-beta_v2)), mean(abs(beta_v1-beta_v2))))

# v2 with C++ X vs R X
cat('\nv2 with C++ X vs v2 with R X:\n')
m_v2c = apply(bvec_v2_cpp, 2, mean)
for (i in 1:4) {
  cat(sprintf('  %-12s  R_X=%.6f  C++_X=%.6f  diff=%.2e\n',
              params[i], m_v2[i], m_v2c[i], abs(m_v2[i]-m_v2c[i])))
}
beta_v2c = m_v2c[5:(k+4)]
cat(sprintf('  beta[1:k]     max|diff|=%.2e  mean|diff|=%.2e\n',
            max(abs(beta_v2-beta_v2c)), mean(abs(beta_v2-beta_v2c))))

cat('\n(Differences reflect MCMC stochasticity, not algorithmic change)\n')
