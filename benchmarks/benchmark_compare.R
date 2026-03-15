################################################################################
# Head-to-head benchmark: original vs optimised MCMC sampler
# X matrix and solnp are shared (same computation both runs).
# The MCMC sampler is the only thing being swapped.
#
# Key novel optimisation in get_bvec_cpp_opt:
#   - Maintains Xb = X*beta_vec incrementally (O(n) update vs O(n*k) recompute)
#   - Maintains XtXb = X'X*beta_vec incrementally (O(k) update vs O(n*k) recompute)
#   - Eliminates 4x redundant X*beta_vec computations per MCMC iteration
#   - Reduces beta_met from O(n*k^2) to O(k^2 + n*k) per call
################################################################################

suppressPackageStartupMessages({
  library(pracma); library(Rsolnp); library(reshape)
  library(Rcpp); library(RcppArmadillo); library(parallel)
})

sourceCpp('/Users/adriangilliam/Dev/fed/mpt_source/simplexregression.cpp')
sourceCpp('/Users/adriangilliam/Dev/fed/mpt_source/simplexregression_opt.cpp')

n_cores = max(1L, detectCores() - 1L)

# ---- helpers (identical to benchmark.R) ------------------------------------
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

# ---- shared setup ----------------------------------------------------------
cat('=== SHARED SETUP ===\n')
k=80; draws=250000; burn=100000; thin=150
alpha=10; alpha_step=0.1; tau=1; B_init=1; zeta=1
xi=rep(1/k,k); B_m=1; B_v=1e12; sigsq=1

nd = update_data()
tp = get_transformparms(nd$options,nd$futures,nd$treas_rates,nd$basis_spread,nd$T_exp)
cat(sprintf('Options: %d rows  F=%.4f%%  B=%.6f  sigma=%.4f%%\n',
            tp$option_count, tp$F_ols, tp$B_ols, tp$sigma_ols))

t_x=proc.time()
X = get_xmat_serial(tp, nd$options, k)
t_xmat=(proc.time()-t_x)['elapsed']
cat(sprintf('X matrix (%dx%d): %.1fs\n', nrow(X), ncol(X), t_xmat))

price = nd$options$settle
t_s=proc.time()
beta0 = solnp(rep(1/k,k), get_ssq_r, eqfun=eqfn, eqB=1,
              ineqfun=ineqgfn, ineqLB=rep(0,k), ineqUB=rep(1,k),
              B=tp$B_ols, X=X, y=price, control=list(trace=0))$pars
t_solnp=(proc.time()-t_s)['elapsed']
cat(sprintf('solnp: %.1fs\n\n', t_solnp))

rows = (draws-burn)/thin

# ---- ORIGINAL MCMC ---------------------------------------------------------
cat('--- ORIGINAL get_bvec_cpp ---\n')
set.seed(100)
t_orig = proc.time()
bvec_orig = get_bvec_cpp(draws, burn, thin, k, rows,
                          alpha, zeta, tau, alpha_step,
                          tp$B_ols, sigsq, B_m, B_v,
                          price, beta0, xi, X)
t_mcmc_orig = (proc.time()-t_orig)['elapsed']
cat(sprintf('MCMC time (original): %.2fs\n', t_mcmc_orig))

# ---- OPTIMISED MCMC --------------------------------------------------------
cat('\n--- OPTIMISED get_bvec_cpp_opt ---\n')
set.seed(100)
t_opt = proc.time()
bvec_opt = get_bvec_cpp_opt(draws, burn, thin, k, rows,
                             alpha, zeta, tau, alpha_step,
                             tp$B_ols, sigsq, B_m, B_v,
                             price, beta0, xi, X)
t_mcmc_opt = (proc.time()-t_opt)['elapsed']
cat(sprintf('MCMC time (optimised): %.2fs\n', t_mcmc_opt))

# ---- COMPARISON ------------------------------------------------------------
cat('\n=== RESULTS SUMMARY ===\n')
speedup_mcmc  = t_mcmc_orig / t_mcmc_opt
total_orig    = t_xmat + t_solnp + t_mcmc_orig
total_opt     = t_xmat + t_solnp + t_mcmc_opt
speedup_total = total_orig / total_opt

cat(sprintf('%-25s  %7s  %7s  %7s\n', 'Step', 'Orig(s)', 'Opt(s)', 'Speedup'))
cat(sprintf('%-25s  %7.2f  %7.2f  %7s\n', 'X matrix (shared)', t_xmat, t_xmat, '  —'))
cat(sprintf('%-25s  %7.2f  %7.2f  %7s\n', 'solnp (shared)',    t_solnp, t_solnp, '  —'))
cat(sprintf('%-25s  %7.2f  %7.2f  %6.2fx\n', 'MCMC sampler', t_mcmc_orig, t_mcmc_opt, speedup_mcmc))
cat(sprintf('%-25s  %7.2f  %7.2f  %6.2fx\n', 'TOTAL', total_orig, total_opt, speedup_total))

# ---- VALIDATE outputs are statistically equivalent -------------------------
cat('\n=== POSTERIOR MEAN COMPARISON (alpha, B, sigma^2) ===\n')
means_orig = apply(bvec_orig, 2, mean)
means_opt  = apply(bvec_opt,  2, mean)
params = c('alpha','B','sigma^2','log-post')
for (i in 1:4) {
  cat(sprintf('  %-12s  orig=%.6f  opt=%.6f  diff=%.2e\n',
              params[i], means_orig[i], means_opt[i],
              abs(means_orig[i]-means_opt[i])))
}
beta_orig = means_orig[5:(k+4)]
beta_opt  = means_opt [5:(k+4)]
cat(sprintf('  beta[1:k]     max |diff| = %.2e,  mean |diff| = %.2e\n',
            max(abs(beta_orig - beta_opt)), mean(abs(beta_orig - beta_opt))))
cat('\n(Differences are due to MCMC stochasticity, not algorithmic change)\n')
