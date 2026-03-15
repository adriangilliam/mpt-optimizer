###########################################
# Market Probability Tracker R Code       #
# Author: Brian Robertson		  #
#	  Federal Reserve Bank of Atlanta #
#	  brian.robertson@atl.frb.org	  #
# --------------------------------------- #
# R libraries that may be helpful:	  #
# tidyverse				  #
# lubridate				  #
# pracma				  #
# Rsolnp				  #
# foreach				  #
# parallel				  #
# doParallel				  #
# Rcpp					  #
# RcppArmadillo				  #
# --------------------------------------- #
# ALL CODE IS PROVIDED WITHOUT WARRANTY   #
# Copyright (c) 2016, Federal Reserve     #
# Bank of Atlanta.  All rights reserved.  #
###########################################


update_data=function(){
  #returns a list with each element containing data for a specific day:
  #-a single futures contract
  #-options for that futures contract
  #-Treasury rates
  #-basis spread, if applicable
}

#returns the transform parameters needed for x_mat, chiefly:
#-futures rate (ols-implied or market)
#-discount factor (ols-implied or market)
#-implied vol (ols-implied or market)
#-basis spread (if applicable)
get_transformparms=function(options,futures,treas_rates,basis_spread){
  dys=unique(options$d_y)
  date=as.Date(unique(options$date))
  contract=unique(options$contract)
  expy_date=as.Date(unique(options$expy))
  days_to_expy=as.numeric(expy_date-date)
  lm.output=data.frame(d_y=dys,date=date,contract=contract,days_to_expy=days_to_expy,F_ols=NA,B_ols=NA,sigma_ols=NA,basis_spread=NA,est_type=NA,yield_curve=NA,option_count=0,option_overlap=0,stringsAsFactors=FALSE)
  lm.output$basis_spread=basis_spread
  x=cast(options[options$settle>0,],rate_strike~put_call,value='settle')
  x_overlap=0
  x_count=length(x[,1])
  x_namelen=length(names(x))
  if(unique(options$date)==unique(treas_rates$date)){
    lm.output$yield_curve='current'
  } else {
    lm.output$yield_curve='lagged'
  }
  if(x_namelen==3){
    x=x[!is.na(x$C) & !is.na(x$P),]
    x_overlap=length(x[,1])
  }
  if(x_overlap<2){
    #avail_data=sum(dys %in% futures$d_y) + sum(options$date[1] %in% treas_rates$date)
    #if(avail_data==2){
      lm.output$F_ols=futures$rate_settle
      lm.output$B_ols=exp(-get_rate(treas_rates,futures$mat)*futures$mat)
      lm.output$est_type='data'
    #}
  }
  else{
    x$y=x$P-x$C
    tmp.lm=lm(x$y~x$rate_strike)
    lm.output$B_ols=-tmp.lm$coefficients[2]
    lm.output$F_ols=tmp.lm$coefficients[1]/-tmp.lm$coefficients[2]
    lm.output$est_type='ols'
  }
  lm.output$option_count=x_count
  lm.output$option_overlap=x_overlap
  if(!is.na(lm.output$F_ols)){
    lm.output$sigma_ols=norm_solvenlm(options$rate_strike,lm.output$F_ols,options$settle,lm.output$B_ols,options$option_flag)
  }
  lm.output$F_ols=lm.output$F_ols*100
  lm.output$sigma_ols=lm.output$sigma_ols*100
  return(lm.output)
}

#Helper function: calculates the value of the basis density.  Note: assumes prices~N(.).
#Also note: f_p is used for puts, f_c is used for calls.
f_p=function(x,parms) (parms[1]-x)*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])
f_c=function(x,parms) (x-parms[1])*dbeta(pnorm(x,mean=parms[4],sd=parms[5]),parms[2],(parms[3]-parms[2]+1))*dnorm(x,mean=parms[4],sd=parms[5])


#Takes the options associated with a specifc futures
#contract and their transform parameters and calculates
#the n x k matrix X of basis densities.
get_xmat=function(transformparms,options,k){
  if(is.na(transformparms$F_ols)){
    X=NA
  }
  else{
    m=transformparms$F_ols/100
    sig=3*transformparms$sigma_ols/100
    s=options$rate_strike
    option=options$option_flag
    s_len=length(s)
    X=matrix(rep(0,(s_len*k)),nrow=s_len,ncol=k)
    for(j in 1:k){
      for(i in 1:s_len){
        parms=c(s[i],j,k,m,sig)
        if (option[i]==0){
          X[i,j]=integral(f_p,-Inf,s[i],parms=parms,reltol=0)
        }
        else{
          X[i,j]=integral(f_c,s[i],Inf,parms=parms,reltol=0)
        }
      }
    }
  }
  return(X)
}

#Helper function: returns the sum of squared errors of y_hat=B*X*Beta
get_ssq=function(beta,B,X,y){
  y_hat=B*X%*%beta
  ssq=sum((y-y_hat)^2)
  return(ssq)
}

#Helper function: equality constraint for the nonlinear solver
eqfn=function(beta,B,X,y){
  return(sum(beta))
}

#Helper function: inequality constraint for the nonlinear solver
ineqgfn=function(beta,B,X,y){
  return(beta)
}

#Gets data, initializes the X matrix, and runs get_bvec_cpp, which returns a 
#rows x 84 matrix of parameter thinned draws, which are averaged over to get the 
#parameter estimates.
estimate_parms=function(parms){
  sourceCpp('simplexregression.cpp')
  #sampler parameters
  parms=list(k=80,
             draws=250000,
             burn=100000,
             thin=150,
             alpha=10,
             alpha_step=0.1,
             tau=1,
             B=1,
             zeta=1,
             xi=rep(1/k,k),
             B_m=1,
             B_v=1e12,
             sigsq=1)
  new_data=update_data()
  transformparms=get_transformparms(new_data$options,new_data$futures,new_data$treas_rates,new_data$basis_spread)
  x_mat=get_xmat(transformparms,new_data$options,parms$k)
  price=i$options$settle
  rows=(parms$draws-parms$burn)/parms$thin
  beta_vec=solnp(rep(1/parms$k,parms$k),get_ssq,eqfun=eqfn,eqB=c(1),ineqfun=ineqfn,ineqLB=rep(0,parms$k),ineqUB=rep(1,parms$k),B=parms$B,X=xmat,y=price,control=list(trace=0))$pars
  tmp_bvec=get_bvec_cpp(parms$draws,parms$burn,parms$thin,parms$k,rows,parms$alpha,parms$zeta,parms$tau,parms$alpha_step,parms$B,parms$sigsq,parms$B_m,parms$B_v,price,beta_vec,parms$xi,xmat)
  tmp_parms=data.frame(c(transformparms,apply(tmp_bvec,2,mean)),stringsAsFactors=FALSE)
  return(tmp_parms)
}

#Helper function: given xx (in basis points) and model parms, 
#returns the probability density esimate y 
f_basis=function(xx,model_parms){
  bnames=paste0('bvec.',1:parms$k)
  xx=xx/100+model_parms$basis_spread/100
  a_vec=rep(1:parms$k)
  b_vec=parms$k-a_vec+1
  mu=as.numeric(model_parms$F_ols)/100
  sigma=3*as.numeric(model_parms$sigma_ols)/100
  b_hat=as.numeric(model_parms[bnames])
  y=t(mapply(function(x) dbeta(pnorm(x,mean=mu,sd=sigma),a_vec,b_vec)*dnorm(x,mean=mu,sd=sigma),xx))
  return(as.numeric(y%*%b_hat))
}


#Helper function: formats axis for printing
fmt=function() function(x) format(x,nsmall=2,scientific=FALSE)

#Helper function: returns the probability between x1 and x2 
#implied by the area underneath the estimated distribution
f_basis_integrate=function(x1,x2,model_parms){
  return(integrate(f_b,x1,x2,model_parms=model_parms,abs.tol=0)$value)
}

#Prints the probability density from the estimated modelparms
print_dist=function(model_parms,xmin,xmax,chart_title=NULL){
  print('print_dist')
  contract=model_parms$contract
  xmin=floor(xmin/25)*25
  xmax=ceiling(xmax/25)*25
  p=ggplot(data.frame(x=c(xmin,xmax),Date=model_parms$date),aes(x))+
    stat_function(fun=f_basis,args=list(model_parms=model_parms),n=500,aes(color=Date))+
    theme_bw()+
    theme(legend.position=c(.25,.75),
          legend.key=element_blank(),
          legend.background=element_blank())+
    scale_y_continuous(labels=fmt())+
    geom_vline(xintercept=seq(from=xmin,to=xmax,by=25)[-1],linetype='dotted')+
    labs(y='Probability Density',
         x='Basis Points')
  if(!is.null(chart_title))
    p=p+labs(title=paste('Contract: ',contract))
  return(p)
}
