///////////////////////////////////////////
// Market Probability Tracker R Code     //
// Author: Brian Robertson                //
//        Federal Reserve Bank of Atlanta//
//        brian.robertson@atl.frb.org    //
// Adapted from Mark Fisher's "Simplex   //
// Regression"                           //
// -----------------------------------  //
// ALL CODE IS PROVIDED WITHOUT WARRANTY //
// Copyright (c) 2016, Federal Reserve   //
// Bank of Atlanta.  All rights reserved.//
///////////////////////////////////////////

#include <iostream>
#include <math.h>
#include <RcppArmadillo.h>
#include <Rmath.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;


// Helper function: calculates the loggamma
// of a column vector x
colvec lgamma_vec(colvec x){
  for(int i=0; i<x.n_rows; i++)
    x[i]=lgamma(x[i]);
  return x;
}

// Helper function: elementwise exponentiates
// two vectors x and y as x^y
colvec expon_vec(colvec x, colvec y){
  for(int i=0; i<x.n_rows; i++)
    x[i]=pow(x[i],y[i]);
  return x;
}

// From 5.7: we do a change of variables z=log(a),
// where the likelihood for z is p(B|z)=p(B|a)|a=exp(z).
// For numerical tractability p(B|a) is calculated as
// the loglikelihood.
double z_like(double z,colvec xi,colvec beta_vec){
  double alpha, like;
  alpha=exp(z);
  like=lgamma(alpha)-sum(lgamma_vec(alpha*xi))+sum(log(expon_vec(beta_vec,(alpha*xi-1))));
  return like;
}

// From 4.2: prior for beta p(B|a), calculated
// as the log prior for numerical tractability.
double beta_prior(double alpha, colvec xi, colvec beta_vec){
  double prior;
  prior=lgamma(alpha)-sum(lgamma_vec(alpha*xi))+sum(log(expon_vec(beta_vec,(alpha*xi-1))));
  return prior;
}

// From 4.3: Use p(z)=Logistic(z|zeta,tau) since a=exp(z).
// Note: parameters are initialized with respect to alpha,
// requiring the log(zeta) transformation.
double z_prior(double z,double zeta,double tau){
  double prior;
  prior=pow((2*tau*(1+cosh((z-log(zeta))/tau))),-1);
  prior=log(prior);
  return prior;
}

// Helper function: posterior=prior*likelihood.  Note: z=log(alpha),
// and z_like and z_prior return log probabilites.
double z_posterior(double z, colvec xi, colvec beta_vec, double zeta, double tau){
  double like, prior;
  like=z_like(z,xi,beta_vec);
  prior=z_prior(z,zeta,tau);
  return(like+prior);
}

// From 5.8 and 5.9: Metropolis-Hastings sampler for alpha
// transformed to z=log(alpha).  Generate proposal of step
// size s as z1~N(z,s^2).  Accept if p(z0,z1)>u~uniform(0,1).
double z_met(double alpha, colvec beta_vec, colvec xi, double zeta, double tau, double alpha_step){
  double v0, v1;
  double z0, z1;
  double ratio, test;
  z0=log(alpha);
  z1=rlogis(1,z0,tau*alpha_step)[0];
  v0=z_posterior(z0,xi,beta_vec,zeta,tau);
  v1=z_posterior(z1,xi,beta_vec,zeta,tau);
  ratio=v1-v0;
  test=log(runif(1)[0]);
  if (ratio>=test) alpha=exp(z1);
  return(alpha);
}

// Helper function: returns the position of
// the largest element of vector x.
int find_max_elem(colvec x){
  int j=1;
  double max_elm=x[0];
  for (int i=1; i<x.n_rows; i++){
    if (x[i]>max_elm){
      j=i;
      max_elm=x[i];
    }
  }
  return j;
}

// Helper function: generates a random draw from the truncated
// normal distribution with endpoints [a,b] using the inverse
// normal CDF.
double trunc_norm_invcdf(double a, double b, double m, double sd){
  double p, cdf_a,cdf_b;
  p=runif(1)[0];
  cdf_a=R::pnorm(a,m,sd,1,0);
  cdf_b=R::pnorm(b,m,sd,1,0);
  return R::qnorm(cdf_a+p*(cdf_b-cdf_a),m,sd,1,0);
}

// From 5.10 through 5.19: Metropolis-Hastings sampler for the vector of
// betas where sum(beta_vec)=1 and matrix X is rank deficient. In order to
// give each element j enough parameter space to sample, we remove element k
// with the largest value during each sweep, leaving b_j=1-beta_j-beta_k.
// Proposals are made from the normal distribution truncated at [0,b_j] with
// a mean and sd shown in 5.15a and 5.15b.  Note that the sampler reduces to
// a Gibbs sampler when the prior for beta is flat.
colvec beta_met(mat X, colvec beta_vec, colvec q, colvec price, double B, double sigsq){
  uword k_index;
  double m,b,v,prop,ratio;
  colvec tmp,X_kj,X_kindex;
  mat X_k;
  beta_vec.max(k_index);
  X_kindex=X.col(k_index);
  X_k=X.each_col()-X.col(k_index);
  for(int j=0; j<X.n_cols; j++){
      if(j!=(int)k_index){
        X_kj=X_k.col(j);
        b=beta_vec[j]+beta_vec[k_index];
        m=(beta_vec[j]+(X_kj.t()*price/B-X_kj.t()*X_kindex-X_kj.t()*X_k*beta_vec)/(X_kj.t()*X_kj))[0];
        tmp=X_kj.t()*X_kj;
        v=sigsq/(B*B*tmp[0]);
        prop=trunc_norm_invcdf(0,b,m,sqrt(v));
        ratio=pow((prop/beta_vec[j]),q[j])*pow((b-prop)/beta_vec[k_index],q[k_index]);
        if(ratio>=runif(1)[0]){
          beta_vec[j]=prop;
          beta_vec[k_index]=b-prop;
        }
      }
  }
  return(beta_vec);
}

// Helper function: returns the sum of squared errors between
// the observed prices y and the estimated prices B*X*beta_vec.
double get_ssq(colvec beta_vec, double B,mat X,colvec price){
  colvec y_hat;
  double ssq;
  y_hat=B*X*beta_vec;
  ssq=sum(pow(price-y_hat,2));
  return ssq;
}

// Helper function: generates random samples from
// the scaled inverse-chi-squared distribution.
double rScaledInvChiSq(double nu, double tau){
  double tmp;
  tmp=rgamma(1,nu/2.0,1/((nu*tau)/2.0))[0];
  return(1/tmp);
}

// From 5.4 and 5.5: Gibbs sampler for the discount factor B (notated as
// lambda in the paper).  Proposals are drawn from a normal distribution
// truncated at 0.
double b_gibbs(colvec price, mat X, colvec beta_vec, double sigsq, double B_m, double B_v){
  colvec Xb,Xb2;
  double md,vd,b_mean,b_sd;
  Xb=X*beta_vec;
  Xb2=Xb.t()*Xb;
  md=((price.t()*Xb)/Xb2)[0];
  vd=sigsq/Xb2[0];
  b_mean=B_m*vd/(B_v+vd)+md*B_v/(B_v+vd);
  b_sd=sqrt(B_v*vd/(B_v+vd));
  return(trunc_norm_invcdf(0,1e10,b_mean,b_sd));
}

// From 5.1: log-posterior probability.  Note: z=log(a) results in p(a)
// coming from a log distribution (in our case, logistic).  Also note:
// zeta is expressed with respect to alpha, not z.
double logposterior(double alpha, colvec beta_vec, double B, double sigsq, double zeta,
                    double tau, colvec xi, colvec price, mat X, double B_m, double B_v){
  double sigsq_lp,B_lp,beta_lp,alpha_lp;
  sigsq_lp=-0.5*(price.n_rows)*log(sigsq)-(1/(2*sigsq))*get_ssq(beta_vec,B,X,price);
  B_lp=-0.5*(pow((B-B_m),2))/B_v;
  beta_lp=beta_prior(alpha,xi,beta_vec);
  alpha_lp=(1/tau-1)*log(alpha)-2*log(pow(alpha,1/tau)+pow(zeta,1/tau));
  return(sigsq_lp+B_lp+beta_lp+alpha_lp);
}

// Generates Markov chains of length out_length.  Returns an
// [out_length x (k+4)] matrix: cols 1-4 are alpha, B, sigsq,
// logposterior; cols 5:(k+4) are the beta estimates.
// [[Rcpp::export]]
mat get_bvec_cpp(int draws,int burn, int thin, int k, int out_length, double alpha, double zeta, double tau, double alpha_step,
                 double B, double sigsq, double B_m, double B_v,
                 colvec price, colvec beta_vec, colvec xi, mat X){
  double lp;
  int ii=0;
  colvec q;
  mat output_mat(out_length,(k+4));
  for (int i=0; i<draws; i++){
    alpha=z_met(alpha, beta_vec, xi, zeta, tau, alpha_step);
    q=alpha*xi-1;
    beta_vec=beta_met(X,beta_vec,q,price,B,sigsq);
    sigsq=rScaledInvChiSq(price.n_rows,get_ssq(beta_vec,B,X,price)/price.n_rows);
    B=b_gibbs(price,X,beta_vec,sigsq,B_m,B_v);
    lp=logposterior(alpha,beta_vec,B,sigsq,zeta,tau,xi,price,X,B_m,B_v);
    if((i+1)>burn & (i+1)%thin==0){
      output_mat(ii,0)=alpha;
      output_mat(ii,1)=B;
      output_mat(ii,2)=sigsq;
      output_mat(ii,3)=lp;
      for(int j=0; j<k; j++){
        output_mat(ii,(j+4))=beta_vec[j];
      }
      ii++;
    }
  }
  return output_mat;
}
