#library(ncvreg)
library(glmnet)
library(SIS)
#input design matrix X
set.seed(100)
real_data<-read.csv("~/Desktop/pcr_sparse_regression/Real_Data_Final/real_data/realdata_189_126.csv")
X<-real_data
X<-(X[,2:ncol(X)])
X<-X[,-c(48)]
X<-data.frame(X)
n=nrow(X)
p=ncol(X)-1
####normalize X######
X = t(t(X)-colMeans(X))
X = t(t(X)/apply(X,2,sd))

index<-c()
p_value<-c()
  i=81 #GS5 let #i=48 denotes the HOUSTNE
  Y=X[,i]
  X1=X[,-i]

  cov<-X1%*%t(X1)/n
  eigvec<-eigen(cov)$vectors
  eigval<-eigen(cov)$values
  K_est<-max(which.min(diff(log(eigval[1:10]))),2) #select K
  #print(K_est)
  hatf<-sqrt(n)*eigvec[,1:K_est]
  hatB<-t(1/n*t(hatf)%*%X1)
  hatU<-X1-hatf%*%t(hatB)
  
  C<-as.matrix(diag(rep(1,p)))
  T<-c()
  for( j in 1:ncol(hatU)){
    fit_u = glmnet(hatU[1:124,-j], hatU[1:124,j], intercept=FALSE,
                   lambda=cv.glmnet(hatU[1:124,-j], hatU[1:124,j],intercept=FALSE)$lambda.1se)
    beta<-as.vector(fit_u$beta)
    C[j,-j]<--beta
    T<-c(T,1/n*sum((hatU[,j]-hatU[,-j]%*%beta)^2)+fit_u$lambda/2*sum(abs(beta)))
    if (j%%10==0){
      print(j)
    }
  }
  T1<-diag(1/T)
  Theta<-T1%*%C  ####Estimate Theta####
    boostrap<-c() ###Conduct Boostrap####
    for(j in 1:2000){
      boostrap<-c(boostrap,max(abs(Theta%*%t(hatU)%*%rnorm(n,mean=0,sd=1))))
    }
    c_alpha<-1/sqrt(n)*quantile(boostrap,0.95) #cutoff value
  
  Idx = 1:(n/2)
  X0 = X1[Idx,]; X2 = X1[-Idx,]
  Sigma0 = tcrossprod(X0)*2/n; Sigma2 = tcrossprod(X2)*2/n
  eigen0 = eigen(Sigma0); eigen2 = eigen(Sigma2)
  eigvec0 = eigen0$vectors; eigvalue0 = eigen0$values
  eigvec2 = eigen2$vectors; eigvalue2 = eigen2$values
  K0 = max(which.min(diff(log(eigvalue0[1:10]))),2)
  K2 = max(which.min(diff(log(eigvalue2[1:10]))),2)
  F.hat0 = eigvec0[,1:K0]*sqrt(n/2); F.hat2 = eigvec2[,1:K2]*sqrt(n/2)
  B.hat0.T = t(F.hat0)%*%X0*2/n; B.hat2.T = t(F.hat2)%*%X2*2/n
  U.hat0 = X0 - F.hat0%*%B.hat0.T; U.hat2 = X2 - F.hat2%*%B.hat2.T
  
  rcv<-function(Y){ ###function for refitting cross_validation  to estimate $\hat\sigma$
    Y0 = Y[Idx]; Y2 = Y[-Idx]
    
    tmp0 = tcrossprod(F.hat0); tmp2 = tcrossprod(F.hat2)
    Y0.new = Y0 - tmp0%*%Y0*2/n
    Y2.new = Y2 - tmp2%*%Y2*2/n
    SIS0 = SIS(U.hat0,Y0.new)
    SIS2 = SIS(U.hat2,Y2.new)
    S.hat0.ISIS = SIS0$ix; S.hat0.SIS = SIS0$ix0
    S.hat2.ISIS = SIS2$ix; S.hat2.SIS = SIS2$ix0 
    X0=cbind(F.hat0,U.hat0)
    X2=cbind(F.hat2,U.hat2)
    X0.ISIS = X0[,c(1:K0,S.hat2.ISIS+K0)]
    #print(c(1:K2,S.hat2.ISIS+K2))
    X0.SIS = X0[,c(1:K0,K0+S.hat2.SIS)]
    X2.ISIS = X2[,c(1:K2,K2+S.hat0.ISIS)]; X2.SIS = X2[,c(1:K2,K2+S.hat0.SIS)]
    lm0.ISIS = lm(Y0~X0.ISIS-1) 
    lm0.SIS = lm(Y0~X0.SIS-1)
    lm2.ISIS = lm(Y2~X2.ISIS-1) 
    lm2.SIS = lm(Y2~X2.SIS-1)
    Q0.ISIS.H1 = sum((resid(lm0.ISIS))^2)
    sigma.hat0.ISIS = Q0.ISIS.H1/(n/2 - K0 - length(S.hat2.ISIS))
    Q0.SIS.H1 = sum((resid(lm0.SIS))^2)
    sigma.hat0.SIS = Q0.SIS.H1/(n/2 - K0 - length(S.hat2.SIS))
    Q2.ISIS.H1 = sum((resid(lm2.ISIS))^2)
    sigma.hat2.ISIS = Q2.ISIS.H1/(n/2 - K2 - length(S.hat0.ISIS))
    Q2.SIS.H1 = sum((resid(lm2.SIS))^2)
    sigma.hat2.SIS = Q2.SIS.H1/(n/2 - K2 - length(S.hat0.SIS))
    sigma.hat.ISIS = mean(c(sigma.hat0.ISIS, sigma.hat2.ISIS))
    sigma.hat.SIS = mean(c(sigma.hat0.SIS, sigma.hat2.SIS))
    c(sigma.hat.ISIS,sigma.hat.SIS)}
  sd_alter1<-rcv(Y)
  Y_new1<-Y-1/n*hatf%*%t(hatf)%*%Y
  fit1 = glmnet(hatU, Y_new1, intercept=FALSE,
                lambda=cv.glmnet(hatU,Y_new1,intercept=FALSE)$lambda.min)
  lambda_beta1<-fit1$lambda
  beta_hat1<-as.vector(fit1$beta) #fitted value
  d_beta1<-beta_hat1+1/n*Theta%*%t(hatU)%*%(Y-hatU%*%beta_hat1) #debiased beta under the alternative
  index<-c(index,sum(sqrt(n)*max(abs(d_beta1))/sqrt(sd_alter1[2])>c_alpha))
  p_value<-c(p_value,sum(1/sqrt(n)*boostrap>((sqrt(2*n)*max(abs(d_beta1))/sqrt(sd_alter1[2]))))/2000)
  print(p_value) #Report p-value
  #write.csv(index,"index.csv")
#write.csv(p_value,"~/Desktop/115_126_pcr_p_value.csv")

