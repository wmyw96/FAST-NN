library(SIS)
library(MASS)
#input design matrix X
#pick n
set.seed(1000)
real_data<-read.csv("~/Desktop/pcr_sparse_regression/Real_Data_Final/real_data/realdata_189_126.csv")
X<-data.frame(real_data)
X=X[,-1]
X<-as.matrix(X)
X<-X[,-c(81,83)]
n=nrow(X)
p=ncol(X)-1
#normalize X
X = t(t(X)-colMeans(X))
X = t(t(X)/apply(X,2,sd))

#Y1 responses, method:1==ISIS;2==SIS
selectfeature<-function(Y1,X){
  cov<-(X[1:floor(n^(0.8)),]-rowMeans(X[1:floor(n^(0.8)),]))%*%t((X[1:floor(n^(0.8)),]-rowMeans(X[1:floor(n^(0.8)),])))/floor(n^(0.8)) #construct factors
  eigvec<-eigen(cov)$vectors
  eigval<-eigen(cov)$values
  K1<-max(which.min(diff(log(eigval[1:10]))),2) #select K
  hatf<-sqrt(floor(n^(0.8)))*eigvec[,1:K1] #hatF
  hatB<-t(1/floor(n^(0.8))*t(hatf)%*%X[1:floor(n^(0.8)),]) #hat B
  hatU<-X[1:floor(n^(0.8)),]-hatf%*%t(hatB) #hat U
  Y_new<-Y1[1:floor(n^(0.8))]-1/floor(n^(0.8))*hatf%*%t(hatf)%*%Y1[1:floor(n^(0.8))] #
  sis<-SIS(hatU,Y_new)
  #select1<-sis$ix #ISIS
  #S.hat0.ISIS = SIS0$ix; S.hat0.SIS = SIS0$ix0
  cov2<-(X[(floor(n^(0.8))+1):n,]-rowMeans(X[(floor(n^(0.8))+1):n,]))%*%t(X[(floor(n^(0.8))+1):n,]-rowMeans(X[(floor(n^(0.8))+1):n,]))/(n-(floor(n^(0.8))))
  eigvec2<-eigen(cov2)$vectors
  eigval2<-eigen(cov2)$values
  K<-max(which.min(diff(log(eigval2[1:10]))),2) #select K
  hatf2<-sqrt(n-(floor(n^(0.8))))*eigvec2[,1:K] #hat F using second half of data
  hatB2<-t(1/(n-floor(n^(0.8)))*t(hatf2)%*%X[(floor(n^(0.8))+1):n,]) #hat B
  hatU2<-X[(floor(n^(0.8))+1):n,]-hatf2%*%t(hatB2) #hat U
  P_f<-hatf2%*%t(hatf2)/(n-(floor(n^(0.8))))
  select2<-sis$ix0 #SIS
  P_x<-X[(floor(n^(0.8))+1):n,select2]%*%solve(t(X[(floor(n^(0.8))+1):n,select2])%*%(X[(floor(n^(0.8))+1):n,select2]))%*%t(X[(floor(n^(0.8))+1):n,select2])
  P_u<-hatU2[,select2]%*%solve(t(hatU2[,select2])%*%hatU2[,select2])%*%t(hatU2[,select2])
  Q2<-t(Y1[(floor(n^(0.8))+1):n])%*%(P_f+P_u-P_x)%*%(P_f+P_u-P_x)%*%Y1[(floor(n^(0.8))+1):n]
  c(Q2,K)
}
rcv<-function(Y1,X){#refitted cross-validation to compute $\sigma$
  Idx1=1:(n/2)
  Idx2=(n/2+1):n
  lis=list(Idx1,Idx2)
  var_ISIS<-c()
  var_SIS<-c()
  for (Idx in lis){
    cov<-2*(X[Idx,]-rowMeans(X[Idx,]))%*%t((X[Idx,]-rowMeans(X[Idx,])))/n #construct factors
    eigvec<-eigen(cov)$vectors
    eigval<-eigen(cov)$values
    K1<-max(which.min(diff(log(eigval[1:10]))),1) #select K
    hatf<-sqrt(n/2)*eigvec[,1:K1] #hatF
    hatB<-t(2/n*t(hatf)%*%X[Idx,]) #hat B
    hatU<-X[Idx,]-hatf%*%t(hatB) #hat U
    Y_new<-Y1[Idx]-2/n*hatf%*%t(hatf)%*%Y1[Idx] #
    sis<-SIS(hatU,Y_new)
    select1<-sis$ix #ISIS  
    select2<-sis$ix0 #SIS
    cov2<-(X[-Idx,]-rowMeans(X[-Idx,]))%*%t(X[-Idx,]-rowMeans(X[-Idx,]))/(n/2)
    eigvec2<-eigen(cov2)$vectors
    eigval2<-eigen(cov2)$values
    K<-max(which.min(diff(log(eigval2[1:10]))),2) #select K
    hatf2<-sqrt(n/2)*eigvec2[,1:K] #hat F using second half of data
    hatB2<-t(2/n*t(hatf2)%*%X[-Idx,]) #hat B
    hatU2<-X[-Idx,]-hatf2%*%t(hatB2) #hat U  
    X0=cbind(hatf2,hatU2)
    X0.ISIS = X0[,c(1:K,select1+K)]
    X0.SIS = X0[,c(1:K,select2+K)]
    lm0.ISIS = lm(Y1[-Idx]~X0.ISIS-1) 
    lm0.SIS = lm(Y1[-Idx]~X0.SIS-1)
    Q0.ISIS.H1 = sum((resid(lm0.ISIS))^2)
    sigma.hat0.ISIS = Q0.ISIS.H1/((n/2) - K - length(select1))
    Q0.SIS.H1 = sum((resid(lm0.SIS))^2)
    sigma.hat0.SIS = Q0.SIS.H1/((n/2) - K - length(select2))
    #X2=cbind(F.hat2,U.hat2)
    var_ISIS<-c(var_ISIS,sigma.hat0.ISIS)
    var_SIS<-c(var_SIS,sigma.hat0.SIS)
  }
  print(var_ISIS)
  print(var_SIS)
  c(mean(var_ISIS),mean(var_SIS))
}


index<-c()
p_value<-c()

i=81 #Choose i=49 represent HOUSENE
Y1=X[,i]
X1=X[,-i]
output1<-try(selectfeature(Y1,X1),silent=TRUE)
if ('try-error' %in% class(output1)) {
  output1<-c(0,1)}
sig_est<-try(rcv(Y1,X1),silent=TRUE)
var_est1<-sig_est[1]
quant_chisq<-c()
for (q in 1:10){
  boostrap<-c()
  for(j in 1:200){
    boostrap<-c(boostrap,rchisq(1,output1[2]))
  }
  quant_chisq<-c(quant_chisq,quantile(boostrap,0.95))
}
c_alpha<-mean(quant_chisq) #Compute cutoff value

est<-output1[1]/var_est1
index<-c(index,sum(est>c_alpha))
p_value<-c(p_value,1-pchisq(est,output1[2]))
print(p_value) #output p_value
#write.csv(p_value,"~/Desktop/240_126_sparse_p_value.csv")
