###  Variational Bayes of Normal Mixture ############################
rm(list=ls())
##close all
#############################################################################
K0 <- 3             ### True clusters
STDTRUE <- 0.3      ### True Standard deviation of each clusters
K <- 3              ### Components of learning clusters
STD <- 0.3          ### 0.1 0.2 0.3 0.4 0.5 ### Standard deviation in learning machine
########################################################################
NNN <- 100          ### Number of samples
KURIKAESHI <- 100   ### Number of recursive process
PRIORSIG <- 0.01    ### 1/PRIORSIG = Variance of Prior
PHI0 <- 0.5         ### Hyperparameter of mixture ratio : 3/2 Kazuho's critical point
####################### True mixture ratios ###############################
KP1 <- 0.2
KP2 <- 0.3
KP3 <- 1-KP1-KP2
############################### make samples ###########################
truecase <- 1
if(truecase==1){
  X0 <- matrix(c(0, 0, 1, 0, 1, 1),nrow=2,byrow=T)
}

if(truecase==2){
  X0 <- matrix(c(0, 0.0, 0.5, 0, 0.5, 0.0),nrow=2,byrow=T)
}
if(truecase==3){
  X0 <- matrix(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),nrow=2,byrow=T)
}

YP <- runif(NNN)
Y0 <- matrix(0,1,NNN)
for(i in 1:NNN){
  if(YP[i]>KP1+KP2){
    Y0[i] <- 3
  } else {
    if(YP[i]>=KP1)
      Y0[i] <- 2
    else
      Y0[i] <- 1
  }
}

XX <- STDTRUE * matrix(rnorm(2*NNN),2,NNN)
for(i in 1:NNN){
  XX[,i] <- XX[,i]+X0[,Y0[i]]
}
######################## make data end ##############
#####################################################
# digamma <- function(x){return((log(x) - 0.5/x - 1/(12*x*x)))}
###################################################
 ########## Initialize VB
 PHI <- NNN/K*matrix(1,1,K)
 ETA0 <- NNN/K*matrix(1,1,K)
 ETA1 <- NNN/K*(mean(XX[1,])+0.1*rnorm(K))
 ETA2 <- NNN/K*(mean(XX[2,])+0.1*rnorm(K))
 YYY <- matrix(0,K,NNN)
 MR <- matrix(0,1,K)
 ########## Recursive VB Start
 for(kuri in 1:KURIKAESHI){
   for(i in 1:NNN){
     DD1 <- ETA1/ETA0-XX[1,i]
     DD2 <- ETA2/ETA0-XX[2,i]
     DDD <- digamma(PHI)-digamma(NNN+3*PHI0)-1/ETA0-(DD1*DD1+DD2*DD2)/(2*STD*STD)
     YYY[,i] <- exp(DDD-max(DDD))/sum(exp(DDD-max(DDD)))
   }
   for(k in 1:K){
     PHI[k] <- PHI0+sum(YYY[k,])
     ETA0[k] <- PRIORSIG+sum(YYY[k,])
     ETA1[k] <- sum(YYY[k,] * XX[1,])
     ETA2[k] <- sum(YYY[k,] * XX[2,])
   }
 }
 #################Free Energy
library(logOfGamma)

 FF1 <- -sum(gammaln(PHI))
 FF2 <- sum(log(ETA0)-(ETA1*ETA1+ETA2*ETA2)/(2*STD*STD*ETA0))
 FF3 <- sum((XX[1,]*XX[1,]+XX[2,]*XX[2,])/(2*STD*STD)+log(STD*STD))
 SSS <- -sum(sum(YYY*log(YYY)))
 FreeEnergy <- FF1+FF2+FF3+SSS
 #################
 cat(sprintf('Free Energy=%.2f, Mixture Ratio=(',FreeEnergy))
 Y01 <- rep(0,K)
 Y02 <- rep(0,K)
 for(j in 1:K){
   MR[j] <- ETA0[j]/(NNN+PRIORSIG*K)
   Y01[j] <- ETA1[j]/ETA0[j]
   Y02[j] <- ETA2[j]/ETA0[j]
   cat(sprintf('%.2f ',MR[j]))
 }
 cat(')\n')
#######################################################
probgauss <- function(x,y,a,b,VA2){
  outer(x,y,
        function(x,y){
          return(exp(-((x-a)^2+(y-b)^2)/VA2)/sqrt(2*pi*VA2))
        })
}
####################### plot samples ##################################
par(mfrow=c(2,2))
plot(XX[1,],XX[2,],col='blue',pch=1,main='Samples',xlim=c(-1,2),ylim=c(-1,2))
#################### plot true and estimated #############
plot(X0[1,],X0[2,],col='red',pch=0,main='True:Red Squares,  Estimated:Brue +', xlim=c(-1,2), ylim=c(-1,2))
points(Y01,Y02,col='blue',pch=3, xlim=c(-1,2), ylim=c(-1,2))
########################################
va2 <- 2*STDTRUE*STDTRUE
x1 <- seq(-1,2,0.1)
y1 <- seq(-1,2,0.1)
zzz <- KP1*probgauss(x1,y1,X0[1,1],X0[2,1],va2) +
  KP2*probgauss(x1,y1,X0[1,2],X0[2,2],va2) +
  KP3*probgauss(x1,y1,X0[1,3],X0[2,3],va2)
persp(x1,y1,zzz)
title('True Probability Density Function')
###############################################################
va2 <- 2*STD*STD
zzz <- 0*x1
for(kk in 1:K){
  zzz <- zzz + MR[kk]*probgauss(x1,y1,Y01[kk],Y02[kk],va2)
}
persp(x1,y1,zzz)
title('Estimated Probability Density Function')
###############################################################
