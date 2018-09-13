###  Expectation Maximization ############################
rm(list=ls())
#############################################################################
K <- 9                 ### Components of learning clusters
########################################################################
n <- 360            ### Number of samples
KURIKAESHI <- 1000
############################### make samples ###########################

#for(j in 1:9){
#  X0[1,j] <- (j-1) %/% 3
#  X0[2,j] <- (j-1) %% 3
#}
#for(i in 1:n){
#  XX[,i] <- 0.3 * matrix(rnorm(2),2,1) + X0[,(i-1) %% 9 + 1]
#}

XX <- 2 * matrix(runif(2*n),2,n)

####################### plot samples ##################################
par(mfrow=c(3,4))

plot(XX[1,],XX[2,],col='black',pch=20,xlim=c(-0.5,2.5),ylim=c(-0.5,2.5))
title('Samples')
#######################################################
gauss <- function(x1,x2,b1,b2,s){return(exp(-((x1-b1)^2+(x2-b2)^2)/(2*s))/(2*pi*s))}
####################### plot samples ##################################
YYY <- matrix(0,K,n)
for(ttt in 1:11) {
  ########## Initialize
  aaa <- 1/K * matrix(1, 1,K)
  bbb1 <- mean(XX[1,]) + 0.2 * matrix(rnorm(K), 1,K)
  bbb2 <- mean(XX[2,]) + 0.2 * matrix(rnorm(K), 1,K)
  sigma <- (var(XX[,1])+var(XX[,2]))/2 * matrix(1, 1,K)
  ########## Recursive VB Start
  aaanew = 0 * aaa
  bbb1new = 0 * bbb1
  bbb2new = 0 * bbb2
  sigmanew = 0 * sigma
  for(kuri in 1:KURIKAESHI){
    for(i in 1:n){
      each <- aaa * gauss(XX[1,i],XX[2,i],bbb1,bbb2,sigma)
      wa <- sum(each)
      YYY[,i] <- each/wa
    }
    for(k in 1:K){
      sumy <- sum(YYY[k,])
      aaanew[k]   <- sumy/n
      bbb1new[k]  <- sum(YYY[k,] * XX[1,]) / sumy
      bbb2new[k]  <- sum(YYY[k,] * XX[2,]) / sumy
      sigmanew[k] <- sum(YYY[k,] * ((XX[1,]-bbb1[k])^2+(XX[2,]-bbb2[k])^2)) / (2*sumy)
    }
    aaa <- aaanew
    bbb1 <- bbb1new
    bbb2 <- bbb2new
    sigma <- sigmanew
  }
  ###############################################################
#  [x1,y1] <- meshgrid(-0.5:0.1:2.5,-0.5:0.1:2.5)
  x1 <- seq(-0.5,2.5,0.1)
  y1 <- seq(-0.5,2.5,0.1)
  zzz <- matrix(0,length(x1),length(y1))
  for(k in 1:K) {
#    zzz  <-  zzz + aaa[k] * gauss(x1,y1,bbb1[k],bbb2[k],sigma[k])
    zzz  <-  zzz + outer(x1,y1,function(x,y){return(aaa[k] * gauss(x,y,bbb1[k],bbb2[k],sigma[k]))})
  }
  loglike <- 0
  for(i in 1:n) {
    loglike <- loglike + sum(aaa*gauss(XX[1,i],XX[2,i],bbb1,bbb2,sigma))
  }

  ###############################################################
  cat(sprintf('%f\n',loglike))
  #################### plot true and estimated #############
  contour(x1,y1,zzz)
}
