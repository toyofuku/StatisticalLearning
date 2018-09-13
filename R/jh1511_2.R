
####### Variational Bayesian Learning
####### True 5 components
####### Learner 3,4,5,6,7
rm(list=ls())
###############################################
STD0 <- 5
STD <- 5            ###  ### Standard deviation in learning machine
########################################################################
n <- 500
KURIKAESHI <- 200   ### Number of recursive process
### PRIORSIG <- 0.01    ### 1/PRIORSIG = Variance of Prior
PHI0 <- 3           ### Hyperparameter of mixture ratio : 3/2 Kazuho's critical point
ETA0 <- 1
####################### True mixture ratios ###############################

######### Set True
true_ratio <- c(0.1, 0.15, 0.25, 0.35, 0.15)
true_mean <- c(10, 30, 50, 70, 90)

########################### Sample generation
XX <- matrix(rep(0,n),1,n)

for(i in 1:n){
    r <- runif(1)
    if(r < sum(true_ratio[1]))
        y <- true_mean[1]
    else if(r < sum(true_ratio[1:2]))
        y <- true_mean[2]
    else if(r < sum(true_ratio[1:3]))
        y <- true_mean[3]
    else if(r < sum(true_ratio[1:4]))
        y <- true_mean[4]
    else
        y <- true_mean[5]
    XX[i] <- y + STD0 * rnorm(1)
}

par(mfrow=c(2,3))
hist(XX,50,xlim=c(0,100),main='Sample Histogram')

probgauss <- function(x,a,VA2){
  return(exp(-((x-a)^2)/VA2)/sqrt(2*pi*VA2))
}

library(logOfGamma)

#############################################################################
for(kur in 1:5){
#############################################################################
K <- kur + 2                 ### Components of learning clusters
######################## make data end ##############
#####################################################
 ###################################################
 ########## Initialize VB
 PHI <- n/K * matrix(1,1,K)
 ETA2 <- n/K * matrix(1,1,K)
 ETA1 <- (n/K)*100 * seq(1/K, 1, 1/K)
 YYY <- matrix(0,K,n)
 MR  <- matrix(0,1,K)
 Y01 <- matrix(0,1,K)
 ########## Recursive VB Start
 for(kuri in 1:KURIKAESHI){
   for(i in 1:n){
     DD1 <- ETA1 / ETA2 - XX[1,i]
     LIK <- digamma(PHI)-digamma(n+PHI0)-0.5/ETA2-(DD1*DD1)/(2*STD*STD)
     likmax <- max(LIK)
     YYY[,i] <- exp(LIK-likmax)/sum(exp(LIK-likmax))
   }
   for(k in 1:K){
     PHI[k] <- PHI0 + sum(YYY[k,])
     ETA1[k] <- sum(YYY[k,] * XX[1,])
     ETA2[k] <- ETA0 + sum(YYY[k,])
   }
 }
 #################Free Energy
 FF1 <- -sum(gammaln(PHI))+gammaln(sum(PHI))
 FF2 <- -K*gammaln(PHI0)+gammaln(K*PHI0)
 FF3 <- sum(0.5*log(ETA2)-(ETA1^2)/(2*STD^2*ETA2))
 FF4 <- K/2*log(ETA0)
 FF5 <- sum((XX[1,] * XX[1,])/(2*STD^2)) + 0.5*n*log(2*pi*STD^2)
 SSS <- sum(sum(YYY * log(YYY)))
 FreeEnergy <- FF1-FF2+FF3-FF4+FF5+SSS
 #################
 cat(sprintf('K=%g, Free Energy=%.2f, Mixture Ratio=(', K, FreeEnergy))
 for(j in 1:K){
   MR[j] <- ETA2[j]/(n+K*ETA0)
   Y01[j] <- ETA1[j]/ETA2[j]
   cat(sprintf('%.2f ', MR[j]))
 }
 cat(')\n')
#######################################################

####################### plot samples ##################################
x <- 1:100
va2 <- STD*STD

y <- matrix(0,1,100)
for(k in 1:K){
    y <- y + MR[k] * probgauss(x,Y01[k],va2)
}
plot(x,y,col='blue',type='o')

y <- matrix(0,1,100)
for(k in 1:5){
    y <- y + true_ratio[k] * probgauss(x,true_mean[k],va2)
}
lines(x,y,col='red',type='o')
title('Red: true, Blue: Estimated');
}

# for(k in 1:K){
#     y <- MR[k] * probgauss(x,Y01[k],va2)
#     plot(x,y,col='blue')
# }
#
# for(k in 1:5){
#     y <- true_ratio[k] * probgauss(x,true_mean[k],va2)
#     plot(x,y,col='red')
# }
# title('Red: true, Blue: Estimated')
# }
