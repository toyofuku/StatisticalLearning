######  Restricted Boltzman machine
rm(list=ls())
##################################
A00 <- data.matrix( read.csv('wtnb/char_learn03.txt', sep=" ", header=F) )
A00 <- A00[,1:30]
##################################
##### 5 times Input data are made by adding noises
A0 <- 0.8 * A00 + 0.1
A <- rbind(A0,A0,A0,A0,A0)
n <- dim(A)[1]
N <- dim(A)[2]
A <- A + 0.1 * matrix(rnorm(n*N),n,N)
n0    <- dim(A0)[1]
Ntest <- dim(A0)[2]
##################################
### n  <-  number of input vectors
### N  <-  dimension of input
###################################
YOKO <- 5   ### Character size
CHARN <- n0  ### Number of Input Patterns
H <- 8     ### hidden variables
ETA <- 0.01 ### Coefficient of learning
CYCLE <- 500 ### cycle of learning
MCMC <- 20  ### MCMC process 1 5 20 50 100
MCMCTEST <- 50 ### MCMC for TEST
NSEE <- 10  ### Visible units in testy 5 10 15 20 25
NOISETEST <- 0.1
#################### sigmoid function ##################
sigmoid <- function(t){return(1/(1+exp(-t)))}
#################### Initial parameters #################
w   <- 0.2 * matrix(rnorm(H*N),H,N)
th1 <- 0.2 * matrix(rnorm(H),H,1)
th2 <- 0.2 * matrix(rnorm(N),N,1)
######################################
for (cycle in 1:CYCLE){
  for (i in 1:n){
    hid_given <- sigmoid(w %*% A[i,] + th1)
    S <- 0.5 * matrix(1, H+N,1)
    SUMS <- matrix(0, H+N,1)
    COVS <- matrix(0, H+N,H+N)
    for (mcmc in 1:MCMC){
      S[1:H, 1]         <- floor(sigmoid(  w  %*% S[(H+1):(H+N),1] + th1) + runif(H))
      S[(H+1):(H+N), 1] <- floor(sigmoid(t(w) %*% S[1:H,        1] + th2) + runif(N))
      SUMS <- SUMS + S
      COVS <- COVS + S %*% t(S)
    }
    SUMS <- SUMS/MCMC
    COVS <- COVS/MCMC
    ###########################################
    w   <- w   + ETA * (hid_given %*% A[i,] - COVS[1:H, (H+1):(H+N)])
    th1 <- th1 + ETA * (hid_given           - SUMS[1:H, 1])
    th2 <- th2 + ETA * (A[i,]               - SUMS[(H+1):(H+N), 1])
  }
}
################### Trained DATA ######################
par(mfrow=c(3,CHARN))

xx <- matrix(0,6,5)
for(i in 1:n0){
  for(j in 1:N){
    xx[(j-1) %/% YOKO + 1, (j-1) %% YOKO + 1] <- A[i,j]
  }
  image(255*(1-0.5*xx), col=gray.colors(255), axes = FALSE)
}
################# TEST DATA ###########
##################################
TESTA <- A00 + NOISETEST * matrix(rnorm(n0*N),n0,N)
for (i in 1:n0){
  for (j in (NSEE+1):N){
    TESTA[i,j] <- 0.5
  }
  #######################################
  for (j in 1:N){
    xx[(j-1) %/% YOKO + 1, (j-1) %% YOKO + 1] <- TESTA[i,j]
  }
  image(255*(1-0.5*xx), col=gray.colors(255), axes = FALSE)
}
################## Results of Test data #######################
cat('Hidden Unit Answer for Test\n')
for (i in 1:n0){
  S <- 0.5 * matrix(1, H+N,1)
  SUMS <- matrix(0, H+N,1)
  for (mcmc in 1:MCMCTEST){
    S[(H+1):(H+NSEE),1] <- t(TESTA[i,1:NSEE])
    S[1:H,           1] <- floor(sigmoid(  w  %*% S[(H+1):(H+N),1] + th1) + matrix(runif(H),H,1))
    S[(H+1):(H+N),   1] <- floor(sigmoid(t(w) %*% S[1:H,        1] + th2) + matrix(runif(N),N,1))
    SUMS <- SUMS + S
  }
  cat(sprintf('Pattern %g: ',i))
  for (ii in 1:H){
    cat(sprintf('%.2f ',SUMS[ii,1]/MCMCTEST))
  }
  cat('\n')
  for (j in 1:N){
    xx[(j-1) %/% YOKO + 1, (j-1) %% YOKO + 1] <- SUMS[H+j,1]/MCMCTEST
  }
  image(255*(1-0.5*xx), col=gray.colors(255), axes = FALSE)
}
