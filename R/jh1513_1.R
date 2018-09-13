################################################
##### 5-layer Neural Network by Simple Backpropgation
#################################################
##### A 5-layer neural network is trained by simple backpropation.
rm(list=ls())
####################################
RIDGE <- 0                    ##### Ridge ###########
LASSO <- 1                    ##### Lasso ###########
HYPERPARAMETER1 <- 0.00001    ##### Hyperparameter ##########
HYPERPARAMETER2 <- 0.000001   ##### Hyperparameter ##########
#####################################################
if(RIDGE == 1) {
  fff <- function(a){return(HYPERPARAMETER1*a)}
} else if(LASSO == 1) {
  fff <- function(a){return(HYPERPARAMETER2*sign(a))}
} else {
  fff <- function(a){return(0)}
}
######################### Input:M, Output: N ########
PIX <- 5
M <- PIX*PIX
N <- 2   ##### Output units #####
####################################
n <- 2000             ################# Training set
ntest <- 2000         ################# Test set
xdata <- matrix(0,M,n)
ydata <- matrix(0,N,n)
xtest <- matrix(0,M,n)
ytest <- matrix(0,N,n)
#################### Training Data reading ########
A <- data.matrix( read.csv('wtnb/char_train.txt', sep=" ", header=F) )
A <- A[,1:25]
xdata <- t(A)
for(i in 1:n) {
  if(i < 1001) {
    ydata[,i] <- c(1,0)
  } else {
    ydata[,i] <- c(0,1)
  }
}
################### Test data
A <- data.matrix( read.csv('wtnb/char_test.txt', sep=" ", header=F) )
A <- A[,1:25]
xtest <- t(A)
for(i in 1:ntest) {
  if(i < 1001) {
    ytest[,i] <- c(1,0)
  } else {
    ytest[,i] <- c(0,1)
  }
}
#############################################################
#############################################################
#############################################################
#############################################################
####################### Training Record #####################
CYCLE <- 500
MODCYC <- 5
Err0 <- matrix(0,1,CYCLE/MODCYC)
Err1 <- matrix(0,1,CYCLE/MODCYC)
Err2 <- matrix(0,1,CYCLE/MODCYC)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
################# Neural NetworkX Architecture ###
##### M=H4 -> H3 -> H2 -> H1 -> H0=N
H0 <- N
H1 <- 4
H2 <- 6
H3 <- 8
H4 <- M
######################## Hyperparameters #####################
ETA0 <- 0.5
ALPHA <- 0.1
EPSILON <- 0.0001
######################## Neural Network Calculation ##########
sig <- function(t){return(1/(1+exp(-t)))}
out <- function(w,t,h){return(sig(w %*% h + t))}
#################### input, hidden, output ##############
h0 <- matrix(0,H0,1)
h1 <- matrix(0,H1,1)
h2 <- matrix(0,H2,1)
h3 <- matrix(0,H3,1)
h4 <- matrix(0,H4,1)
###################### Training Initialization ########
w0 <- 0.1*matrix(rnorm(H0*H1),H0,H1)
w1 <- 0.1*matrix(rnorm(H1*H2),H1,H2)
w2 <- 0.1*matrix(rnorm(H2*H3),H2,H3)
w3 <- 0.1*matrix(rnorm(H3*H4),H3,H4)
th0 <- 0.1*matrix(rnorm(N),N,1)
th1 <- 0.1*matrix(rnorm(H1),H1,1)
th2 <- 0.1*matrix(rnorm(H2),H2,1)
th3 <- 0.1*matrix(rnorm(H3),H3,1)
#################### Accelerator
dw0 <- matrix(0,H0,H1)
dw1 <- matrix(0,H1,H2)
dw2 <- matrix(0,H2,H3)
dw3 <- matrix(0,H3,H4)
dth0 <- matrix(0,H0,1)
dth1 <- matrix(0,H1,1)
dth2 <- matrix(0,H2,1)
dth3 <- matrix(0,H3,1)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#################### Backpropagation Learning ############
for (cycle in 0:(CYCLE-1)){
  ETA <- ETA0*CYCLE/(CYCLE+10*cycle)
  for(i in 1:n){
    ii <- floor(n/2) * ((i-1) %% 2) + (i+1) %/% 2
    h4 <- matrix(xdata[,ii])
    t  <- matrix(ydata[,ii])
    h3 <- out(w3,th3,h4)
    h2 <- out(w2,th2,h3)
    h1 <- out(w1,th1,h2)
    h0 <- out(w0,th0,h1)
    ##############################
    delta0 <- (h0-t)*(h0*(1-h0)+EPSILON)
    delta1 <- t(t(delta0) %*% w0) * (h1*(1-h1)+EPSILON)
    delta2 <- t(t(delta1) %*% w1) * (h2*(1-h2)+EPSILON)
    delta3 <- t(t(delta2) %*% w2) * (h3*(1-h3)+EPSILON)
    ################## gradient ###########
    dw0  <- -ETA*delta0 %*% t(h1) +ALPHA*dw0
    dth0 <- -ETA*delta0           +ALPHA*dth0
    dw1  <- -ETA*delta1 %*% t(h2) +ALPHA*dw1
    dth1 <- -ETA*delta1           +ALPHA*dth1
    dw2  <- -ETA*delta2 %*% t(h3) +ALPHA*dw2
    dth2 <- -ETA*delta2           +ALPHA*dth2
    dw3  <- -ETA*delta3 %*% t(h4) +ALPHA*dw3
    dth3 <- -ETA*delta3           +ALPHA*dth3
    ################### steepest descent ##########
    w0  <- w0+dw0-fff(w0)
    th0 <- th0+dth0-fff(th0)
    w1  <- w1+dw1-fff(w1)
    th1 <- th1+dth1-fff(th1)
    w2  <- w2+dw2-fff(w2)
    th2 <- th2+dth2-fff(th2)
    w3  <- w3+dw3-fff(w3)
    th3 <- th3+dth3-fff(th3)
  }
  ############## Calculation of Training and Generalization Errors ####
  if(cycle %% MODCYC == 0){
    Err0[cycle/MODCYC+1] <- cycle
    err1 <- 0
    for(i in 1:n){
      h4 <- matrix(xdata[,i])
      t  <- matrix(ydata[,i])
      h3 <- out(w3,th3,h4)
      h2 <- out(w2,th2,h3)
      h1 <- out(w1,th1,h2)
      h0 <- out(w0,th0,h1)
  	  err1 <- err1 + t(t-h0)%*%(t-h0)
    }
    Err1[cycle/MODCYC+1] <- err1/n
    err2 <- 0
    for( i in 1:ntest){
      h4 <- matrix(xtest[,i])
      t  <- matrix(ytest[,i])
      h3 <- out(w3,th3,h4)
      h2 <- out(w2,th2,h3)
      h1 <- out(w1,th1,h2)
      h0 <- out(w0,th0,h1)
  	  err2 <- err2 + t(t-h0)%*%(t-h0)
    }
    Err2[cycle/MODCYC+1] <- err2/ntest
    cat(sprintf('[%g] Training error <- %f, Test error <- %f\n',cycle,err1,err2))
 #   deep_see
  }
}
plot(Err0,Err1,col='blue',type='l',ylim=c(0,0.5),xlab="",ylab="")
par(new=T)
plot(Err0,Err2,col='red',type='l',ylim=c(0,0.5),xlab="",ylab="")
title('X: Training Cycle. Blue: Training Error, Red: Test Error.')
######################## Backpropagation End ################
#############################################################
#############################################################
#############################################################
#############################################################
########################## Trained  Data ###############################
counter1 <- 0
cat('Error in Train:')
for (i in 1:n){
  h4 <- matrix(xdata[,i])
  t  <- matrix(ydata[,i])
  h3 <- out(w3,th3,h4)
  h2 <- out(w2,th2,h3)
  h1 <- out(w1,th1,h2)
  o  <- out(w0,th0,h1)
  max1 <- max(o)
  maxarg1 <- which.max(o)
  max2 <- max(t)
  maxarg2 <- which.max(t)
  if(maxarg1 != maxarg2){
    cat(sprintf('%g ',i))
    counter1 <- counter1 + 1
  }
}
cat(sprintf('\n   Error/TRAINED = %g/%g = %.3f \n',counter1,n,counter1/n))
########################## Test Data ###############################
counter2 <- 0
cat('Error in Test:')
for(i in 1:ntest){
  h4 <- matrix(xtest[,i])
  t  <- matrix(ytest[,i])
  h3 <- out(w3,th3,h4)
  h2 <- out(w2,th2,h3)
  h1 <- out(w1,th1,h2)
  o  <- out(w0,th0,h1)
  max1 <- max(o)
  maxarg1 <- which.max(o)
  max2 <- max(t)
  maxarg2 <- which.max(t)
  if(maxarg1 != maxarg2){
    cat(sprintf('%g ',i))
    counter2 <- counter2 + 1
  }
}
cat(sprintf('\n   Error/TEST = %g/%g = %.3f \n',counter2,ntest,counter2/ntest))
### deep_see
