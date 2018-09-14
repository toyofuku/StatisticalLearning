###################################################################
############  Neural Network for Classification ###################
###################################################################
rm(list=ls())
############## True Classfication Rule ############################
true_rule <- function(x1,x2){return(x2-x1^3+2*x1)}
#############################################################
######################## Hyperparameters #############
HYPERPARAMETER <- 0.0000      ##### 0.00000  0.00002
diffhyper <- function(a){return(HYPERPARAMETER*a)}
####################### Training Conditions #################
CYCLEONE <- 5  ########## 2 5 50
CYCLEALL <- 12*CYCLEONE   ### training cycles
####################### Neural Network Architecture ###########
N <- 1          ### output Units
H <- 8          ### hidden Units
M <- 2          ### input Units
####################### Training Parameters #################
ETA <- 0.8      ### gradient constant
ALPHA <- 0.3    ### accelerator
EPSILON <- 0.01 ### regularization
###################### Training Initialization ########
u <- 0.3*matrix(rnorm(M*H),N,H)   ### weight from hidden to output
w <- 0.3*matrix(rnorm(H*M),H,M)   ### weight from input to hidden
ph <- 0.3*matrix(rnorm(N),N,1)  ### bias of output
th <- 0.3*matrix(rnorm(H),H,1)  ### bias of hidden
du <- matrix(0,N,H)      ### gradient weight from hidden to output
dw <- matrix(0,H,M)      ### gradient weight from input to hidden
dph <- matrix(0,N,1)     ### gradient bias of output
dth <- matrix(0,H,1)     ### gradient bias of hidden
################## Neural Network Sigmoid Function ############
neuron <- function(u,ph,h)(1/(1+exp(-(u %*% h+ph))))
##################### Generate Training Data #####################
n <- 100     ##### Number of Training samples
xdata <- -4 * matrix(runif(2*n),2,n) + 2
ydata <- matrix(0,1,n)
ydata[1,] <- 0.01+0.98*(sign(true_rule(xdata[1,],xdata[2,]))+1)/2
#################### Test Points #######################
TESTXNUMBER <- 41
testx1 <- matrix(0,TESTXNUMBER,TESTXNUMBER)
testx2 <- matrix(0,TESTXNUMBER,TESTXNUMBER)
for (j in 1:TESTXNUMBER){
   for (k in 1:TESTXNUMBER){
    testx1[j,k] <- -(TESTXNUMBER-1)/20+4*(j-1)/(TESTXNUMBER-1)
    testx2[j,k] <- -(TESTXNUMBER-1)/20+4*(k-1)/(TESTXNUMBER-1)
   }
}
##################### Draw Training Samples ####################
layout(matrix(c(1,15,14,14,2,3,4,5,6,7,8,9,10,11,12,13),4,4,byrow=T))

xlim  <-  c(-2,2)
ylim  <-  c(-2,2)

plot(0,0,type='n',xlim=xlim,ylim=ylim)
for(i in 1:n){
   if(ydata[i]>0.5){
     points(xdata[1,i],xdata[2,i],col='red',pch=1,xlim=xlim,ylim=ylim)
   } else {
     points(xdata[1,i],xdata[2,i],col='blue',pch='*',xlim=xlim,ylim=ylim)
   }
}
title('Training data: Red:1, Blue:0')
#################### Backpropagation Learning ############
training_err <- matrix(0,CYCLEONE)
test_err <- matrix(0,CYCLEONE)
train_process <- matrix(0,CYCLEONE)

for(cycle in 1:CYCLEALL){
 training_e <- 0
 for(i in 1:n){
  x <- xdata[,i]
  t <- ydata[,i]
  h <- neuron(w,th,x)
  o <- neuron(u,ph,h)
  training_e <- training_e+(t-o)^2
  ################## delta calculation ############
  delta1 <- (o-t)*(o*(1-o)+EPSILON)
  delta2 <- t(t(delta1) %*% u) * (h*(1-h)+EPSILON)
  ################## gradient ###########
  du  <- delta1 %*% t(h) +ALPHA*du
  dph <- delta1          +ALPHA*dph
  dw  <- delta2 %*% t(x) +ALPHA*dw
  dth <- delta2          +ALPHA*dth
  ################### stochastic steepest descent ##########
  u  <- u  -ETA * du  - diffhyper(u)
  ph <- ph -ETA * dph
  w  <- w  -ETA * dw  - diffhyper(w)
  th <- th -ETA * dth
 }
 ########## Draw Trained Results ################
 xxx <- matrix(0,2,1)
 truey <- matrix(0,1,1)
 testy <- matrix(0,TESTXNUMBER,TESTXNUMBER)
 if(cycle %% CYCLEONE == 0){
  test_e <- 0
  for(j in 1:TESTXNUMBER){
   for(k in 1:TESTXNUMBER){
    xxx[1,1] <- testx1[j,k]
    xxx[2,1] <- testx2[j,k]
    truey[1,1] <- 0.01+0.98*(sign(true_rule(xxx[1,1],xxx[2,1]))+1)/2
    h <- neuron(w,th,xxx)
    output <- neuron(u,ph,h)
    testy[j,k] <- output
    test_e <- test_e+(output-truey)^2
   }
  }
  training_err[cycle/CYCLEONE] <- training_e/n
  test_err[cycle/CYCLEONE]     <- test_e/(TESTXNUMBER^2)
  contour(seq(-2,2,0.1),seq(-2,2,0.1), testy, nlevels=5)
  train_process[cycle/CYCLEONE] <- cycle
 }
#######################################################
}

#################### Training and Test Errors ##############
plot(train_process,training_err, col='blue',type='l',ylim=c(0,0.3),xlab="",ylab="")
par(new=T)
plot(train_process,test_err, col='red',type='l',ylim=c(0,0.3),xlab="",ylab="")
title('Training Cycle. Blue:Training Error, Red:Test Error')

#################### Trained Neural Network Output ########
plot(0,0,type='n',xlim=xlim,ylim=ylim)
for(i in 1:n){
   if(ydata[i]>0.5){
     points(xdata[1,i],xdata[2,i],col='red',pch=1,xlim=xlim,ylim=ylim)
   } else {
    points(xdata[1,i],xdata[2,i],col='blue',pch='*',xlim=xlim,ylim=ylim)
   }
}

for(j in 1:TESTXNUMBER){
   for(k in 1:TESTXNUMBER){
    xxx[1,1] <- testx1[j,k]
    xxx[2,1] <- testx2[j,k]
    h <- neuron(w,th,xxx)
    testy[j,k] <- neuron(u,ph,h)
   }
}
contour(seq(-2,2,0.1),seq(-2,2,0.1),testy, add=T, drawlabels=F, levels=c(0.5))

title('Trained Neural Network')
########## } ################
