################################################
##### Convolution neural network
##### for time series prediction
#################################################
rm(list=ls())
###################################################
###################################################
###################################################
###################################################
CONV <- 0           ##### Convolution Network 0 1 2 (0: all)
LinearPred <- 1     ##### Initialize by linear prediction 0 1
RIDGE <- 0          ##### Ridge 0 1
LASSO <- 0          ##### Lasso 0 1
###################################################
###################################################
################## Training Cycles ################
CYCLE <- 2000
MODCYC <- 10
Err0 <- matrix(0,1,CYCLE/MODCYC)
Err1 <- matrix(0,1,CYCLE/MODCYC)
###################################################
###################################################
####################### DATA READING ##############
AA <- data.matrix( read.csv('wtnb/hakusai.txt', sep="\t", header=F) )
###  http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
###################################################
###################################################
T1 <- dim(AA)[1]
T2 <- dim(AA)[2]
AAA <- AA[,2]
meanval <- mean(AAA)
maxval <- max(abs(AAA-meanval))
AAA <- 0.49 * (AAA-meanval)/maxval+0.5
ZZZ <- 27
alldata <- T1-(ZZZ+1)
ntrain <- alldata %/% 2
ntest <- alldata - ntrain
ydata <- matrix(0,1,alldata)
xdata <- matrix(0,ZZZ,alldata)
for(i in 1:alldata){
  ydata[1,i] <- AAA[i+ZZZ]
  xdata[,i]  <- AAA[i:(i+ZZZ-1)]
}
###################################################
###################################################
###################################################
############### RIDGE, LASSO, Network #############
###################################################
HYPERPARAMETER1 <- 0.000004     ##### RIDGE Hyperparameter ##########
HYPERPARAMETER2 <- 0.000002      ##### LASSO Hyperparameter ##########
if(RIDGE==1){
  fff <- function(a){return(HYPERPARAMETER1*a)}
} else if(LASSO==1){
  fff <- function(a){return(HYPERPARAMETER2*sign(a))}
} else{
  fff <- function(a){return(0)}
}
################# Neural Network Architecture ###
##### M=H3 -> H2 -> H1 -> H0=N
N <- 1
H0 <- 1
H1 <- 3
H2 <- 9
H3 <- ZZZ
######################## Training Paramaters #################
ETA0 <- 0.5
ALPHA <- 0.1
EPSILON <- 0.01
######################## Neural Network Calculation ##########
sig <- function(t){return(1/(1+exp(-t)))}
out <- function(w,t,h){return(sig(w %*% h + t))}
#################### input, hidden, output ##############
h0 <- matrix(0,H0,1)
h1 <- matrix(0,H1,1)
h2 <- matrix(0,H2,1)
###################### Training Initialization ########
w0 <- 0.1*matrix(rnorm(H0,H1),H0,H1)
w1 <- 0.1*matrix(rnorm(H1*H2),H1,H2)
w2 <- 0.1*matrix(rnorm(H2*H3),H2,H3)
th0 <- 0.01*matrix(rnorm(H0),H0,1)
th1 <- 0.01*matrix(rnorm(H1),H1,1)
th2 <- 0.01*matrix(rnorm(H2),H2,1)
#############################################################
#############################################################
#############################################################
#############################################################
########### Initial weight is determined by linear prediction
############################################################
if(LinearPred==1){
  xtr   <- xdata[,1:ntrain]
  ytr   <- matrix(ydata[1, 1:ntrain])
  xtest <- xdata[,(ntrain+1):alldata]
  ytest <- matrix(ydata[1,(ntrain+1):alldata])
  syx   <- xtr %*% ytr
  sxx   <- xtr %*% t(xtr)
  wlinear <- solve(sxx) %*% syx
  for(i in 1:H2){
    for(j in 1:H3){
      if((j+2) %/% 3 ==i){
        w2[i,j] = tanh(5*wlinear[j,1])
      }
    }
  }
  Tlinear <- sum((t(ytr)   - t(wlinear) %*% xtr)^2)
  Glinear <- sum((t(ytest) - t(wlinear) %*% xtest)^2)
}
#############################################################
#############################################################
#############################################################
#############################################################
### Convolution network
### CONV==0 --> all weights are used. Not convolution
### CONV==1 --> Convoltion (-1,0,1)
### CONV==2 --> Convolution (-2,-1,0,1,2)

if(CONV==0){
  wmask1 <- matrix(1, H1,H2)
  wmask2 <- matrix(1, H2,H3)
} else {
  wmask1 <- matrix(0, H1,H2)
  wmask2 <- matrix(0, H2,H3)
  for(i in 1:H1){
    for(j in 1:H2){
      if(abs(3*i-j-1) < CONV+1){
        wmask1[i,j] <- 1
      }
    }
  }
  for(i in 1:H2){
    for(j in 1:H3){
      if(abs(3*i-j-1) < CONV+1){
        wmask2[i,j] <- 1
      }
    }
  }
}

#################### Accelerator
dw0 <- matrix(0,H0,H1)
dw1 <- matrix(0,H1,H2)
dw2 <- matrix(0,H2,H3)
dth0 <- matrix(0,H0,1)
dth1 <- matrix(0,H1,1)
dth2 <- matrix(0,H2,1)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
Err0 <- Err1 <- Err2 <- rep(0,MODCYC)
#################### Backpropagation Learning ############
for(cycle in 0:(CYCLE-1)){
  ETA <- ETA0*sqrt(100/(100+cycle))
  for (i in 1:ntrain){
    h3 <- matrix(xdata[,i])
    t <- matrix(ydata[,i])
    w2 <- wmask2 * w2
    w1 <- wmask1 * w1
    h2 <- out(w2,th2,h3)
    h1 <- out(w1,th1,h2)
    h0 <- out(w0,th0,h1)
    ##############################
    delta0 <- (h0-t) * (h0 * (1-h0)+EPSILON)
    delta1 <- t(t(delta0) %*% w0) * (h1 * (1-h1)+EPSILON)
    delta2 <- t(t(delta1) %*% w1) * (h2 * (1-h2)+EPSILON)
    ################## gradient ###########
    dw0 <- -ETA*delta0 %*% t(h1)+ALPHA*dw0
    dth0 <- -ETA*delta0+ALPHA*dth0
    dw1 <- -ETA*delta1 %*% t(h2)+ALPHA*dw1
    dth1 <- -ETA*delta1+ALPHA*dth1
    dw2 <- -ETA*delta2 %*% t(h3)+ALPHA*dw2
    dth2 <- -ETA*delta2+ALPHA*dth2
    ################### steepest descent ##########
    w0 <- w0+dw0-fff(w0)
    th0 <- th0+dth0-fff(th0)
    w1 <- w1+dw1-fff(w1)
    th1 <- th1+dth1-fff(th1)
    w2 <- w2+dw2-fff(w2)
    th2 <- th2+dth2-fff(th2)
    w2 <- wmask2 * w2
    w1 <- wmask1 * w1
  }
 ############## Calculation of Training and Generalization Errors ####
 if(cycle %% MODCYC==0){
   Err0[cycle/MODCYC+1] <- cycle
   err1 <- 0
   for(i in 1:ntrain){
     ii <- (i%% 2)*(ntrain %/% 2)+((i+1) %/% 2)
     h3 <- matrix(xdata[,ii])
     t <- matrix(ydata[1,ii])
     h2 <- out(w2,th2,h3)
     h1 <- out(w1,th1,h2)
     h0 <- out(w0,th0,h1)
     err1 <- err1+ t(t-h0) %*% (t-h0)
   }
   Err1[cycle/MODCYC+1] <- err1/ntrain
   err2 <- 0
   for(i in (ntrain+1):alldata){
     h3 <- matrix(xdata[,i])
     t <- matrix(ydata[1,i])
     h2 <- out(w2,th2,h3)
     h1 <- out(w1,th1,h2)
     h0 <- out(w0,th0,h1)
     err2 <- err2+ t(t-h0) %*% (t-h0)
   }
   Err2[cycle/MODCYC+1] <- err2/ntest
   cat(sprintf('[%g] Training error=%f, Test error=%f\n',cycle,err1,err2))
 }
}
######################## Backpropagation End ################
#############################################################
#############################################################
#############################################################
if(LinearPred==1){
  cat(sprintf('[Linear Pred]:Training Err=%f, Test Err=%f\n',Tlinear,Glinear))
}
#############################################################
#############################################################

# par(mfrow=c(4,1))
layout(matrix(c(1,1,4,2,2,4,3,3,4),3,3,byrow=T))
###### Draw Results 1 : Training and Generalization Errors

plot(Err0,Err1,col='blue',type='l',ylim=c(0,0.032),xlab="",ylab="")
par(new=T)
plot(Err0,Err2,col='red',type='l',ylim=c(0,0.032),xlab="",ylab="")
title('X: Training Cycle. Blue: Training Error, Red: Test Error.')

#############################################################
#############################################################
true1 <- rep(0,ntrain)
ans1 <- rep(0,ntrain)
CCC1 <- rep(0,ntrain)
true2 <- rep(0,ntest)
ans2 <- rep(0,ntest)
CCC2 <- rep(0,ntest)
###### Draw Results 2 : Training and Unknwon Time Series Predictions

for(i in 1:ntrain){
  CCC1[i] <- i
  h3 <- matrix(xdata[,i])
  h2 <- out(w2,th2,h3)
  h1 <- out(w1,th1,h2)
  h0 <- out(w0,th0,h1)
  true1[i] <- ydata[i]
  ans1[i] <- h0[1]
}

for(i in 1:ntest){
  CCC2[i] <- i
  h3 <- matrix(xdata[,i+ntrain])
  h2 <- out(w2,th2,h3)
  h1 <- out(w1,th1,h2)
  h0 <- out(w0,th0,h1)
  true2[i] <- ydata[i+ntrain]
  ans2[i] <- h0[1]
}

#subplot(2,1,1)
plot(CCC1,true1,col='red',type='l',ylim=c(0,1),xlab="",ylab="")
par(new=T)
plot(CCC1,ans1,col='blue',type='l',ylim=c(0,1),xlab="",ylab="")
title('Trained Data: red:true, blue:predicttion')

#subplot(2,1,2)
plot(CCC2,true2,col='red',type='l',ylim=c(0,1),xlab="",ylab="")
par(new=T)
plot(CCC2,ans2,col='blue',type='l',ylim=c(0,1),xlab="",ylab="")
title('Unknown: red:true, blue:predicttion')

#############################################################
#############################################################
###### Draw Results 3 : Trained Neural Networks

Y0 <- 3
Y1 <- 2
Y2 <- 1
Y3 <- 0

px <- c(H3/2,0)
py <- c(Y0, Y1)

plot(px,py,type='n',xlim=c(0,ZZZ),ylim=c(-0.1,3.1),xlab="",ylab="",axes=F)

for(j in 1:H1){
  px[2] <- H2*(j-0.5)
  bold <- floor((abs(w0[1,j])+0.9))
  if(bold > 0){
    if(w0[1,j] > 0){
      lines(px,py,col='red')
    } else{
      lines(px,py,col='blue')
    }
  }
}
points(px[1],Y0,col='black')

py <- c(Y1, Y2)
for(i in 1:H1){
  px[1] <- H2*(i-0.5)
  for(j in 1:H2){
    px[2] <- H1*(j-0.5)
    bold <- floor(abs(w1[i,j])+0.9)
    if(bold > 0){
      if(w1[i,j] > 0){
        lines(px,py,col='red')
      } else {
        lines(px,py,col='blue')
      }
    }
  }
  points(px[1],2,col='black')
}

py <- c(Y2, Y3)
for(i in 1:H2){
  px[1] <- H1*(i-0.5)
  for(j in 1:H3){
    px[2] <- (j-0.5)
    bold <- floor(abs(w2[i,j])+0.9)
    if(bold > 0){
      if(w2[i,j] > 0){
        lines(px,py,col='red')
      } else {
        lines(px,py,col='blue')
      }
    }
  }
  points(px[1],Y2,col='black')
}

for(i in 1:H3){
  px[1] <- (i-0.5)
  points(px[1],Y3,col='black')
}

# set(prp,'Color', [0,0,0], 'MarkerFaceColor', [1,1,1],'MarkerSize',8, 'LineWidth', 2)

title('Trained Network. Input: 27 months ago ---> Last month')

#######################################
