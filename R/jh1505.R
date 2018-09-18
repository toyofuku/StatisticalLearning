rm(list=ls())
#############################################################
################# Neural Network for Character Recognition ###
#############################################################
#################### Character Image Reading ###########
PIX <- 5
#################### READ Training data ############################
cat('Training Characters.\n')
A <- data.matrix( read.csv('wtnb/data/jh1505_train.txt', sep=" ", header=F) )
A <- A[,1:25]

layout(matrix(c(1,2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20, 21,21,21,21,21,21,21,21,21,21, 22,23,24,25,26,27,28,29,30,31),4,10,byrow=T))

oo <- matrix(0, 5,5)
for(i in 1:10){
  for(j in 1:5){
    for(k in 1:5){
      oo[j,k] <- A[i,5*(j-1)+k]
    }
  }
  oo <- t(apply(oo,2,rev))
  image(floor(254*(1-oo)), xaxt="n", yaxt="n",col=gray.colors(255))
  title('0')
}
for(i in 201:210){
  for(j in 1:5){
    for(k in 1:5){
      oo[j,k] <- A[i,5*(j-1)+k]
    }
  }
  oo <- t(apply(oo,2,rev))
  image(floor(254*(1-oo)), xaxt="n", yaxt="n",col=gray.colors(255))
  title('6')
}

################# Neural Network  ##############################
M <- PIX*PIX
H <- 8   ##### Hidden units >=4 ##### 6 10
N <- 2   ##### Output units #####
CYCLE <- 500  ##### Training Cycle ########
n <- 400
ntest <- 400
################# RIDGE and LASSO ############################
RIDGE <- 0                    ##### Ridge ###########
LASSO <- 0                    ##### Lasso ###########
HYPERPARAMETER1 <- 0.00005      ##### Hyperparameter ##########
HYPERPARAMETER2 <- 0.00005      ##### Hyperparameter ##########
#####################################################
if(RIDGE==1){
  fff <- function(a){return(HYPERPARAMETER1*a)}
} else if(LASSO==1){
  fff <- function(a){return(HYPERPARAMETER2*sign(a))}
} else{
  fff <- function(a){return(0)}
}
#################### input, hidden, output ##############
x <- matrix(0,M,1)
h <- matrix(0,H,1)
o <- matrix(0,N,1)
xdata <- matrix(0,M,n)
ydata <- matrix(0,N,n)
xtest <- matrix(0,M,n)
ytest <- matrix(0,N,n)
######################## Neural Network Calculation ##########
sig <- function(t){return(1/(1+exp(-t)))}
hid <- function(w,th,x){return(sig(w %*% x + th))}
out <- function(u,ph,h){return(sig(u %*% h + ph))}
#################### Training Data reading ########
xdata <- t(A)
xdata <- xdata
for(i in 1:n){
  if(i < 201){
    ydata[,i] <- c(1,0)
  } else {
    ydata[,i] <- c(0,1)
  }
}
####################### Test data
cat('Training Characters.\n')
B <- data.matrix( read.csv('wtnb/data/jh1505_test.txt', sep=" ", header=F) )
B <- B[,1:25]
xtest <- t(B)
for(i in 1:ntest){
  if(i < 201){
    ytest[,i] <- c(1,0)
  } else {
    ytest[,i] <- c(0,1)
  }
}
####################### Training Conditions ######
MODCYC <- CYCLE/10
Err0 <- matrix(0,1,CYCLE/MODCYC)
Err1 <- matrix(0,1,CYCLE/MODCYC)
Err2 <- matrix(0,1,CYCLE/MODCYC)
###################### Training Initialization ########
u   <- 0.1*matrix(rnorm(N*H),N,H)
w   <- 0.1*matrix(rnorm(H*M),H,M)
ph  <- 0.1*matrix(rnorm(N),N,1)
th  <- 0.1*matrix(rnorm(H),H,1)
du  <- matrix(0,N,H)
dw  <- matrix(0,H,M)
dph <- matrix(0,N,1)
dth <- matrix(0,H,1)
ALPHA <- 0.3
EPSILON <- 0.001
#################### Backpropagation Learning ############
for(cycle in 0:(CYCLE-1)){
  ETA <- 0.8*CYCLE/(5*cycle+CYCLE)
  for(i in 1:n){
    ii <- (n/2) * ((i-1) %% 2) + (i+1) %/% 2
    x  <- xdata[,ii]
    t  <- ydata[,ii]
    h  <- hid(w,th,x)
    o  <- out(u,ph,h)
    ##############################
    delta1 <- (o-t)*(o*(1-o)+EPSILON)
    delta2 <- t(t(delta1) %*% u) * (h*(1-h)+EPSILON)
    ################## gradient ###########
    du  <- delta1 %*% t(h) +ALPHA*du
    dph <- delta1          +ALPHA*dph
    dw  <- delta2 %*% t(x) +ALPHA*dw
    dth <- delta2          +ALPHA*dth
    ################### steepest descent ##########
    u  <- u-ETA*du-fff(u)
    ph <- ph-ETA*dph
    w  <- w-ETA*dw-fff(w)
    th <- th-ETA*dth
  }
  ############## Calculation of Training and Generalization Errors ####
  if(cycle %% MODCYC ==0){
    Err0[cycle/MODCYC+1] <- cycle
    err1 <- 0
    for(i in 1:n){
      x <- xdata[,i]
      t <- ydata[,i]
 	    h <- hid(w,th,x)
 	    o <- out(u,ph,h)
 	    err1 <- err1 + t(t-o) %*% (t-o)
   }
   Err1[cycle/MODCYC+1] <- err1/n
   err2 <- 0
   for(i in 1:ntest){
     x <- xtest[,i]
     t <- ytest[,i]
	   h <- hid(w,th,x)
	   o <- out(u,ph,h)
	   err2 <- err2 + t(t-o) %*% (t-o)
   }
   Err2[cycle/MODCYC+1] <- err2/ntest
   cat(sprintf('[%g] Training error=%f, Test error=%f\n',cycle,err1,err2))
 }
}

plot(Err0,Err1,col='blue',type='l',ylim=c(0,0.2),ylab="")
par(new=T)
plot(Err0,Err2,col='red',type='l',ylim=c(0,0.2),ylab="")
title('X: Training Cycle. Blue: Training Error, Red: Test Error.')
########################## Trained  Data ###############################
counter <- 0
cat('Error in Train:')
for(i in 1:n){
    x <- xdata[,i]
    t <- ydata[,i]
    h <- hid(w,th,x)
    o <- out(u,ph,h)
    max1    <- max(o)
    maxarg1 <- which.max(o)
    max2    <- max(t)
    maxarg2 <- which.max(t)
    if(maxarg1 != maxarg2){
	    cat(sprintf('%g ',i))
      counter <- counter+1
    }
}
cat(sprintf('\n   Error/TRAINED <- %g/%g <- %.3f \n',counter,n,counter/n))
########################## Test Data ###############################
counter<-0
cat('Error in Test:')
for(i in 1:ntest){
    x <- xtest[,i]
    t <- ytest[,i]
    h <- hid(w,th,x)
    o <- out(u,ph,h)
    max1    <- max(o)
    maxarg1 <- which.max(o)
    max2    <- max(t)
    maxarg2 <- which.max(t)
    if(maxarg1 != maxarg2){
      cat(sprintf('%g ',i))
      counter <- counter+1
    }
}
cat(sprintf('\n   Error/TEST=%g/%g=%.3f \n',counter,ntest,counter/ntest))
############################# From  Hidden to output ######################################
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

for(i in 1:2){
  hh <- matrix(0,2,H/2)
  for(j in 1:2){
    for(k in 1:(H/2)){
      hh[j,k] <- 1/(1+exp(-u[i,(H/2)*(j-1)+k]))
    }
  }
  hh <- t(apply(hh,2,rev))
  image(254*hh, xaxt="n", yaxt="n") #,col=jet.colors(255))
  title('output <- hidden')
}
############################# From Input to Hidden ######################################
for(i in 1:H){
  oo <- matrix(0,PIX,PIX)
  for(j in 1:PIX){
    for(k in 1:PIX){
      oo[j,k] <- 1/(1+exp(-w[i,PIX*(j-1)+k]))
    }
  }
  oo <- t(apply(oo,2,rev))
  image(254*oo, xaxt="n", yaxt="n") #,col=jet.colors(255))
  title('hidden <- input')
}
############################### END ######################
